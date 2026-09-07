"""Integration check: real SM-ALL, both protocols concurrently, tiny CPU data.

This checks execution and restart behavior, not NTU120 accuracy or T4 speed.
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
import numpy as np
from .data import prepare
from .io_utils import atomic_json
from .launch import validate_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    root = Path(args.outdir)
    root.mkdir(parents=True, exist_ok=True)
    annotations = []
    for i in range(5):
        x = np.zeros((2, 16+i, 25, 3), np.float32)
        x[0] = np.random.default_rng(i).normal(0, .1, x[0].shape) + [1, 2, 3]
        x[0, :, :, 0] += np.arange(16+i)[:, None] * .05
        annotations.append(dict(frame_dir=f'id{i}', keypoint=x, label=i))
    splits = dict(xsub_train=['id0', 'id1', 'id2'], xsub_val=['id3', 'id4'],
                  xset_train=['id1', 'id3', 'id4'], xset_val=['id0', 'id2'])
    dataset = root/'tiny.pkl'
    dataset.write_bytes(pickle.dumps(dict(annotations=annotations, split=splits)))
    prepare(dataset, root/'cache', root/'prepare_status.json')
    config = validate_config(dict(epochs=2, micro_batch=2, accumulation_steps=2, eval_batch=4))
    atomic_json(root/'config.json', config)
    env = dict(os.environ, JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[3])
    commands = {}
    processes, streams = [], []
    try:
        for protocol in ('xsub', 'xset'):
            cmd = [sys.executable, '-m', 'experiments.nestsar_sm_all_t16.streaming.worker',
                   '--protocol', protocol, '--cache', str(root/'cache'), '--outdir', str(root/'run'),
                   '--config', str(root/'config.json'), '--allow-cpu']
            commands[protocol] = cmd
            stream = (root/f'{protocol}.log').open('w')
            streams.append(stream)
            processes.append(subprocess.Popen(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT))
        for protocol, proc in zip(commands, processes):
            if proc.wait() != 0:
                raise RuntimeError((root/f'{protocol}.log').read_text()[-5000:])
        for protocol, cmd in commands.items():
            # Delete aliases only; the committed checkpoint reference must repair them.
            out = root/'run'/protocol
            (out/'best.msgpack').unlink()
            (out/'history.json').unlink()
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(result.stderr[-5000:])
            status = json.loads((out/'status.json').read_text())
            history = json.loads((out/'history.json').read_text())
            assert status['done'] and status['completed_epoch'] == 2
            assert status['best_epoch'] > 0 and len(history) == 2
            assert all(row['train_samples'] == 3 and row['val_samples'] == 2 for row in history)
            assert (out/'best.msgpack').exists()
        atomic_json(root/'smoke_report.json', dict(
            backend='cpu', real_model_params=1826556, protocols=['xsub', 'xset'],
            concurrent_workers=True, epochs_per_protocol=2, train_samples_per_protocol=3,
            val_samples_per_protocol=2, resume_and_alias_repair_passed=True,
            real_ntu_accuracy_measured=False, dual_t4_executed=False))
    finally:
        # CPU smoke workers do not own process groups, so terminate them directly.
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=10)
        for stream in streams:
            stream.close()


if __name__ == '__main__':
    main()
