"""Dual-T4 SM-ALL entrypoint with two parent-owned persistent TQDM bars."""
import argparse

from experiments.nestsar_sm_all_t16.streaming.launch import DEFAULTS, run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=None)
    p.add_argument("--outdir", default="/kaggle/working/NestSAR_SM_ALL_T16_SharedCache_v2")
    p.add_argument("--cache-dir", default="/kaggle/working/NestSAR_SM_ALL_SharedCache_v2")
    p.add_argument("--raw-layout", choices=("MTVC", "TMVC"), default="MTVC")
    p.add_argument("--audit-first", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--runtime-mode", choices=("host", "local-cuda"), default="host")
    p.add_argument("--batch-size", type=int, help="Effective batch; must divide evenly by --micro-batch")
    for key, value in DEFAULTS.items():
        names = ["--" + key.replace("_", "-")]
        if key == "eval_batch":
            names.append("--eval-batch-size")
        if key == "jitter_shift":
            names.append("--jitter-max-shift")
        if isinstance(value, bool):
            p.add_argument(*names, dest=key, action=argparse.BooleanOptionalAction, default=value)
        else:
            p.add_argument(*names, dest=key, type=type(value), default=value)
    args = p.parse_args()
    config = {key: getattr(args, key) for key in DEFAULTS}
    if args.batch_size is not None:
        if args.batch_size < 1 or args.batch_size % config["micro_batch"]:
            p.error("--batch-size must be a positive multiple of --micro-batch")
        config["accumulation_steps"] = args.batch_size // config["micro_batch"]
    return run(args.dataset, args.outdir, args.cache_dir, config, args.raw_layout, args.audit_first, args.runtime_mode)


if __name__ == "__main__":
    main()
