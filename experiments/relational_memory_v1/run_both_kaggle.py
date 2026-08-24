#!/usr/bin/env python3
import argparse, json, os, subprocess, time
from pathlib import Path
from tqdm.auto import tqdm

MARK='@@NESTSAR@@'

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--trainer',default='/kaggle/working/train_relational_memory_v1.py')
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--batch',type=int,default=32)
    p.add_argument('--eval-batch',type=int,default=64)
    p.add_argument('--relation-dim',type=int,default=64)
    p.add_argument('--fr-weight',type=float,default=0.10)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--peak-lr',default='6e-4')
    p.add_argument('--end-lr',default='2e-5')
    p.add_argument('--warmup-frac',default='0.08')
    p.add_argument('--weight-decay',default='0.03')
    p.add_argument('--label-smoothing',default='0.05')
    p.add_argument('--grad-clip',default='1.0')
    p.add_argument('--ema',default='0.998')
    p.add_argument('--patience',type=int,default=4)
    return p.parse_args()

def env_for(gpu):
    e=os.environ.copy(); e['CUDA_VISIBLE_DEVICES']=str(gpu); e['XLA_PYTHON_CLIENT_PREALLOCATE']='false'; e['XLA_PYTHON_CLIENT_MEM_FRACTION']='0.92'; e['MALLOC_ARENA_MAX']='2'; e['PYTHONUNBUFFERED']='1'; return e

def main():
    a=parse_args(); trainer=Path(a.trainer)
    if not trainer.is_file(): raise FileNotFoundError(trainer)
    logs={'XSUB':Path('/kaggle/working/relmem_xsub.log'),'XSET':Path('/kaggle/working/relmem_xset.log')}
    for p in logs.values():
        if p.exists(): p.unlink()
    common=['python','-u',str(trainer),'--epochs',str(a.epochs),'--batch',str(a.batch),'--eval-batch',str(a.eval_batch),'--relation-dim',str(a.relation_dim),'--fr-weight',str(a.fr_weight),'--seed',str(a.seed),'--peak-lr',a.peak_lr,'--end-lr',a.end_lr,'--warmup-frac',a.warmup_frac,'--weight-decay',a.weight_decay,'--label-smoothing',a.label_smoothing,'--grad-clip',a.grad_clip,'--ema',a.ema,'--patience',str(a.patience),'--progress-json']
    handles={}; procs={}
    for name,gpu,proto in [('XSUB',0,'xsub'),('XSET',1,'xset')]:
        handles[name]=open(logs[name],'w',buffering=1)
        procs[name]=subprocess.Popen(common+['--protocol',proto],stdout=handles[name],stderr=subprocess.STDOUT,env=env_for(gpu))
    bars={
        'XSUB':tqdm(total=1,desc='XSUB GPU0 | INIT',position=0,leave=True,dynamic_ncols=True),
        'XSET':tqdm(total=1,desc='XSET GPU1 | INIT',position=1,leave=True,dynamic_ncols=True),
    }
    offsets={k:0 for k in logs}; state={k:{'epoch':0,'phase':'init','step':0,'total':1,'meta':{}} for k in logs}
    try:
        while True:
            alive=False
            for name in ('XSUB','XSET'):
                if procs[name].poll() is None: alive=True
                path=logs[name]
                if not path.exists(): continue
                with open(path,'r',errors='ignore') as f:
                    f.seek(offsets[name]); chunk=f.read(); offsets[name]=f.tell()
                for line in chunk.splitlines():
                    if not line.startswith(MARK): continue
                    try: ev=json.loads(line[len(MARK):])
                    except Exception: continue
                    bar=bars[name]; s=state[name]; kind=ev.get('kind')
                    if kind=='meta':
                        s['meta']=ev; g=ev.get('xla_forward_gflops'); p=ev.get('params'); post={}
                        if p is not None: post['P']=f"{p/1e6:.3f}M"
                        if g is not None: post['G']=f"{g:.5f}"
                        bar.set_postfix(post)
                    elif kind=='progress':
                        phase=ev['phase']; epoch=int(ev['epoch']); step=int(ev['step']); total=int(ev['total'])
                        if phase!=s['phase'] or epoch!=s['epoch'] or total!=s['total']:
                            bar.reset(total=total); s.update({'phase':phase,'epoch':epoch,'step':0,'total':total})
                        if step>s['step']: bar.update(step-s['step']); s['step']=step
                        bar.set_description(f"{name} GPU{0 if name=='XSUB' else 1} | {phase.upper():5s} E{epoch:03d}")
                        if phase=='train':
                            bar.set_postfix(loss=f"{ev.get('loss',0):.3f}",acc=f"{100*ev.get('acc',0):.2f}%",gate=f"{ev.get('gate',0):.3f}",grad=f"{ev.get('grad',0):.2f}")
                        else:
                            bar.set_postfix(acc=f"{100*ev.get('acc',0):.2f}%",best=f"{100*ev.get('best',0):.2f}%",gate=f"{ev.get('gate',0):.3f}",rms=f"{ev.get('rms',0):.4f}")
                    elif kind=='done':
                        bar.set_description(f"{name} GPU{0 if name=='XSUB' else 1} | DONE")
                        bar.set_postfix(best=f"{100*ev.get('best',0):.2f}%")
            if not alive: break
            time.sleep(0.25)
    finally:
        for h in handles.values(): h.close()
        for b in bars.values(): b.close()
    failed=[]
    for name,p in procs.items():
        if p.returncode!=0: failed.append(name)
    if failed:
        for name in failed:
            tail=logs[name].read_text(errors='ignore').splitlines()[-30:]
            print(f"\n{name} FAILED\n"+'\n'.join(tail))
        raise SystemExit(1)

if __name__=='__main__': main()
