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
    p.add_argument('--refresh-seconds',type=float,default=0.50)
    return p.parse_args()

def env_for(gpu):
    e=os.environ.copy()
    e['CUDA_VISIBLE_DEVICES']=str(gpu)
    e['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
    e['XLA_PYTHON_CLIENT_MEM_FRACTION']='0.92'
    e['MALLOC_ARENA_MAX']='2'
    e['PYTHONUNBUFFERED']='1'
    return e

def global_fraction(s,epochs):
    if s.get('done'): return 1.0
    epoch=max(1,int(s.get('epoch',1)))
    phase=s.get('phase','init')
    step=max(0,int(s.get('step',0)))
    total=max(1,int(s.get('total',1)))
    if phase=='init': within=0.0
    elif phase=='train': within=0.85*min(1.0,step/total)
    elif phase=='val': within=0.85+0.15*min(1.0,step/total)
    else: within=0.0
    return min(1.0,max(0.0,((epoch-1)+within)/max(1,epochs)))

def compact_status(name,s):
    if s.get('done'):
        return f"{name}:DONE best={100*s.get('best',0.0):.2f}%"
    meta=s.get('meta',{})
    if s.get('phase')=='init':
        p=meta.get('params'); g=meta.get('xla_forward_gflops')
        if p is not None and g is not None: return f"{name}:INIT P={p/1e6:.3f}M G={g:.5f}"
        return f"{name}:INIT"
    phase=s.get('phase','?').upper()
    epoch=int(s.get('epoch',0)); step=int(s.get('step',0)); total=max(1,int(s.get('total',1))); pct=100*step/total
    if s.get('phase')=='train':
        return f"{name}:{phase} E{epoch:03d} {pct:4.1f}% acc={100*s.get('acc',0.0):.2f}% loss={s.get('loss',0.0):.3f}"
    return f"{name}:{phase} E{epoch:03d} {pct:4.1f}% acc={100*s.get('acc',0.0):.2f}% best={100*s.get('best',0.0):.2f}%"

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
    state={k:{'epoch':1,'phase':'init','step':0,'total':1,'meta':{},'done':False,'best':0.0} for k in logs}
    offsets={k:0 for k in logs}
    bar=tqdm(total=1000,desc='NESTSAR REL-MEM | XSUB GPU0 + XSET GPU1',unit='‰',dynamic_ncols=True,mininterval=max(0.25,a.refresh_seconds),leave=True)
    last_render=0.0
    try:
        while True:
            alive=False; changed=False
            for name in ('XSUB','XSET'):
                if procs[name].poll() is None: alive=True
                path=logs[name]
                if not path.exists(): continue
                with open(path,'r',errors='ignore') as f:
                    f.seek(offsets[name]); chunk=f.read(); offsets[name]=f.tell()
                latest=None
                for line in chunk.splitlines():
                    if not line.startswith(MARK): continue
                    try: latest=json.loads(line[len(MARK):])
                    except Exception: continue
                    kind=latest.get('kind'); s=state[name]
                    if kind=='meta': s['meta']=latest; changed=True
                    elif kind=='progress':
                        s.update({'phase':latest.get('phase','train'),'epoch':int(latest.get('epoch',s['epoch'])),'step':int(latest.get('step',s['step'])),'total':int(latest.get('total',s['total'])),'loss':float(latest.get('loss',s.get('loss',0.0))),'acc':float(latest.get('acc',s.get('acc',0.0))),'gate':float(latest.get('gate',s.get('gate',0.0))),'grad':float(latest.get('grad',s.get('grad',0.0))),'best':float(latest.get('best',s.get('best',0.0))),'rms':float(latest.get('rms',s.get('rms',0.0)))})
                        changed=True
                    elif kind=='done': s.update({'done':True,'best':float(latest.get('best',s.get('best',0.0)))}) ; changed=True
            now=time.monotonic()
            if changed and (now-last_render>=a.refresh_seconds or not alive):
                frac=0.5*(global_fraction(state['XSUB'],a.epochs)+global_fraction(state['XSET'],a.epochs))
                bar.n=int(round(1000*frac))
                bar.set_postfix_str(compact_status('XSUB',state['XSUB'])+' | '+compact_status('XSET',state['XSET']),refresh=False)
                bar.refresh(); last_render=now
            if not alive: break
            time.sleep(0.20)
    finally:
        for h in handles.values(): h.close()
        bar.close()
    failed=[name for name,p in procs.items() if p.returncode!=0]
    if failed:
        for name in failed:
            tail=logs[name].read_text(errors='ignore').splitlines()[-30:]
            print(f"\n{name} FAILED\n"+'\n'.join(tail))
        raise SystemExit(1)

if __name__=='__main__': main()
