#!/usr/bin/env python3
import argparse, json, os, subprocess, time
from pathlib import Path
from tqdm.auto import tqdm

MARK='@@NESTSAR@@'

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--trainer',default='/kaggle/working/train_relational_memory_v2.py'); p.add_argument('--epochs',type=int,default=30); p.add_argument('--batch',type=int,default=32); p.add_argument('--eval-batch',type=int,default=64); p.add_argument('--token-dim',type=int,default=160); p.add_argument('--relation-dim',type=int,default=80); p.add_argument('--relation-blocks',type=int,default=2); p.add_argument('--memory-dim',type=int,default=96); p.add_argument('--controller-rank',type=int,default=48); p.add_argument('--frame-blocks',type=int,default=2); p.add_argument('--chunk-blocks',type=int,default=2); p.add_argument('--clip-blocks',type=int,default=2); p.add_argument('--controller-blocks',type=int,default=2); p.add_argument('--dropout',default='0.15'); p.add_argument('--seed',type=int,default=42); p.add_argument('--peak-lr',default='6e-4'); p.add_argument('--end-lr',default='2e-5'); p.add_argument('--warmup-frac',default='0.08'); p.add_argument('--weight-decay',default='0.03'); p.add_argument('--label-smoothing',default='0.05'); p.add_argument('--grad-clip',default='1.0'); p.add_argument('--ema',default='0.998'); p.add_argument('--patience',type=int,default=6); p.add_argument('--hard-weight-min',default='0.85'); p.add_argument('--hard-weight-max',default='1.60'); p.add_argument('--difficulty-ema',default='0.95'); p.add_argument('--hard-start',type=int,default=4); p.add_argument('--pair-start',type=int,default=7); p.add_argument('--hard-negatives',type=int,default=3); p.add_argument('--pair-margin',default='0.20'); p.add_argument('--pair-weight',default='0.10'); p.add_argument('--prototype-weight',default='0.05'); p.add_argument('--direction-weight',default='0.03'); p.add_argument('--refresh-seconds',type=float,default=0.50); return p.parse_args()

def env_for(gpu):
    e=os.environ.copy(); e['CUDA_VISIBLE_DEVICES']=str(gpu); e['XLA_PYTHON_CLIENT_PREALLOCATE']='false'; e['XLA_PYTHON_CLIENT_MEM_FRACTION']='0.92'; e['MALLOC_ARENA_MAX']='2'; e['PYTHONUNBUFFERED']='1'; return e

def best_text(s):
    e=int(s.get('best_epoch',0)); b=100*float(s.get('best',0.0)); return '--' if e<=0 else f'{b:.2f}%@E{e:03d}'

def main():
    a=parse_args(); trainer=Path(a.trainer)
    if not trainer.is_file(): raise FileNotFoundError(trainer)
    logs={'XSUB':Path('/kaggle/working/relmem_v2_xsub.log'),'XSET':Path('/kaggle/working/relmem_v2_xset.log')}
    for p in logs.values():
        if p.exists(): p.unlink()
    common=['python','-u',str(trainer),'--epochs',str(a.epochs),'--batch',str(a.batch),'--eval-batch',str(a.eval_batch),'--token-dim',str(a.token_dim),'--relation-dim',str(a.relation_dim),'--relation-blocks',str(a.relation_blocks),'--memory-dim',str(a.memory_dim),'--controller-rank',str(a.controller_rank),'--frame-blocks',str(a.frame_blocks),'--chunk-blocks',str(a.chunk_blocks),'--clip-blocks',str(a.clip_blocks),'--controller-blocks',str(a.controller_blocks),'--dropout',a.dropout,'--seed',str(a.seed),'--peak-lr',a.peak_lr,'--end-lr',a.end_lr,'--warmup-frac',a.warmup_frac,'--weight-decay',a.weight_decay,'--label-smoothing',a.label_smoothing,'--grad-clip',a.grad_clip,'--ema',a.ema,'--patience',str(a.patience),'--hard-weight-min',a.hard_weight_min,'--hard-weight-max',a.hard_weight_max,'--difficulty-ema',a.difficulty_ema,'--hard-start',str(a.hard_start),'--pair-start',str(a.pair_start),'--hard-negatives',str(a.hard_negatives),'--pair-margin',a.pair_margin,'--pair-weight',a.pair_weight,'--prototype-weight',a.prototype_weight,'--direction-weight',a.direction_weight,'--progress-json']
    handles={}; procs={}
    for name,gpu,proto in [('XSUB',0,'xsub'),('XSET',1,'xset')]:
        handles[name]=open(logs[name],'w',buffering=1); procs[name]=subprocess.Popen(common+['--protocol',proto],stdout=handles[name],stderr=subprocess.STDOUT,env=env_for(gpu))
    bars={'XSUB':tqdm(total=1,desc='XSUB GPU0 | INIT',position=0,leave=True,dynamic_ncols=True,mininterval=a.refresh_seconds),'XSET':tqdm(total=1,desc='XSET GPU1 | INIT',position=1,leave=True,dynamic_ncols=True,mininterval=a.refresh_seconds)}
    offsets={k:0 for k in logs}; state={k:{'epoch':0,'phase':'init','step':0,'total':1,'best':0.0,'best_epoch':0,'last_render':0.0} for k in logs}
    try:
        while True:
            alive=False
            for name in ('XSUB','XSET'):
                if procs[name].poll() is None: alive=True
                path=logs[name]
                if not path.exists(): continue
                with open(path,'r',errors='ignore') as f: f.seek(offsets[name]); chunk=f.read(); offsets[name]=f.tell()
                events=[]
                for line in chunk.splitlines():
                    if line.startswith(MARK):
                        try: events.append(json.loads(line[len(MARK):]))
                        except Exception: pass
                if not events: continue
                s=state[name]; bar=bars[name]; gpu=0 if name=='XSUB' else 1
                for ev in events:
                    kind=ev.get('kind')
                    if kind=='meta':
                        p=ev.get('params'); g=ev.get('xla_forward_gflops'); post=[]
                        if p is not None: post.append(f'P={p/1e6:.3f}M')
                        if g is not None: post.append(f'G={g:.5f}')
                        bar.set_description_str(f'{name} GPU{gpu} | INIT',refresh=False); bar.set_postfix_str(' '.join(post),refresh=False)
                    elif kind=='progress':
                        phase=str(ev.get('phase','train')); epoch=int(ev.get('epoch',1)); step=int(ev.get('step',0)); total=max(1,int(ev.get('total',1))); s['best']=float(ev.get('best',s['best'])); s['best_epoch']=int(ev.get('best_epoch',s['best_epoch']))
                        if phase!=s['phase'] or epoch!=s['epoch'] or total!=s['total']: bar.total=total; bar.n=0; s.update({'phase':phase,'epoch':epoch,'step':0,'total':total})
                        bar.n=min(step,total); s['step']=step; bar.set_description_str(f'{name} GPU{gpu} | {phase.upper():5s} E{epoch:03d}',refresh=False)
                        if phase=='train': bar.set_postfix_str(f"loss={float(ev.get('loss',0)):.3f} acc={100*float(ev.get('acc',0)):.2f}% best={best_text(s)}",refresh=False)
                        else: bar.set_postfix_str(f"acc={100*float(ev.get('acc',0)):.2f}% best={best_text(s)} gate={float(ev.get('gate',0)):.3f}",refresh=False)
                    elif kind=='epoch_end':
                        s['best']=float(ev.get('best',s['best'])); s['best_epoch']=int(ev.get('best_epoch',s['best_epoch'])); bar.set_postfix_str(f"acc={100*float(ev.get('accuracy',0)):.2f}% best={best_text(s)}",refresh=False)
                    elif kind=='done':
                        s['best']=float(ev.get('best',s['best'])); s['best_epoch']=int(ev.get('best_epoch',s['best_epoch'])); bar.n=bar.total; bar.set_description_str(f'{name} GPU{gpu} | DONE',refresh=False); bar.set_postfix_str(f'best={best_text(s)}',refresh=False)
                now=time.monotonic()
                if now-s['last_render']>=a.refresh_seconds or any(e.get('kind') in ('epoch_end','done') for e in events): bar.refresh(); s['last_render']=now
            if not alive: break
            time.sleep(0.20)
    finally:
        for h in handles.values(): h.close()
        for b in bars.values(): b.close()
    failed=[name for name,p in procs.items() if p.returncode!=0]
    if failed:
        for name in failed:
            tail=logs[name].read_text(errors='ignore').splitlines()[-40:]; print(f'\n{name} FAILED\n'+'\n'.join(tail))
        raise SystemExit(1)

if __name__=='__main__': main()
