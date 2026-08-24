#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys, types
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp, optax
from flax import serialization
from model_relational_memory_v2 import NUM_CLASSES,FRAMES,NestSARRelationalMemoryV2,set_support_m4

EXPECTED_COUNTS={"xsub":(63_026,50_919),"xset":(54_468,59_477)}
SOURCE_RUNS={"xsub":Path("/kaggle/working/NestSAR_HOPE_Attention_Lite_D128_XSUB_SEED_42_paper_dual_t4_p12_v2"),"xset":Path("/kaggle/working/NestSAR_HOPE_Attention_Lite_D128_XSET_SEED_42_paper_dual_t4_p12_v2")}

def tree_numel(tree): return int(sum(np.prod(np.asarray(x).shape,dtype=np.int64) for x in jax.tree_util.tree_leaves(tree)))
def tree_leaf_count(tree): return len(jax.tree_util.tree_leaves(tree))
def find_dataset(explicit="auto"):
    if explicit and explicit!="auto":
        p=Path(explicit)
        if p.is_file(): return p
        raise FileNotFoundError(p)
    preferred=Path("/kaggle/input/models/paolamaydana/ntudanno/other/default/1/ntu120_3danno.pkl")
    if preferred.is_file(): return preferred
    for root in (Path("/kaggle/input"),Path("/kaggle/working")):
        if root.exists():
            hits=list(root.rglob("ntu120_3danno.pkl"))
            if hits: return hits[0]
    raise FileNotFoundError("ntu120_3danno.pkl not found")
def generated_source(protocol):
    run_dir=SOURCE_RUNS[protocol]; exact=run_dir/"generated_source"/f"attention_lite_{protocol}_seed_42_generated.py"
    if exact.is_file(): return exact
    hits=list((run_dir/"generated_source").glob("*.py")) if (run_dir/"generated_source").exists() else []
    if len(hits)==1: return hits[0]
    raise FileNotFoundError(f"Exact generated source not found under {run_dir}")
def load_support(source,protocol):
    text=source.read_text(encoding="utf-8"); marker="# 8. INITIALIZE"; cut=text.find(marker)
    if cut<0: raise RuntimeError(f"Could not find {marker!r}")
    section=text.rfind("# ==========================================================================================",0,cut); prefix=text[:section if section>0 else cut]
    for name in list(sys.modules):
        if name=="nestsar" or name.startswith("nestsar_"): del sys.modules[name]
    sys.path[:]=[p for p in sys.path if "NestSAR_HOPE_FIDELITY_UNIVERSAL" not in str(p)]
    name=f"_relmem_v2_support_{protocol}"; mod=types.ModuleType(name); mod.__file__=str(source); mod.__package__=None; sys.modules[name]=mod; exec(compile(prefix,str(source),"exec"),mod.__dict__)
    for s in ("m4","ns"):
        if not hasattr(mod,s): raise RuntimeError(f"Generated source missing {s}")
    return mod

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--protocol",choices=("xsub","xset"),required=True); p.add_argument("--epochs",type=int,default=30); p.add_argument("--batch",type=int,default=32); p.add_argument("--eval-batch",type=int,default=64); p.add_argument("--token-dim",type=int,default=160); p.add_argument("--relation-dim",type=int,default=80); p.add_argument("--relation-blocks",type=int,default=2); p.add_argument("--memory-dim",type=int,default=96); p.add_argument("--controller-rank",type=int,default=48); p.add_argument("--frame-blocks",type=int,default=2); p.add_argument("--chunk-blocks",type=int,default=2); p.add_argument("--clip-blocks",type=int,default=2); p.add_argument("--controller-blocks",type=int,default=2); p.add_argument("--dropout",type=float,default=0.15); p.add_argument("--seed",type=int,default=42); p.add_argument("--peak-lr",type=float,default=6e-4); p.add_argument("--end-lr",type=float,default=2e-5); p.add_argument("--warmup-frac",type=float,default=0.08); p.add_argument("--weight-decay",type=float,default=0.03); p.add_argument("--label-smoothing",type=float,default=0.05); p.add_argument("--grad-clip",type=float,default=1.0); p.add_argument("--ema",type=float,default=0.998); p.add_argument("--patience",type=int,default=6); p.add_argument("--hard-weight-min",type=float,default=0.85); p.add_argument("--hard-weight-max",type=float,default=1.60); p.add_argument("--difficulty-ema",type=float,default=0.95); p.add_argument("--hard-start",type=int,default=4); p.add_argument("--pair-start",type=int,default=7); p.add_argument("--hard-negatives",type=int,default=3); p.add_argument("--pair-margin",type=float,default=0.20); p.add_argument("--pair-weight",type=float,default=0.10); p.add_argument("--prototype-weight",type=float,default=0.05); p.add_argument("--direction-weight",type=float,default=0.03); p.add_argument("--dataset",default="auto"); p.add_argument("--progress-json",action="store_true"); p.add_argument("--skip-xla-audit",action="store_true"); return p.parse_args()
def emit(a,payload):
    if a.progress_json: print("@@NESTSAR@@"+json.dumps(payload,separators=(",",":")),flush=True)
def xla_flops(model,params):
    dummy=jnp.zeros((1,FRAMES,150),jnp.float32)
    def f(p,x): return model.apply({"params":p},x,training=False)["logits"]
    try:
        ca=jax.jit(f).lower(params,dummy).compile().cost_analysis(); ca=ca[0] if isinstance(ca,(list,tuple)) and ca else ca
        if isinstance(ca,dict):
            for key in ("flops","FLOPs","flop_count"):
                if key in ca: return float(ca[key])
    except Exception as e: print(f"XLA audit warning: {e}",file=sys.stderr,flush=True)
    return float("nan")

def main():
    a=parse_args(); protocol=a.protocol; devices=list(jax.local_devices())
    if jax.default_backend()!="gpu" or not devices: raise RuntimeError(f"Expected GPU backend, got {jax.default_backend()} {devices}")
    support=load_support(generated_source(protocol),protocol); set_support_m4(support.m4); ns=support.ns
    model=NestSARRelationalMemoryV2(token_dim=a.token_dim,rel_dim=a.relation_dim,relation_blocks=a.relation_blocks,memory_dim=a.memory_dim,controller_rank=a.controller_rank,frame_blocks=a.frame_blocks,chunk_blocks=a.chunk_blocks,clip_blocks=a.clip_blocks,controller_blocks=a.controller_blocks,dropout=a.dropout)
    raw=ns.load_pickle(find_dataset(a.dataset)); train_samples,val_samples=ns.build_samples(raw,protocol=protocol,max_train=0,max_val=0,seed=a.seed); et,ev=EXPECTED_COUNTS[protocol]
    if len(train_samples)!=et or len(val_samples)!=ev: raise RuntimeError(f"Split mismatch train={len(train_samples)} val={len(val_samples)}")
    train_ds=ns.SkeletonDataset(train_samples); val_ds=ns.SkeletonDataset(val_samples)
    rng=jax.random.PRNGKey(a.seed); rng,ir,dr=jax.random.split(rng,3); dummy=jnp.zeros((2,FRAMES,150),jnp.float32); params=model.init({"params":ir,"dropout":dr},dummy,training=True)["params"]
    pc=tree_numel(params); lc=tree_leaf_count(params); flops=float("nan") if a.skip_xla_audit else xla_flops(model,params); gf=flops/1e9 if np.isfinite(flops) else None
    out_dir=Path(f"/kaggle/working/NestSAR_RELATIONAL_MEMORY_V2_{protocol.upper()}_SEED{a.seed}"); out_dir.mkdir(parents=True,exist_ok=True)
    meta={"architecture":"NestSAR-Relational-Memory-v2","protocol":protocol,"seed":a.seed,"frames":FRAMES,"params":pc,"leaves":lc,"xla_forward_flops":None if not np.isfinite(flops) else flops,"xla_forward_gflops":gf,"token_dim":a.token_dim,"relation_dim":a.relation_dim,"relation_blocks":a.relation_blocks,"memory_dim":a.memory_dim,"controller_rank":a.controller_rank,"blocks":{"frame":a.frame_blocks,"chunk":a.chunk_blocks,"clip":a.clip_blocks,"controller":a.controller_blocks},"hard_class_learning":{"hard_start":a.hard_start,"pair_start":a.pair_start,"hard_negatives":a.hard_negatives,"pair_margin":a.pair_margin,"pair_weight":a.pair_weight,"prototype_weight":a.prototype_weight,"direction_weight":a.direction_weight}}
    (out_dir/"run_meta.json").write_text(json.dumps(meta,indent=2)); emit(a,{"kind":"meta",**meta})
    spe=len(train_ds)//a.batch; total=max(1,spe*a.epochs); warm=max(1,int(total*a.warmup_frac)); schedule=optax.warmup_cosine_decay_schedule(0.0,a.peak_lr,warm,total,a.end_lr); tx=optax.chain(optax.clip_by_global_norm(a.grad_clip),optax.adamw(schedule,weight_decay=a.weight_decay)); opt_state=tx.init(params); ema=params
    def ce120(logits,y,smoothing):
        target=jax.nn.one_hot(y,NUM_CLASSES); target=target*(1-smoothing)+smoothing/NUM_CLASSES; return -jnp.sum(target*jax.nn.log_softmax(logits),axis=-1)
    @jax.jit
    def train_step(params,ema,opt_state,x,y,cw,dk,pw,prw,dw):
        def loss_fn(p):
            o=model.apply({"params":p},x,training=True,rngs={"dropout":dk}); sw=cw[y]; ce=jnp.mean(sw*ce120(o["logits"],y,a.label_smoothing)); true=jnp.take_along_axis(o["logits"],y[:,None],axis=1)[:,0]; neg=jax.lax.top_k(o["logits"]-jax.nn.one_hot(y,NUM_CLASSES)*1e9,a.hard_negatives)[0]; pair=jnp.mean(sw[:,None]*jax.nn.relu(a.pair_margin-true[:,None]+neg)); proto=jnp.mean(sw*ce120(o["proto_logits"],y,0.0)); b=y.shape[0]; dy=jnp.concatenate((jnp.zeros((b,),jnp.int32),jnp.ones((b,),jnp.int32))); direction=-jnp.mean(jnp.take_along_axis(jax.nn.log_softmax(o["direction_logits"],axis=-1),dy[:,None],axis=1)[:,0]); loss=ce+pw*pair+prw*proto+dw*direction; pred=jnp.argmax(o["logits"],axis=-1); acc=jnp.mean(pred==y); return loss,(ce,pair,proto,direction,acc,pred,o["relation_gate"],o["relation_rms"],o["nested_gate"],o["nested_alpha"],o["relation_routes"],o["level_weights"])
        (loss,aux),grads=jax.value_and_grad(loss_fn,has_aux=True)(params); gn=optax.global_norm(grads); updates,opt_state=tx.update(grads,opt_state,params); params=optax.apply_updates(params,updates); ema=jax.tree_util.tree_map(lambda e,p:a.ema*e+(1-a.ema)*p,ema,params); return params,ema,opt_state,loss,gn,aux
    @jax.jit
    def eval_step(params,x,y,mask):
        o=model.apply({"params":params},x,training=False); pred=jnp.argmax(o["logits"],axis=-1); m=mask.astype(jnp.float32); return jnp.sum((pred==y).astype(jnp.float32)*m),jnp.sum(m),o["relation_gate"],o["relation_rms"],o["nested_gate"],o["nested_alpha"],o["relation_routes"],o["level_weights"]
    best=0.0; best_epoch=0; stale=0; history=[]; difficulty=np.full(NUM_CLASSES,0.5,np.float32); class_weights=np.ones(NUM_CLASSES,np.float32)
    for epoch in range(1,a.epochs+1):
        pw=a.pair_weight if epoch>=a.pair_start else 0.0; prw=a.prototype_weight if epoch>=a.hard_start else 0.0; dw=a.direction_weight if epoch>=a.hard_start else 0.0; counts=np.zeros(NUM_CLASSES,np.int64); corrects=np.zeros(NUM_CLASSES,np.int64); keys=("loss","ce","pair","proto","direction","acc","grad","gate","rms","nested_gate","nested_alpha"); sums={k:0.0 for k in keys}; routes_sum=np.zeros(4); levels_sum=np.zeros(4); nsteps=0
        for step,(xb,yb) in enumerate(ns.batch_iterator(train_ds,batch_size=a.batch,shuffle=True,seed=a.seed+epoch,drop_last=True),1):
            rng,dk=jax.random.split(rng); x=jnp.asarray(np.asarray(xb,np.float32)); y=jnp.asarray(np.asarray(yb,np.int32)); params,ema,opt_state,loss,gn,aux=train_step(params,ema,opt_state,x,y,jnp.asarray(class_weights),dk,jnp.asarray(pw,jnp.float32),jnp.asarray(prw,jnp.float32),jnp.asarray(dw,jnp.float32)); ce,pair,proto,direction,acc,pred,gate,rrms,ngate,nalpha,routes,levels=aux; yh=np.asarray(yb,np.int32); ph=np.asarray(pred); np.add.at(counts,yh,1); np.add.at(corrects,yh,(yh==ph).astype(np.int64)); nsteps+=1
            for k,v in zip(keys,(loss,ce,pair,proto,direction,acc,gn,gate,rrms,ngate,nalpha)): sums[k]+=float(np.asarray(v))
            routes_sum+=np.asarray(routes); levels_sum+=np.asarray(levels)
            if step==1 or step%5==0 or step==spe: emit(a,{"kind":"progress","phase":"train","epoch":epoch,"step":step,"total":spe,"loss":sums["loss"]/nsteps,"acc":sums["acc"]/nsteps,"gate":sums["gate"]/nsteps,"grad":sums["grad"]/nsteps,"best":best,"best_epoch":best_epoch})
        if epoch>=a.hard_start:
            cls_acc=corrects.astype(np.float64)/np.maximum(counts,1); current=np.where(counts>0,1-cls_acc,difficulty); difficulty=a.difficulty_ema*difficulty+(1-a.difficulty_ema)*current.astype(np.float32); class_weights=np.clip(a.hard_weight_min+(a.hard_weight_max-a.hard_weight_min)*difficulty,a.hard_weight_min,a.hard_weight_max).astype(np.float32)
        vc=vn=vg=vr=vng=vna=0.0; vrt=np.zeros(4); vlw=np.zeros(4); vs=0; vt=(len(val_ds)+a.eval_batch-1)//a.eval_batch
        for step,(xb,yb) in enumerate(ns.batch_iterator(val_ds,batch_size=a.eval_batch,shuffle=False,seed=0,drop_last=False),1):
            xa=np.asarray(xb,np.float32); ya=np.asarray(yb,np.int32); actual=len(ya); pad=a.eval_batch-actual
            if pad: xa=np.concatenate((xa,np.zeros((pad,FRAMES,150),np.float32))); ya=np.concatenate((ya,np.zeros((pad,),np.int32)))
            mask=np.zeros(a.eval_batch,np.float32); mask[:actual]=1; c,n,g,r,ng,na,rt,lw=eval_step(ema,jnp.asarray(xa),jnp.asarray(ya),jnp.asarray(mask)); vc+=float(c); vn+=float(n); vg+=float(g); vr+=float(r); vng+=float(ng); vna+=float(na); vrt+=np.asarray(rt); vlw+=np.asarray(lw); vs+=1; emit(a,{"kind":"progress","phase":"val","epoch":epoch,"step":step,"total":vt,"acc":vc/max(1,vn),"gate":vg/vs,"rms":vr/vs,"best":best,"best_epoch":best_epoch})
        va=vc/max(1,vn)
        if va>best:
            best=va; best_epoch=epoch; stale=0; (out_dir/"best_ema.msgpack").write_bytes(serialization.to_bytes(ema)); (out_dir/"best.json").write_text(json.dumps({"epoch":epoch,"accuracy":best,"correct":int(vc),"count":int(vn)},indent=2))
        else: stale+=1
        emit(a,{"kind":"epoch_end","epoch":epoch,"accuracy":va,"best":best,"best_epoch":best_epoch}); hard=np.argsort(-difficulty)[:10].astype(int).tolist(); rec={"epoch":epoch,"train":{k:v/max(1,nsteps) for k,v in sums.items()},"train_relation_routes":(routes_sum/max(1,nsteps)).tolist(),"train_level_weights":(levels_sum/max(1,nsteps)).tolist(),"hardest_class_ids":hard,"hardest_difficulty":[float(difficulty[i]) for i in hard],"class_weight_min":float(class_weights.min()),"class_weight_max":float(class_weights.max()),"loss_weights":{"pair":pw,"prototype":prw,"direction":dw},"val":{"accuracy":va,"correct":int(vc),"count":int(vn),"gate":vg/max(1,vs),"relation_rms":vr/max(1,vs),"nested_gate":vng/max(1,vs),"nested_alpha":vna/max(1,vs),"relation_routes":(vrt/max(1,vs)).tolist(),"level_weights":(vlw/max(1,vs)).tolist()},"best":best,"best_epoch":best_epoch}; history.append(rec); (out_dir/"history.json").write_text(json.dumps(history,indent=2))
        if stale>=a.patience: break
    emit(a,{"kind":"done","best":best,"best_epoch":best_epoch,"output":str(out_dir)})
if __name__=="__main__": main()
