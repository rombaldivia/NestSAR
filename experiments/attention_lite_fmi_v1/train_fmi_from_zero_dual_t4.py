#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, math, sys, time, types
from pathlib import Path
from functools import partial
import numpy as np
import jax, jax.numpy as jnp, optax
from flax import linen as nn, serialization
from tqdm.auto import tqdm

NUM_CLASSES=120
FRAMES=16
PERSONS=2
JOINTS=25
COORDS=3
FINE_JOINT_IDS=(2,3,4,5,6,7,8,9,10,11,20,21,22,23,24)
EXPECTED_COUNTS={"xsub":(63_026,50_919),"xset":(54_468,59_477)}
SOURCE_RUNS={
    "xsub":Path("/kaggle/working/NestSAR_HOPE_Attention_Lite_D128_XSUB_SEED_42_paper_dual_t4_p12_v2"),
    "xset":Path("/kaggle/working/NestSAR_HOPE_Attention_Lite_D128_XSET_SEED_42_paper_dual_t4_p12_v2"),
}

def tree_numel(tree): return int(sum(np.prod(np.asarray(x).shape,dtype=np.int64) for x in jax.tree_util.tree_leaves(tree)))
def tree_leaves(tree): return len(jax.tree_util.tree_leaves(tree))

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
    run_dir=SOURCE_RUNS[protocol]
    exact=run_dir/"generated_source"/f"attention_lite_{protocol}_seed_42_generated.py"
    if exact.is_file(): return exact
    hits=list((run_dir/"generated_source").glob("*.py")) if (run_dir/"generated_source").exists() else []
    if len(hits)==1: return hits[0]
    raise FileNotFoundError(f"Exact Attention-Lite generated source not found under {run_dir}")

def load_base_module(source,protocol):
    text=source.read_text(encoding="utf-8")
    marker="# 8. INITIALIZE"
    cut=text.find(marker)
    if cut<0: raise RuntimeError(f"Could not find {marker!r} in {source}")
    section=text.rfind("# ==========================================================================================",0,cut)
    prefix=text[:section if section>0 else cut]
    for name in list(sys.modules):
        if name=="nestsar" or name.startswith("nestsar_"): del sys.modules[name]
    sys.path[:]=[p for p in sys.path if "NestSAR_HOPE_FIDELITY_UNIVERSAL" not in str(p)]
    name=f"_attention_lite_seed42_{protocol}_fmi_zero_base"
    mod=types.ModuleType(name); mod.__file__=str(source); mod.__package__=None; sys.modules[name]=mod
    exec(compile(prefix,str(source),"exec"),mod.__dict__)
    for symbol in ("NestSARHOPEAttentionLiteVec31","VectorizedNonJointBackbone","m4","ns"):
        if not hasattr(mod,symbol): raise RuntimeError(f"Generated source missing {symbol}")
    return mod

class FineMotionInjector(nn.Module):
    hidden_dim:int=64
    @nn.compact
    def __call__(self,joint_motion):
        if joint_motion.ndim!=5: raise ValueError(f"Expected [B,T,M,V,C], got {joint_motion.shape}")
        acceleration=jnp.concatenate((jnp.zeros_like(joint_motion[:,:1]),joint_motion[:,1:]-joint_motion[:,:-1]),axis=1)
        ids=jnp.asarray(FINE_JOINT_IDS,dtype=jnp.int32)
        fine_v=jnp.take(joint_motion,ids,axis=3)
        fine_a=jnp.take(acceleration,ids,axis=3)
        b,t=joint_motion.shape[:2]
        features=jnp.concatenate((fine_v.reshape(b,t,-1),fine_a.reshape(b,t,-1)),axis=-1)
        h=nn.gelu(nn.Dense(self.hidden_dim,kernel_init=nn.initializers.xavier_uniform(),bias_init=nn.initializers.zeros,name="fmi_in")(features))
        delta=nn.Dense(PERSONS*len(FINE_JOINT_IDS)*COORDS,kernel_init=nn.initializers.normal(stddev=1e-3),bias_init=nn.initializers.zeros,name="fmi_out")(h)
        delta=delta.reshape(b,t,PERSONS,len(FINE_JOINT_IDS),COORDS)
        full_delta=jnp.zeros_like(joint_motion).at[:,:,:,ids,:].set(delta)
        gate_logit=self.param("gate_logit",nn.initializers.constant(-4.59512),(1,))
        gate=jax.nn.sigmoid(gate_logit)[0]
        enhanced=joint_motion+gate*full_delta
        return enhanced,gate,jnp.sqrt(jnp.mean(jnp.square(gate*full_delta))+1e-12)

def make_fmi_model(mod,hidden_dim):
    Base=mod.NestSARHOPEAttentionLiteVec31
    m4=mod.m4
    class NestSARAttentionLiteFMI(Base):
        fmi_hidden_dim:int=hidden_dim
        @nn.compact
        def __call__(self,x,training=False):
            if x.ndim!=3: raise ValueError(f"Expected [B,T,150], got {x.shape}")
            b,t,d=x.shape
            skel=x.reshape(b,t,PERSONS,JOINTS,COORDS)
            velocity=jnp.concatenate((jnp.zeros_like(skel[:,:1]),skel[:,1:]-skel[:,:-1]),axis=1)
            enhanced_motion,fmi_gate,fmi_rms=FineMotionInjector(self.fmi_hidden_dim,name="fine_motion_injector")(velocity)
            streams={"joint":skel,"bone":m4.bone_features(skel),"joint_motion":enhanced_motion,"bone_motion":m4.bone_features(enhanced_motion)}
            stream_outputs=[]
            stream_logits=[]
            predictions=[]
            motion_targets=[]
            for stream_name in ("joint","bone","joint_motion","bone_motion"):
                stream=streams[stream_name]
                backbone=self._stream_backbone(stream_name)
                out=backbone(stream,training=training)
                stream_outputs.append(out["embedding"])
                stream_logits.append(out["logits"])
                predictions.append(out.get("prediction",jnp.zeros_like(out["embedding"])))
                motion_targets.append(out.get("motion_target",jnp.zeros_like(out["embedding"])))
            stacked=jnp.stack(stream_outputs,axis=1)
            logits_stacked=jnp.stack(stream_logits,axis=1)
            fusion_logits=self.param("fusion_logits",nn.initializers.zeros,(4,))
            fusion_weights=jax.nn.softmax(fusion_logits)
            fused_embedding=jnp.sum(stacked*fusion_weights[None,:,None],axis=1)
            logits=jnp.sum(logits_stacked*fusion_weights[None,:,None],axis=1)
            prediction=jnp.sum(jnp.stack(predictions,axis=1)*fusion_weights[None,:,None],axis=1)
            motion_target=jnp.sum(jnp.stack(motion_targets,axis=1)*fusion_weights[None,:,None],axis=1)
            return {"logits":logits,"embedding":fused_embedding,"stream_logits":logits_stacked,"fusion_weights":fusion_weights,"prediction":prediction,"motion_target":motion_target,"fmi_gate":fmi_gate,"fmi_rms":fmi_rms}
    return NestSARAttentionLiteFMI

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--protocol",choices=("xsub","xset"),default="xsub")
    p.add_argument("--epochs",type=int,default=60)
    p.add_argument("--batch",type=int,default=64)
    p.add_argument("--eval-batch",type=int,default=128)
    p.add_argument("--hidden-dim",type=int,default=64)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--peak-lr",type=float,default=6e-4)
    p.add_argument("--end-lr",type=float,default=2e-5)
    p.add_argument("--warmup-frac",type=float,default=0.08)
    p.add_argument("--weight-decay",type=float,default=0.03)
    p.add_argument("--label-smoothing",type=float,default=0.05)
    p.add_argument("--grad-clip",type=float,default=1.0)
    p.add_argument("--ema",type=float,default=0.998)
    p.add_argument("--patience",type=int,default=12)
    p.add_argument("--dataset",default="auto")
    return p.parse_args()

def main():
    a=parse_args(); protocol=a.protocol; seed=a.seed
    devices=list(jax.local_devices()); ndev=len(devices)
    if jax.default_backend()!="gpu" or ndev!=2: raise RuntimeError(f"Expected 2 GPUs, got backend={jax.default_backend()} devices={devices}")
    if a.batch%ndev or a.eval_batch%ndev: raise RuntimeError("batch and eval-batch must be divisible by device count")
    source=generated_source(protocol); mod=load_base_module(source,protocol); ns=mod.ns
    Model=make_fmi_model(mod,a.hidden_dim)
    try:
        base=mod.build_model()
        fields={name:getattr(base,name) for name in base.__dataclass_fields__ if name not in ("parent","name") and hasattr(base,name)}
        fields["fmi_hidden_dim"]=a.hidden_dim
        model=Model(**fields)
    except Exception:
        model=Model(fmi_hidden_dim=a.hidden_dim)
    dataset=find_dataset(a.dataset); raw=ns.load_pickle(dataset)
    train_samples,val_samples=ns.build_samples(raw,protocol=protocol,max_train=0,max_val=0,seed=seed)
    expected_train,expected_val=EXPECTED_COUNTS[protocol]
    if len(train_samples)!=expected_train or len(val_samples)!=expected_val: raise RuntimeError(f"Split mismatch train={len(train_samples)} val={len(val_samples)}")
    train_ds=ns.SkeletonDataset(train_samples); val_ds=ns.SkeletonDataset(val_samples)
    rng=jax.random.PRNGKey(seed); rng,init_rng,drop_rng=jax.random.split(rng,3)
    dummy=jnp.zeros((2,FRAMES,PERSONS*JOINTS*COORDS),jnp.float32)
    params=model.init({"params":init_rng,"dropout":drop_rng},dummy,training=True)["params"]
    param_count=tree_numel(params); leaf_count=tree_leaves(params)
    steps_per_epoch=len(train_ds)//a.batch; total_steps=steps_per_epoch*a.epochs; warmup_steps=max(1,int(total_steps*a.warmup_frac))
    schedule=optax.warmup_cosine_decay_schedule(0.0,a.peak_lr,warmup_steps,total_steps,a.end_lr)
    tx=optax.chain(optax.clip_by_global_norm(a.grad_clip),optax.adamw(schedule,weight_decay=a.weight_decay,b1=0.9,b2=0.999,eps=1e-8))
    opt_state=tx.init(params); ema_params=params
    def repl(tree): return jax.device_put_replicated(tree,devices)
    params_r=repl(params); ema_r=repl(ema_params); opt_r=repl(opt_state)
    local_train=a.batch//ndev; local_eval=a.eval_batch//ndev
    def ce(logits,labels):
        targets=jax.nn.one_hot(labels,NUM_CLASSES); targets=targets*(1-a.label_smoothing)+a.label_smoothing/NUM_CLASSES
        return -jnp.mean(jnp.sum(targets*jax.nn.log_softmax(logits),axis=-1))
    @partial(jax.pmap,axis_name="gpu")
    def train_step(params,ema,opt_state,x,y,dropout_key):
        def loss_fn(p):
            out=model.apply({"params":p},x,training=True,rngs={"dropout":dropout_key}); loss=ce(out["logits"],y); acc=jnp.mean(jnp.argmax(out["logits"],axis=-1)==y)
            return loss,(acc,out["fmi_gate"],out["fmi_rms"])
        (loss,(acc,gate,rms)),grads=jax.value_and_grad(loss_fn,has_aux=True)(params)
        grads=jax.tree_util.tree_map(lambda g:jax.lax.pmean(g,"gpu"),grads); grad_norm=optax.global_norm(grads)
        updates,opt_state=tx.update(grads,opt_state,params); params=optax.apply_updates(params,updates)
        ema=jax.tree_util.tree_map(lambda e,p:a.ema*e+(1-a.ema)*p,ema,params)
        metrics={"loss":loss,"acc":acc,"gate":gate,"fmi_rms":rms,"grad_norm":grad_norm}
        metrics=jax.tree_util.tree_map(lambda z:jax.lax.pmean(z,"gpu"),metrics)
        return params,ema,opt_state,metrics
    @partial(jax.pmap,axis_name="gpu")
    def eval_step(params,x,y,mask):
        out=model.apply({"params":params},x,training=False); pred=jnp.argmax(out["logits"],axis=-1); m=mask.astype(jnp.float32)
        return {"correct":jnp.sum((pred==y).astype(jnp.float32)*m),"count":jnp.sum(m),"gate_sum":jnp.sum(jnp.ones_like(m)*out["fmi_gate"]*m),"rms_sum":jnp.sum(jnp.ones_like(m)*out["fmi_rms"]*m)}
    def shard_train(x,y):
        return jnp.asarray(np.asarray(x,np.float32).reshape(ndev,local_train,FRAMES,150)),jnp.asarray(np.asarray(y,np.int32).reshape(ndev,local_train))
    def shard_eval(x,y):
        x=np.asarray(x,np.float32); y=np.asarray(y,np.int32); actual=len(y); pad=a.eval_batch-actual
        if pad<0: raise ValueError("eval batch too large")
        if pad:
            x=np.concatenate((x,np.zeros((pad,FRAMES,150),np.float32)),axis=0); y=np.concatenate((y,np.zeros((pad,),np.int32)),axis=0)
        mask=np.zeros((a.eval_batch,),np.float32); mask[:actual]=1
        return jnp.asarray(x.reshape(ndev,local_eval,FRAMES,150)),jnp.asarray(y.reshape(ndev,local_eval)),jnp.asarray(mask.reshape(ndev,local_eval))
    def evaluate(params):
        correct=count=gate_sum=rms_sum=0.0
        for x,y in ns.batch_iterator(val_ds,batch_size=a.eval_batch,shuffle=False,seed=0,drop_last=False):
            xs,ys,ms=shard_eval(x,y); out=eval_step(params,xs,ys,ms)
            correct+=float(np.asarray(out["correct"]).sum()); count+=float(np.asarray(out["count"]).sum()); gate_sum+=float(np.asarray(out["gate_sum"]).sum()); rms_sum+=float(np.asarray(out["rms_sum"]).sum())
        return {"accuracy":correct/count,"correct":int(correct),"count":int(count),"gate":gate_sum/count,"fmi_rms":rms_sum/count}
    out_dir=Path(f"/kaggle/working/NestSAR_AttentionLite_FMI_FROM_ZERO_{protocol.upper()}_SEED{seed}"); out_dir.mkdir(parents=True,exist_ok=True)
    print("="*110); print("NESTSAR ATTENTION-LITE FMI — FROM ZERO"); print("="*110)
    print(f"Protocol: {protocol.upper()} | seed: {seed} | devices: {devices}"); print(f"Train/val: {len(train_ds):,}/{len(val_ds):,}"); print(f"Params: {param_count:,} | leaves: {leaf_count} | FMI hidden: {a.hidden_dim}"); print(f"Epochs: {a.epochs} | batch: {a.batch} | steps/epoch: {steps_per_epoch} | peak LR: {a.peak_lr}"); print("Initialization: RANDOM — no pretrained checkpoint loaded"); print("="*110)
    best=-1.0; best_epoch=0; no_improve=0; history=[]; start=time.time()
    for epoch in range(1,a.epochs+1):
        totals={"loss":0.0,"acc":0.0,"gate":0.0,"fmi_rms":0.0,"grad_norm":0.0}; nb=0
        iterator=ns.batch_iterator(train_ds,batch_size=a.batch,shuffle=True,seed=seed+epoch,drop_last=True)
        bar=tqdm(iterator,total=steps_per_epoch,desc=f"{protocol.upper()} FMI-ZERO E{epoch:03d}",leave=True)
        for x,y in bar:
            xs,ys=shard_train(x,y); rng,step_rng=jax.random.split(rng); keys=jax.random.split(step_rng,ndev)
            params_r,ema_r,opt_r,m=train_step(params_r,ema_r,opt_r,xs,ys,keys)
            host={k:float(np.asarray(v)[0]) for k,v in m.items()}
            for k in totals: totals[k]+=host[k]
            nb+=1
            if nb==1 or nb%25==0 or nb==steps_per_epoch:
                lr=float(schedule((epoch-1)*steps_per_epoch+nb-1)); bar.set_postfix(loss=f"{host['loss']:.4f}",acc=f"{100*host['acc']:.2f}%",gate=f"{host['gate']:.3f}",grad=f"{host['grad_norm']:.3f}",lr=f"{lr:.2e}")
        train_mean={k:v/max(nb,1) for k,v in totals.items()}; val=evaluate(ema_r); gain=val["accuracy"]
        record={"epoch":epoch,"train":train_mean,"val":val}; history.append(record); (out_dir/"history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
        print(f"E{epoch:03d} | train {100*train_mean['acc']:.3f}% | val {100*val['accuracy']:.5f}% ({val['correct']:,}/{val['count']:,}) | gate {val['gate']:.4f} | FMI-RMS {val['fmi_rms']:.6f}")
        if gain>best:
            best=gain; best_epoch=epoch; no_improve=0
            host_ema=jax.tree_util.tree_map(lambda z:np.asarray(z[0]),ema_r)
            payload={"ema_params":host_ema,"epoch":epoch,"val_accuracy":float(best),"protocol":protocol,"seed":seed,"architecture":"NestSAR-HOPE-Attention-Lite-D128-FMI-v1","from_zero":True}
            (out_dir/"best_ema.msgpack").write_bytes(serialization.to_bytes(payload)); (out_dir/"best.json").write_text(json.dumps({"epoch":epoch,"val_accuracy":best,"params":param_count,"leaves":leaf_count,"hidden_dim":a.hidden_dim},indent=2),encoding="utf-8")
            print(f"🔥 NEW BEST {100*best:.5f}%")
        else:
            no_improve+=1; print(f"No improvement: {no_improve}/{a.patience}")
        if no_improve>=a.patience:
            print("EARLY STOPPING"); break
    print("="*110); print(f"DONE | best={100*best:.5f}% @ epoch {best_epoch} | wall={(time.time()-start)/3600:.3f} h"); print(f"Output: {out_dir}"); print("="*110)

if __name__=="__main__": main()
