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

EXPECTED_BASE_PARAMS=2_381_028
EXPECTED_BASE_LEAVES=705
NUM_CLASSES=120
FRAMES=16
PERSONS=2
JOINTS=25
COORDS=3
FINE_JOINT_IDS=(2,3,4,5,6,7,8,9,10,11,20,21,22,23,24)
DEFAULT_RUNS={
    "xsub":Path("/kaggle/working/NestSAR_HOPE_Attention_Lite_D128_XSUB_SEED_42_paper_dual_t4_p12_v2"),
    "xset":Path("/kaggle/working/NestSAR_HOPE_Attention_Lite_D128_XSET_SEED_42_paper_dual_t4_p12_v2"),
}
EXPECTED_COUNTS={"xsub":(63_026,50_919),"xset":(54_468,59_477)}

def tree_numel(tree): return int(sum(np.prod(np.asarray(x).shape,dtype=np.int64) for x in jax.tree_util.tree_leaves(tree)))
def tree_leaves(tree): return len(jax.tree_util.tree_leaves(tree))

def find_dataset():
    preferred=Path("/kaggle/input/models/paolamaydana/ntudanno/other/default/1/ntu120_3danno.pkl")
    if preferred.is_file(): return preferred
    for root in (Path("/kaggle/input"),Path("/kaggle/working")):
        if root.exists():
            hits=list(root.rglob("ntu120_3danno.pkl"))
            if hits: return hits[0]
    raise FileNotFoundError("ntu120_3danno.pkl not found")

def generated_source(run_dir:Path,protocol:str):
    exact=run_dir/"generated_source"/f"attention_lite_{protocol}_seed_42_generated.py"
    if exact.is_file(): return exact
    hits=list((run_dir/"generated_source").glob("*.py")) if (run_dir/"generated_source").exists() else []
    if len(hits)==1: return hits[0]
    raise FileNotFoundError(f"Exact generated Attention-Lite source not found under {run_dir}")

def load_base_module(source:Path,protocol:str):
    text=source.read_text(encoding="utf-8")
    marker="# 8. INITIALIZE"
    cut=text.find(marker)
    if cut<0: raise RuntimeError(f"Could not find {marker!r} in {source}")
    section=text.rfind("# ==========================================================================================",0,cut)
    prefix=text[:section if section>0 else cut]
    for name in list(sys.modules):
        if name=="nestsar" or name.startswith("nestsar_"): del sys.modules[name]
    sys.path[:]=[p for p in sys.path if "NestSAR_HOPE_FIDELITY_UNIVERSAL" not in str(p)]
    name=f"_attention_lite_seed42_{protocol}_fmi_base"
    mod=types.ModuleType(name); mod.__file__=str(source); mod.__package__=None; sys.modules[name]=mod
    exec(compile(prefix,str(source),"exec"),mod.__dict__)
    for symbol in ("NestSARHOPEAttentionLiteVec31","StreamBackbone","VectorizedNonJointBackbone","m4","ns","build_model"):
        if not hasattr(mod,symbol): raise RuntimeError(f"Generated source missing {symbol}")
    return mod

class FineMotionInjector(nn.Module):
    hidden_dim:int=64
    @nn.compact
    def __call__(self,joint_motion:jnp.ndarray):
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
        rms=jnp.sqrt(jnp.mean(jnp.square(gate*full_delta))+1e-12)
        return enhanced,gate,rms

def make_fmi_model(base_mod,hidden_dim:int=64):
    StreamBackbone=base_mod.StreamBackbone
    VectorizedNonJointBackbone=base_mod.VectorizedNonJointBackbone
    m4=base_mod.m4
    NUM_STREAMS=int(getattr(base_mod,"NUM_STREAMS",4))
    class NestSARHOPEAttentionLiteFMI(nn.Module):
        num_classes:int; model_dim:int; memory_dim:int; dropout:float; memory_residual_scale:float; initial_eta:float; initial_alpha:float
        frame_blocks:int=2; chunk_blocks:int=2; clip_blocks:int=2; controller_blocks:int=2; chunk_size:int=4; clip_size:int=8; controller_rank:int=32
        @nn.compact
        def __call__(self,x:jnp.ndarray,training:bool):
            if x.shape[1]%self.chunk_size: raise ValueError("frames must be divisible by chunk_size")
            if x.shape[1]%self.clip_size: raise ValueError("frames must be divisible by clip_size")
            streams=m4.build_four_streams(x)
            joint_xyz=streams["joint"]
            geometry=m4.geometric_features(joint_xyz)
            joint_motion_fmi,fmi_gate,fmi_delta_rms=FineMotionInjector(hidden_dim=hidden_dim,name="fine_motion_injector")(streams["joint_motion"])
            joint_out=StreamBackbone(stream_name="joint",num_classes=self.num_classes,model_dim=self.model_dim,memory_dim=self.memory_dim,dropout=self.dropout,memory_residual_scale=self.memory_residual_scale,initial_eta=self.initial_eta,initial_alpha=self.initial_alpha,frame_blocks=self.frame_blocks,chunk_blocks=self.chunk_blocks,clip_blocks=self.clip_blocks,controller_blocks=self.controller_blocks,chunk_size=self.chunk_size,clip_size=self.clip_size,controller_rank=self.controller_rank,use_part_attention=False,name="stream_joint")(joint_xyz,joint_xyz,geometry,training)
            nonjoint_xyz=jnp.stack((streams["bone"],joint_motion_fmi,streams["bone_motion"]),axis=0)
            nonjoint_out=VectorizedNonJointBackbone(stream_name="bone",num_classes=self.num_classes,model_dim=self.model_dim,memory_dim=self.memory_dim,dropout=self.dropout,memory_residual_scale=self.memory_residual_scale,initial_eta=self.initial_eta,initial_alpha=self.initial_alpha,frame_blocks=self.frame_blocks,chunk_blocks=self.chunk_blocks,clip_blocks=self.clip_blocks,controller_blocks=self.controller_blocks,chunk_size=self.chunk_size,clip_size=self.clip_size,controller_rank=self.controller_rank,use_part_attention=False,name="stream_nonjoint_bank")(nonjoint_xyz,joint_xyz,geometry,training)
            stream_logits=jnp.swapaxes(jnp.concatenate((joint_out["logits"][None,...],nonjoint_out["logits"]),axis=0),0,1)
            fusion_logits=self.param("stream_fusion_logits",nn.initializers.zeros,(NUM_STREAMS,))
            fusion_weights=jax.nn.softmax(fusion_logits)
            logits=jnp.einsum("s,bsc->bc",fusion_weights,stream_logits)
            return {"logits":logits,"stream_logits":stream_logits,"fusion_weights":fusion_weights,"fmi_gate":fmi_gate,"fmi_delta_rms":fmi_delta_rms}
    return NestSARHOPEAttentionLiteFMI

def instantiate_fmi(base_mod,hidden_dim:int):
    base=base_mod.build_model()
    Cls=make_fmi_model(base_mod,hidden_dim)
    return Cls(num_classes=base.num_classes,model_dim=base.model_dim,memory_dim=base.memory_dim,dropout=base.dropout,memory_residual_scale=base.memory_residual_scale,initial_eta=base.initial_eta,initial_alpha=base.initial_alpha,frame_blocks=base.frame_blocks,chunk_blocks=base.chunk_blocks,clip_blocks=base.clip_blocks,controller_blocks=base.controller_blocks,chunk_size=base.chunk_size,clip_size=base.clip_size,controller_rank=base.controller_rank)

def replace_base_with_ema(initialized,base_ema):
    out=dict(initialized)
    fmi=out.pop("fine_motion_injector")
    merged=dict(base_ema)
    missing=[k for k in merged if k not in out]
    extra=[k for k in out if k not in merged]
    if missing or extra: raise RuntimeError(f"Base/FMI tree mismatch before overlay. missing={missing}, extra={extra}")
    for k,v in merged.items(): out[k]=v
    out["fine_motion_injector"]=fmi
    return out

def label_smooth_ce(logits,labels,smoothing=0.02):
    target=jax.nn.one_hot(labels,NUM_CLASSES)*(1-smoothing)+smoothing/NUM_CLASSES
    return -jnp.mean(jnp.sum(target*jax.nn.log_softmax(logits,axis=-1),axis=-1))

def kl_preserve(base_logits,new_logits,temp=2.0):
    t=jnp.asarray(temp,jnp.float32); a=jax.nn.log_softmax(base_logits/t,axis=-1); b=jax.nn.log_softmax(new_logits/t,axis=-1); return t*t*jnp.mean(jnp.sum(jnp.exp(a)*(a-b),axis=-1))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--protocol",choices=("xsub","xset"),default="xsub")
    ap.add_argument("--epochs",type=int,default=20)
    ap.add_argument("--batch",type=int,default=64)
    ap.add_argument("--eval-batch",type=int,default=128)
    ap.add_argument("--hidden-dim",type=int,default=64)
    ap.add_argument("--peak-lr",type=float,default=3e-4)
    ap.add_argument("--end-lr",type=float,default=2e-5)
    ap.add_argument("--warmup-epochs",type=int,default=2)
    ap.add_argument("--patience",type=int,default=6)
    ap.add_argument("--run-dir",type=Path,default=None)
    args=ap.parse_args()

    protocol=args.protocol; run_dir=(args.run_dir or DEFAULT_RUNS[protocol]).resolve(); ckpt=run_dir/"best_ema.msgpack"; source=generated_source(run_dir,protocol)
    if not ckpt.is_file(): raise FileNotFoundError(ckpt)
    devices=list(jax.local_devices()); ndev=len(devices)
    if jax.default_backend()!="gpu" or ndev!=2: raise RuntimeError(f"Expected Kaggle 2xGPU, got backend={jax.default_backend()} devices={devices}")
    if args.batch%ndev or args.eval_batch%ndev: raise RuntimeError("Batch sizes must be divisible by 2")

    print("="*118); print("NESTSAR ATTENTION-LITE + FMI v1 | SINGLE TRUNK | SEED 42 | 2xT4"); print("="*118)
    print(f"Protocol: {protocol.upper()} | run: {run_dir}"); print(f"Checkpoint: {ckpt}"); print(f"Source: {source}"); print(f"Devices: {devices}")

    base_mod=load_base_module(source,protocol); ns=base_mod.ns; BASE_MODEL=base_mod.build_model(); FMI_MODEL=instantiate_fmi(base_mod,args.hidden_dim)
    raw=serialization.msgpack_restore(ckpt.read_bytes())
    if "ema_params" not in raw: raise RuntimeError("Checkpoint has no ema_params")
    base_params=jax.tree_util.tree_map(jnp.asarray,raw["ema_params"])
    if tree_numel(base_params)!=EXPECTED_BASE_PARAMS or tree_leaves(base_params)!=EXPECTED_BASE_LEAVES: raise RuntimeError(f"Wrong base tree: {tree_numel(base_params):,} params / {tree_leaves(base_params)} leaves")

    key=jax.random.PRNGKey(42); key,kinit=jax.random.split(key)
    dummy=jnp.zeros((2,FRAMES,PERSONS*JOINTS*COORDS),jnp.float32)
    init_params=FMI_MODEL.init({"params":kinit,"dropout":kinit},dummy,training=False)["params"]
    full_params=replace_base_with_ema(init_params,base_params)
    fmi_params=full_params["fine_motion_injector"]
    fmi_count=tree_numel(fmi_params); total_count=EXPECTED_BASE_PARAMS+fmi_count
    print(f"Base params: {EXPECTED_BASE_PARAMS:,} | FMI params: {fmi_count:,} | total: {total_count:,}")
    print(f"Approx expected compute: ~0.0610 GFLOPs/clip vs 0.0604169 baseline")

    data_path=find_dataset(); data=ns.load_pickle(data_path); train_samples,val_samples=ns.build_samples(data,protocol=protocol,max_train=0,max_val=0,seed=42)
    exp_train,exp_val=EXPECTED_COUNTS[protocol]
    if len(train_samples)!=exp_train or len(val_samples)!=exp_val: raise RuntimeError(f"Split mismatch: {len(train_samples)}/{len(val_samples)} expected {exp_train}/{exp_val}")
    train_ds=ns.SkeletonDataset(train_samples); val_ds=ns.SkeletonDataset(val_samples)
    print(f"Dataset: {data_path} | train={len(train_ds):,} val={len(val_ds):,}")

    steps_per_epoch=len(train_ds)//args.batch; total_steps=steps_per_epoch*args.epochs; warmup_steps=steps_per_epoch*args.warmup_epochs
    schedule=optax.warmup_cosine_decay_schedule(0.0,args.peak_lr,warmup_steps,total_steps,args.end_lr)
    tx=optax.chain(optax.clip_by_global_norm(1.0),optax.adamw(schedule,weight_decay=1e-2))
    opt_state=tx.init(fmi_params); ema=fmi_params

    base_repl=jax.device_put_replicated(base_params,devices); fmi_repl=jax.device_put_replicated(fmi_params,devices); ema_repl=jax.device_put_replicated(ema,devices); opt_repl=jax.device_put_replicated(opt_state,devices)

    def merge(base,fmi):
        p=dict(base); p["fine_motion_injector"]=fmi; return p

    @partial(jax.pmap,axis_name="gpu")
    def train_step(fmi,ema,opt_state,base,x,y,dropkey):
        base_logits=jax.lax.stop_gradient(BASE_MODEL.apply({"params":base},x,training=False)["logits"])
        def loss_fn(fp):
            out=FMI_MODEL.apply({"params":merge(base,fp)},x,training=True,rngs={"dropout":dropkey}); logits=out["logits"]
            ce=label_smooth_ce(logits,y); kl=kl_preserve(base_logits,logits); gate=out["fmi_gate"]; loss=ce+0.10*kl+0.001*gate
            acc=jnp.mean((jnp.argmax(logits,-1)==y).astype(jnp.float32)); bacc=jnp.mean((jnp.argmax(base_logits,-1)==y).astype(jnp.float32))
            return loss,{"loss":loss,"ce":ce,"kl":kl,"acc":acc,"base_acc":bacc,"gate":gate,"delta_rms":out["fmi_delta_rms"]}
        (_,m),g=jax.value_and_grad(loss_fn,has_aux=True)(fmi); g=jax.tree_util.tree_map(lambda z:jax.lax.pmean(z,"gpu"),g); gn=optax.global_norm(g)
        updates,opt_state=tx.update(g,opt_state,fmi); fmi=optax.apply_updates(fmi,updates); ema=jax.tree_util.tree_map(lambda a,b:0.995*a+0.005*b,ema,fmi); m["grad_norm"]=gn; m=jax.tree_util.tree_map(lambda z:jax.lax.pmean(z,"gpu"),m)
        return fmi,ema,opt_state,m

    @partial(jax.pmap,axis_name="gpu")
    def eval_step(fmi,base,x,y,mask):
        base_logits=BASE_MODEL.apply({"params":base},x,training=False)["logits"]; out=FMI_MODEL.apply({"params":merge(base,fmi)},x,training=False); logits=out["logits"]
        pred=jnp.argmax(logits,-1); bpred=jnp.argmax(base_logits,-1); mask=mask.astype(jnp.float32)
        return {"correct":jnp.sum((pred==y)*mask),"base_correct":jnp.sum((bpred==y)*mask),"count":jnp.sum(mask),"gate_sum":out["fmi_gate"]*jnp.sum(mask),"delta_sum":out["fmi_delta_rms"]*jnp.sum(mask)}

    local=args.batch//ndev; local_eval=args.eval_batch//ndev
    def shard_train(x,y): return jnp.asarray(np.asarray(x,np.float32).reshape(ndev,local,FRAMES,150)),jnp.asarray(np.asarray(y,np.int32).reshape(ndev,local))
    def shard_eval(x,y):
        x=np.asarray(x,np.float32); y=np.asarray(y,np.int32); n=len(y); pad=args.eval_batch-n
        if pad<0: raise RuntimeError("eval batch larger than configured")
        if pad: x=np.concatenate((x,np.zeros((pad,FRAMES,150),np.float32))); y=np.concatenate((y,np.zeros((pad,),np.int32)))
        mask=np.zeros((args.eval_batch,),np.float32); mask[:n]=1
        return jnp.asarray(x.reshape(ndev,local_eval,FRAMES,150)),jnp.asarray(y.reshape(ndev,local_eval)),jnp.asarray(mask.reshape(ndev,local_eval))

    def evaluate(fp):
        c=bc=n=gs=ds=0.0
        for x,y in ns.batch_iterator(val_ds,batch_size=args.eval_batch,shuffle=False,seed=0,drop_last=False):
            sx,sy,sm=shard_eval(x,y); m=eval_step(fp,base_repl,sx,sy,sm); c+=float(np.asarray(m["correct"]).sum()); bc+=float(np.asarray(m["base_correct"]).sum()); n+=float(np.asarray(m["count"]).sum()); gs+=float(np.asarray(m["gate_sum"]).sum()); ds+=float(np.asarray(m["delta_sum"]).sum())
        return {"acc":c/n,"base_acc":bc/n,"correct":int(c),"base_correct":int(bc),"count":int(n),"gate":gs/n,"delta_rms":ds/n}

    pre=evaluate(ema_repl)
    print("="*118); print(f"PRETRAIN | base={100*pre['base_acc']:.5f}% | FMI-init={100*pre['acc']:.5f}% | diff={100*(pre['acc']-pre['base_acc']):+.5f} pp | gate={pre['gate']:.5f}"); print("="*118)
    if abs(pre["acc"]-pre["base_acc"])>0.002: raise RuntimeError("FMI initialization changed accuracy by >0.20 pp")

    out_dir=Path("/kaggle/working")/f"NestSAR_AttentionLite_FMI_{protocol.upper()}_SEED42_v1"; out_dir.mkdir(parents=True,exist_ok=True)
    best=pre["acc"]; best_epoch=0; stale=0; history=[]; start=time.time()
    for epoch in range(1,args.epochs+1):
        epoch_start=time.time(); sums={"loss":0.,"ce":0.,"kl":0.,"acc":0.,"base_acc":0.,"gate":0.,"delta_rms":0.,"grad_norm":0.}; nb=0
        iterator=ns.batch_iterator(train_ds,batch_size=args.batch,shuffle=True,seed=42+epoch,drop_last=True)
        bar=tqdm(iterator,total=steps_per_epoch,desc=f"{protocol.upper()} FMI E{epoch:02d}/{args.epochs:02d}",dynamic_ncols=True,leave=True)
        for x,y in bar:
            sx,sy=shard_train(x,y); key,sk=jax.random.split(key); dkeys=jax.random.split(sk,ndev); fmi_repl,ema_repl,opt_repl,m=train_step(fmi_repl,ema_repl,opt_repl,base_repl,sx,sy,dkeys)
            h={k:float(np.asarray(v)[0]) for k,v in m.items()}; nb+=1
            for k in sums: sums[k]+=h[k]
            bar.set_postfix(loss=f"{h['loss']:.3f}",acc=f"{100*h['acc']:.1f}%",base=f"{100*h['base_acc']:.1f}%",gate=f"{h['gate']:.3f}",grad=f"{h['grad_norm']:.2f}")
        train={k:v/max(nb,1) for k,v in sums.items()}; val=evaluate(ema_repl); gain=100*(val["acc"]-val["base_acc"]); sec=time.time()-epoch_start
        print(f"E{epoch:02d} | train={100*train['acc']:.3f}% | base={100*val['base_acc']:.5f}% | FMI={100*val['acc']:.5f}% | gain={gain:+.5f} pp | gate={val['gate']:.4f} | delta={val['delta_rms']:.5f} | {sec/60:.2f} min")
        history.append({"epoch":epoch,"train":train,"val":val,"gain_pp":gain,"seconds":sec}); (out_dir/"history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
        fmi_host=jax.tree_util.tree_map(lambda z:np.asarray(z[0]),fmi_repl); ema_host=jax.tree_util.tree_map(lambda z:np.asarray(z[0]),ema_repl); opt_host=jax.tree_util.tree_map(lambda z:np.asarray(z[0]),opt_repl)
        (out_dir/"last_resume.msgpack").write_bytes(serialization.to_bytes({"fmi_params":fmi_host,"fmi_ema_params":ema_host,"opt_state":opt_host,"epoch":epoch,"val_accuracy":val["acc"]}))
        if val["acc"]>best:
            best=val["acc"]; best_epoch=epoch; stale=0
            (out_dir/"best_fmi_ema.msgpack").write_bytes(serialization.to_bytes({"base_ema_params":jax.device_get(base_params),"fmi_ema_params":ema_host,"epoch":epoch,"protocol":protocol,"val_accuracy":best,"base_accuracy":val["base_acc"]}))
            (out_dir/"best_result.json").write_text(json.dumps({"architecture":"NestSAR-HOPE-Attention-Lite-D128+FMI-v1","single_trunk":True,"protocol":protocol,"seed":42,"base_parameters":EXPECTED_BASE_PARAMS,"fmi_parameters":fmi_count,"total_parameters":total_count,"best_epoch":best_epoch,"base_accuracy":val["base_acc"],"best_accuracy":best,"gain_pp":gain,"fmi_gate":val["gate"],"fine_joints":list(FINE_JOINT_IDS)},indent=2),encoding="utf-8")
            print(f"🔥 NEW BEST {100*best:.5f}% | {gain:+.5f} pp")
        else:
            stale+=1; print(f"No improvement {stale}/{args.patience}")
            if stale>=args.patience: print("EARLY STOPPING"); break
    print("="*118); print(f"DONE | baseline={100*pre['base_acc']:.5f}% | best FMI={100*best:.5f}% | gain={100*(best-pre['base_acc']):+.5f} pp | best epoch={best_epoch} | wall={(time.time()-start)/3600:.3f} h"); print(f"Output: {out_dir}"); print("="*118)

if __name__=="__main__": main()
