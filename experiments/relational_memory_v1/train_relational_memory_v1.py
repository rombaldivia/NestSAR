#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, math, sys, types
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
EXPECTED_COUNTS={"xsub":(63_026,50_919),"xset":(54_468,59_477)}
SOURCE_RUNS={
    "xsub":Path("/kaggle/working/NestSAR_HOPE_Attention_Lite_D128_XSUB_SEED_42_paper_dual_t4_p12_v2"),
    "xset":Path("/kaggle/working/NestSAR_HOPE_Attention_Lite_D128_XSET_SEED_42_paper_dual_t4_p12_v2"),
}
NTU_EDGES=((0,1),(1,20),(20,2),(2,3),(20,4),(4,5),(5,6),(6,7),(7,21),(7,22),(20,8),(8,9),(9,10),(10,11),(11,23),(11,24),(0,12),(12,13),(13,14),(14,15),(0,16),(16,17),(17,18),(18,19))

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
    run_dir=SOURCE_RUNS[protocol]
    exact=run_dir/"generated_source"/f"attention_lite_{protocol}_seed_42_generated.py"
    if exact.is_file(): return exact
    hits=list((run_dir/"generated_source").glob("*.py")) if (run_dir/"generated_source").exists() else []
    if len(hits)==1: return hits[0]
    raise FileNotFoundError(f"Exact Attention-Lite generated source not found under {run_dir}")

def load_base_module(source,protocol):
    text=source.read_text(encoding="utf-8"); marker="# 8. INITIALIZE"; cut=text.find(marker)
    if cut<0: raise RuntimeError(f"Could not find {marker!r} in {source}")
    section=text.rfind("# ==========================================================================================",0,cut); prefix=text[:section if section>0 else cut]
    for name in list(sys.modules):
        if name=="nestsar" or name.startswith("nestsar_"): del sys.modules[name]
    sys.path[:]=[p for p in sys.path if "NestSAR_HOPE_FIDELITY_UNIVERSAL" not in str(p)]
    name=f"_attention_lite_relmem_{protocol}"; mod=types.ModuleType(name); mod.__file__=str(source); mod.__package__=None; sys.modules[name]=mod
    exec(compile(prefix,str(source),"exec"),mod.__dict__)
    for symbol in ("NestSARHOPEAttentionLiteVec31","StreamBackbone","m4","ns","build_model"):
        if not hasattr(mod,symbol): raise RuntimeError(f"Generated source missing {symbol}")
    return mod

def _graph_tables():
    adj=[set([i]) for i in range(JOINTS)]
    for a,b in NTU_EDGES: adj[a].add(b); adj[b].add(a)
    dist=np.full((JOINTS,JOINTS),999,np.int32)
    for src in range(JOINTS):
        dist[src,src]=0; q=[src]
        for u in q:
            for v in adj[u]:
                if dist[src,v]>dist[src,u]+1: dist[src,v]=dist[src,u]+1; q.append(v)
    local=[]; far=[]
    for j in range(JOINTS):
        local.append(sorted(adj[j],key=lambda x:(dist[j,x],x))[:4])
        far.append([x for x in np.argsort(-dist[j]).tolist() if x!=j][:4])
    return local,far

def build_relation_tables():
    local_joints,far_joints=_graph_tables(); anchors=(0,4,8,12,15); other_key=(6,10)
    def tid(t,m,j): return (t*PERSONS+m)*JOINTS+j
    branches={"ll":[],"dl":[],"lg":[],"dg":[]}
    for t in range(FRAMES):
        near=[tt for tt in (t-1,t,t+1) if 0<=tt<FRAMES]
        distant=[tt for tt in anchors if abs(tt-t)>=3] or [0 if t>FRAMES//2 else FRAMES-1]
        for m in range(PERSONS):
            om=1-m
            for j in range(JOINTS):
                ll=[tid(tt,m,jj) for tt in near for jj in local_joints[j]]
                dj=far_joints[j]; cross=[]
                for jj in (j,)+other_key:
                    if jj not in cross: cross.append(jj)
                dl=[tid(tt,m,jj) for tt in near for jj in dj]+[tid(tt,om,jj) for tt in near for jj in cross]
                lg=[tid(tt,m,jj) for tt in distant for jj in local_joints[j]]
                dg=[tid(tt,m,jj) for tt in distant for jj in dj]+[tid(tt,om,jj) for tt in distant for jj in cross]
                branches["ll"].append(ll); branches["dl"].append(dl); branches["lg"].append(lg); branches["dg"].append(dg)
    packed={}
    for name,rows in branches.items():
        k=max(len(r) for r in rows); idx=np.zeros((len(rows),k),np.int32); mask=np.zeros((len(rows),k),np.bool_)
        for i,row in enumerate(rows):
            row=list(dict.fromkeys(row)); idx[i,:len(row)]=row; mask[i,:len(row)]=True
        packed[name]=(idx,mask)
    return packed
RELATION_TABLES=build_relation_tables()

class SparseRelationBranch(nn.Module):
    branch:str
    head_dim:int
    @nn.compact
    def __call__(self,q,k,v):
        idx_np,mask_np=RELATION_TABLES[self.branch]; idx=jnp.asarray(idx_np,dtype=jnp.int32); mask=jnp.asarray(mask_np,dtype=jnp.bool_)
        kg=jnp.take(k,idx,axis=1); vg=jnp.take(v,idx,axis=1)
        score=jnp.einsum("bnd,bnkd->bnk",q,kg)/jnp.sqrt(jnp.asarray(self.head_dim,jnp.float32))
        score=jnp.where(mask[None,:,:],score,jnp.asarray(-1e9,score.dtype)); attn=jax.nn.softmax(score,axis=-1)
        return jnp.einsum("bnk,bnkd->bnd",attn,vg)

class AdaptiveRelationalFrontEnd(nn.Module):
    rel_dim:int=64
    gate_init_logit:float=-2.944439
    @nn.compact
    def __call__(self,joint,bone,joint_motion,bone_motion):
        if self.rel_dim%4: raise ValueError("rel_dim must be divisible by 4")
        acceleration=jnp.concatenate((jnp.zeros_like(joint_motion[:,:1]),joint_motion[:,1:]-joint_motion[:,:-1]),axis=1)
        raw=jnp.concatenate((joint,bone,joint_motion,bone_motion,acceleration),axis=-1)
        b=raw.shape[0]; n=FRAMES*PERSONS*JOINTS; hd=self.rel_dim//4; tokens=raw.reshape(b,n,-1)
        z=nn.Dense(self.rel_dim,kernel_init=nn.initializers.xavier_uniform(),name="modal_fuse")(tokens); z=nn.gelu(nn.LayerNorm(name="modal_ln")(z))
        q=nn.Dense(self.rel_dim,use_bias=False,kernel_init=nn.initializers.xavier_uniform(),name="q")(z)
        k=nn.Dense(self.rel_dim,use_bias=False,kernel_init=nn.initializers.xavier_uniform(),name="k")(z)
        v=nn.Dense(self.rel_dim,use_bias=False,kernel_init=nn.initializers.xavier_uniform(),name="v")(z)
        qh=q.reshape(b,n,4,hd); kh=k.reshape(b,n,4,hd); vh=v.reshape(b,n,4,hd); outs=[]
        for h,name in enumerate(("ll","dl","lg","dg")):
            outs.append(SparseRelationBranch(name,hd,name=f"rel_{name}")(qh[:,:,h,:],kh[:,:,h,:],vh[:,:,h,:]))
        mixed=jnp.concatenate(outs,axis=-1); mixed=nn.Dense(self.rel_dim,kernel_init=nn.initializers.xavier_uniform(),name="rel_out")(mixed)
        mixed=nn.gelu(nn.LayerNorm(name="rel_ln")(z+mixed))
        delta=nn.Dense(COORDS,kernel_init=nn.initializers.normal(stddev=1e-3),bias_init=nn.initializers.zeros,name="xyz_delta")(mixed)
        gate_logit=self.param("gate_logit",nn.initializers.constant(self.gate_init_logit),(1,)); gate=jax.nn.sigmoid(gate_logit)[0]
        residual=(gate*delta).reshape(b,FRAMES,PERSONS,JOINTS,COORDS); relational_xyz=joint+residual
        pooled=jnp.mean(mixed.reshape(b,FRAMES,PERSONS,JOINTS,self.rel_dim),axis=(1,2,3)); rms=jnp.sqrt(jnp.mean(jnp.square(residual))+1e-12)
        return relational_xyz,pooled,gate,rms

def make_relational_memory_model(mod,rel_dim):
    Base=mod.NestSARHOPEAttentionLiteVec31; StreamBackbone=mod.StreamBackbone; m4=mod.m4
    class NestSARRelationalMemoryV1(Base):
        relation_dim:int=rel_dim
        @nn.compact
        def __call__(self,x,training=False):
            if x.shape[1]%self.chunk_size: raise ValueError("frames must be divisible by chunk_size")
            if x.shape[1]%self.clip_size: raise ValueError("frames must be divisible by clip_size")
            streams=dict(m4.build_four_streams(x)); joint=streams["joint"]
            rel_xyz,rel_feat,rel_gate,rel_rms=AdaptiveRelationalFrontEnd(self.relation_dim,name="adaptive_joint_time_relations")(joint,streams["bone"],streams["joint_motion"],streams["bone_motion"])
            geometry=m4.geometric_features(rel_xyz)
            trunk=StreamBackbone(stream_name="joint",num_classes=self.num_classes,model_dim=self.model_dim,memory_dim=self.memory_dim,dropout=self.dropout,memory_residual_scale=self.memory_residual_scale,initial_eta=self.initial_eta,initial_alpha=self.initial_alpha,frame_blocks=self.frame_blocks,chunk_blocks=self.chunk_blocks,clip_blocks=self.clip_blocks,controller_blocks=self.controller_blocks,chunk_size=self.chunk_size,clip_size=self.clip_size,controller_rank=self.controller_rank,use_part_attention=False,name="relational_nested_trunk")(rel_xyz,rel_xyz,geometry,training)
            proto=self.param("ambiguity_prototypes",nn.initializers.normal(stddev=0.02),(self.num_classes,self.relation_dim))
            if training:
                feat_n=rel_feat/(jnp.linalg.norm(rel_feat,axis=-1,keepdims=True)+1e-8); proto_n=proto/(jnp.linalg.norm(proto,axis=-1,keepdims=True)+1e-8)
                ambiguity_logits=jnp.matmul(feat_n,proto_n.T)/0.125
            else:
                ambiguity_logits=jnp.zeros((x.shape[0],self.num_classes),dtype=trunk["logits"].dtype)
            result=dict(trunk); result["logits"]=trunk["logits"]; result["stream_logits"]=trunk["logits"][:,None,:]; result["fusion_weights"]=jnp.ones((1,),dtype=trunk["logits"].dtype)
            result["relation_gate"]=rel_gate; result["relation_rms"]=rel_rms; result["ambiguity_logits"]=ambiguity_logits; result["relation_feature_norm"]=jnp.mean(jnp.linalg.norm(rel_feat,axis=-1))
            return result
    return NestSARRelationalMemoryV1

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--protocol",choices=("xsub","xset"),required=True); p.add_argument("--epochs",type=int,default=20); p.add_argument("--batch",type=int,default=32); p.add_argument("--eval-batch",type=int,default=64); p.add_argument("--relation-dim",type=int,default=64); p.add_argument("--fr-weight",type=float,default=0.10); p.add_argument("--seed",type=int,default=42); p.add_argument("--peak-lr",type=float,default=6e-4); p.add_argument("--end-lr",type=float,default=2e-5); p.add_argument("--warmup-frac",type=float,default=0.08); p.add_argument("--weight-decay",type=float,default=0.03); p.add_argument("--label-smoothing",type=float,default=0.05); p.add_argument("--grad-clip",type=float,default=1.0); p.add_argument("--ema",type=float,default=0.998); p.add_argument("--patience",type=int,default=4); p.add_argument("--dataset",default="auto"); p.add_argument("--progress-json",action="store_true"); p.add_argument("--skip-xla-audit",action="store_true"); return p.parse_args()

def emit(a,payload):
    if a.progress_json: print("@@NESTSAR@@"+json.dumps(payload,separators=(",",":")),flush=True)

def xla_flops(model,params):
    dummy=jnp.zeros((1,FRAMES,150),jnp.float32)
    def f(p,x): return model.apply({"params":p},x,training=False)["logits"]
    try:
        compiled=jax.jit(f).lower(params,dummy).compile(); ca=compiled.cost_analysis()
        if isinstance(ca,(list,tuple)) and ca: ca=ca[0]
        if isinstance(ca,dict):
            for key in ("flops","FLOPs","flop_count"):
                if key in ca: return float(ca[key])
    except Exception: pass
    return float("nan")

def main():
    a=parse_args(); protocol=a.protocol; seed=a.seed; devices=list(jax.local_devices()); ndev=len(devices)
    if jax.default_backend()!="gpu" or ndev<1: raise RuntimeError(f"Expected GPU backend, got {jax.default_backend()} {devices}")
    if a.batch%ndev or a.eval_batch%ndev: raise RuntimeError("batch and eval-batch must be divisible by local device count")
    source=generated_source(protocol); mod=load_base_module(source,protocol); ns=mod.ns; Model=make_relational_memory_model(mod,a.relation_dim); base=mod.build_model()
    fields={name:getattr(base,name) for name in base.__dataclass_fields__ if name not in ("parent","name") and hasattr(base,name)}; fields["relation_dim"]=a.relation_dim; model=Model(**fields)
    dataset=find_dataset(a.dataset); raw=ns.load_pickle(dataset); train_samples,val_samples=ns.build_samples(raw,protocol=protocol,max_train=0,max_val=0,seed=seed)
    expected_train,expected_val=EXPECTED_COUNTS[protocol]
    if len(train_samples)!=expected_train or len(val_samples)!=expected_val: raise RuntimeError(f"Split mismatch train={len(train_samples)} val={len(val_samples)}")
    train_ds=ns.SkeletonDataset(train_samples); val_ds=ns.SkeletonDataset(val_samples); rng=jax.random.PRNGKey(seed); rng,init_rng,drop_rng=jax.random.split(rng,3)
    dummy=jnp.zeros((2,FRAMES,150),jnp.float32); params=model.init({"params":init_rng,"dropout":drop_rng},dummy,training=True)["params"]; param_count=tree_numel(params); leaf_count=tree_leaf_count(params); flops=float("nan") if a.skip_xla_audit else xla_flops(model,params)
    steps_per_epoch=len(train_ds)//a.batch; total_steps=steps_per_epoch*a.epochs; warmup_steps=max(1,int(total_steps*a.warmup_frac)); schedule=optax.warmup_cosine_decay_schedule(0.0,a.peak_lr,warmup_steps,total_steps,a.end_lr)
    tx=optax.chain(optax.clip_by_global_norm(a.grad_clip),optax.adamw(schedule,weight_decay=a.weight_decay,b1=0.9,b2=0.999,eps=1e-8)); opt_state=tx.init(params); ema_params=params
    def repl(tree): return jax.device_put_replicated(tree,devices)
    params_r=repl(params); ema_r=repl(ema_params); opt_r=repl(opt_state); local_train=a.batch//ndev; local_eval=a.eval_batch//ndev
    def smooth_ce(logits,labels):
        targets=jax.nn.one_hot(labels,NUM_CLASSES); targets=targets*(1-a.label_smoothing)+a.label_smoothing/NUM_CLASSES; return -jnp.mean(jnp.sum(targets*jax.nn.log_softmax(logits),axis=-1))
    def hard_ce(logits,labels): return -jnp.mean(jnp.take_along_axis(jax.nn.log_softmax(logits),labels[:,None],axis=-1)[:,0])
    @partial(jax.pmap,axis_name="gpu")
    def train_step(params,ema,opt_state,x,y,dropout_key):
        def loss_fn(p):
            out=model.apply({"params":p},x,training=True,rngs={"dropout":dropout_key}); ce=smooth_ce(out["logits"],y); probs=jax.nn.softmax(out["logits"],axis=-1); ptrue=jnp.take_along_axis(probs,y[:,None],axis=-1)[:,0]
            ambiguity_weight=jax.lax.stop_gradient(jnp.mean(1.0-ptrue)); fr=hard_ce(out["ambiguity_logits"],y); loss=ce+a.fr_weight*ambiguity_weight*fr; acc=jnp.mean(jnp.argmax(out["logits"],axis=-1)==y)
            return loss,(acc,ce,fr,ambiguity_weight,out["relation_gate"],out["relation_rms"])
        (loss,(acc,ce,fr,ambw,gate,rms)),grads=jax.value_and_grad(loss_fn,has_aux=True)(params); grads=jax.tree_util.tree_map(lambda g:jax.lax.pmean(g,"gpu"),grads); grad_norm=optax.global_norm(grads); updates,opt_state=tx.update(grads,opt_state,params); params=optax.apply_updates(params,updates); ema=jax.tree_util.tree_map(lambda e,p:a.ema*e+(1-a.ema)*p,ema,params)
        metrics={"loss":loss,"ce":ce,"fr":fr,"ambw":ambw,"acc":acc,"gate":gate,"rms":rms,"grad":grad_norm}; metrics=jax.tree_util.tree_map(lambda z:jax.lax.pmean(z,"gpu"),metrics); return params,ema,opt_state,metrics
    @partial(jax.pmap,axis_name="gpu")
    def eval_step(params,x,y,mask):
        out=model.apply({"params":params},x,training=False); pred=jnp.argmax(out["logits"],axis=-1); m=mask.astype(jnp.float32)
        return {"correct":jnp.sum((pred==y).astype(jnp.float32)*m),"count":jnp.sum(m),"gate_sum":jnp.sum(jnp.ones_like(m)*out["relation_gate"]*m),"rms_sum":jnp.sum(jnp.ones_like(m)*out["relation_rms"]*m)}
    def shard_train(x,y): return jnp.asarray(np.asarray(x,np.float32).reshape(ndev,local_train,FRAMES,150)),jnp.asarray(np.asarray(y,np.int32).reshape(ndev,local_train))
    def shard_eval(x,y):
        x=np.asarray(x,np.float32); y=np.asarray(y,np.int32); actual=len(y); pad=a.eval_batch-actual
        if pad<0: raise ValueError("eval batch too large")
        if pad: x=np.concatenate((x,np.zeros((pad,FRAMES,150),np.float32)),axis=0); y=np.concatenate((y,np.zeros((pad,),np.int32)),axis=0)
        mask=np.zeros((a.eval_batch,),np.float32); mask[:actual]=1; return jnp.asarray(x.reshape(ndev,local_eval,FRAMES,150)),jnp.asarray(y.reshape(ndev,local_eval)),jnp.asarray(mask.reshape(ndev,local_eval))
    out_dir=Path(f"/kaggle/working/NestSAR_RELATIONAL_MEMORY_V1_{protocol.upper()}_SEED{seed}"); out_dir.mkdir(parents=True,exist_ok=True)
    meta={"architecture":"NestSAR-Relational-Memory-v1","protocol":protocol,"frames":FRAMES,"seed":seed,"params":param_count,"leaves":leaf_count,"xla_forward_flops":flops,"xla_forward_gflops":None if math.isnan(flops) else flops/1e9,"relation_dim":a.relation_dim,"fr_weight":a.fr_weight,"train_count":len(train_ds),"val_count":len(val_ds)}; (out_dir/"run_meta.json").write_text(json.dumps(meta,indent=2)); emit(a,{"kind":"meta",**meta})
    history=[]; best=-1.0; bad=0; global_step=0
    for epoch in range(1,a.epochs+1):
        sums={k:0.0 for k in ("loss","ce","fr","ambw","acc","gate","rms","grad")}; nsteps=0; iterator=ns.batch_iterator(train_ds,batch_size=a.batch,shuffle=True,seed=seed+epoch,drop_last=True); bar=None if a.progress_json else tqdm(iterator,total=steps_per_epoch,desc=f"{protocol.upper()} RELMEM TRAIN E{epoch:03d}",leave=True,dynamic_ncols=True); iterator=bar if bar is not None else iterator
        for step,(x,y) in enumerate(iterator,1):
            xs,ys=shard_train(x,y); rng,sub=jax.random.split(rng); keys=jax.random.split(sub,ndev); params_r,ema_r,opt_r,m=train_step(params_r,ema_r,opt_r,xs,ys,keys); mm={k:float(np.asarray(v[0])) for k,v in m.items()}
            for k in sums: sums[k]+=mm[k]
            nsteps+=1; global_step+=1; running={k:sums[k]/nsteps for k in sums}
            if bar is not None: bar.set_postfix(loss=f"{running['loss']:.3f}",acc=f"{100*running['acc']:.2f}%",gate=f"{running['gate']:.3f}",grad=f"{running['grad']:.2f}",lr=f"{float(schedule(global_step)):.1e}")
            else: emit(a,{"kind":"progress","protocol":protocol,"phase":"train","epoch":epoch,"step":step,"total":steps_per_epoch,"loss":running["loss"],"acc":running["acc"],"gate":running["gate"],"rms":running["rms"],"grad":running["grad"]})
        trainm={k:sums[k]/max(1,nsteps) for k in sums}; correct=count=gate_sum=rms_sum=0.0; val_steps=(len(val_ds)+a.eval_batch-1)//a.eval_batch; viter=ns.batch_iterator(val_ds,batch_size=a.eval_batch,shuffle=False,seed=0,drop_last=False); vbar=None if a.progress_json else tqdm(viter,total=val_steps,desc=f"{protocol.upper()} RELMEM VAL   E{epoch:03d}",leave=True,dynamic_ncols=True); viter=vbar if vbar is not None else viter
        for step,(x,y) in enumerate(viter,1):
            xs,ys,ms=shard_eval(x,y); o=eval_step(ema_r,xs,ys,ms); c=float(np.asarray(o["correct"]).sum()); ct=float(np.asarray(o["count"]).sum()); gs=float(np.asarray(o["gate_sum"]).sum()); rs=float(np.asarray(o["rms_sum"]).sum()); correct+=c; count+=ct; gate_sum+=gs; rms_sum+=rs; vacc=correct/max(1.0,count)
            if vbar is not None: vbar.set_postfix(acc=f"{100*vacc:.2f}%",gate=f"{gate_sum/max(1.0,count):.3f}",rms=f"{rms_sum/max(1.0,count):.4f}",best=f"{100*max(best,vacc):.2f}%")
            else: emit(a,{"kind":"progress","protocol":protocol,"phase":"val","epoch":epoch,"step":step,"total":val_steps,"acc":vacc,"gate":gate_sum/max(1.0,count),"rms":rms_sum/max(1.0,count),"best":max(best,vacc)})
        val_acc=correct/max(1.0,count); val_gate=gate_sum/max(1.0,count); val_rms=rms_sum/max(1.0,count); record={"epoch":epoch,"train":trainm,"val":{"accuracy":val_acc,"correct":int(round(correct)),"count":int(round(count)),"gate":val_gate,"relation_rms":val_rms}}; history.append(record); (out_dir/"history.json").write_text(json.dumps(history,indent=2)); improved=val_acc>best
        if improved:
            best=val_acc; bad=0; host_ema=jax.tree_util.tree_map(lambda z:np.asarray(z[0]),ema_r); (out_dir/"best_ema.msgpack").write_bytes(serialization.to_bytes(host_ema)); (out_dir/"best.json").write_text(json.dumps({"epoch":epoch,"accuracy":best,"correct":int(round(correct)),"count":int(round(count))},indent=2))
        else: bad+=1
        emit(a,{"kind":"epoch","protocol":protocol,"epoch":epoch,"train_acc":trainm["acc"],"train_loss":trainm["loss"],"val_acc":val_acc,"best":best,"gate":val_gate,"rms":val_rms,"bad":bad})
        if bad>=a.patience: break
    emit(a,{"kind":"done","protocol":protocol,"best":best,"out_dir":str(out_dir)})

if __name__=="__main__": main()
