#!/usr/bin/env python3
from __future__ import annotations
import math
import numpy as np
import jax, jax.numpy as jnp
from flax import linen as nn

NUM_CLASSES=120
FRAMES=16
PERSONS=2
JOINTS=25
COORDS=3
PARTS=6
NTU_EDGES=((0,1),(1,20),(20,2),(2,3),(20,4),(4,5),(5,6),(6,7),(7,21),(7,22),(20,8),(8,9),(9,10),(10,11),(11,23),(11,24),(0,12),(12,13),(13,14),(14,15),(0,16),(16,17),(17,18),(18,19))
PART_IDS=((2,3),(0,1,20),(4,5,6,7,21,22),(8,9,10,11,23,24),(12,13,14,15),(16,17,18,19))
SUPPORT_M4=None

def set_support_m4(m4):
    global SUPPORT_M4
    SUPPORT_M4=m4

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
        idx_np,mask_np=RELATION_TABLES[self.branch]; idx=jnp.asarray(idx_np,jnp.int32); mask=jnp.asarray(mask_np,jnp.bool_)
        kg=jnp.take(k,idx,axis=1); vg=jnp.take(v,idx,axis=1)
        score=jnp.einsum("bnd,bnkd->bnk",q,kg)/jnp.sqrt(jnp.asarray(self.head_dim,jnp.float32))
        score=jnp.where(mask[None,:,:],score,jnp.asarray(-1e9,score.dtype)); attn=jax.nn.softmax(score,axis=-1)
        return jnp.einsum("bnk,bnkd->bnd",attn,vg)

class AdaptiveRelationalBlock(nn.Module):
    token_dim:int=160
    rel_dim:int=80
    dropout:float=0.15
    @nn.compact
    def __call__(self,z,training=False):
        if self.rel_dim%4: raise ValueError("rel_dim must be divisible by 4")
        b,n,_=z.shape; hd=self.rel_dim//4; norm=nn.LayerNorm(name="pre_ln")(z)
        q=nn.Dense(self.rel_dim,use_bias=False,name="q")(norm); k=nn.Dense(self.rel_dim,use_bias=False,name="k")(norm); v=nn.Dense(self.rel_dim,use_bias=False,name="v")(norm)
        qh=q.reshape(b,n,4,hd); kh=k.reshape(b,n,4,hd); vh=v.reshape(b,n,4,hd)
        route=jax.nn.softmax(nn.Dense(4,kernel_init=nn.initializers.zeros,bias_init=nn.initializers.zeros,name="route")(jnp.mean(norm,axis=1)),axis=-1)
        outs=[]
        for h,name in enumerate(("ll","dl","lg","dg")):
            ctx=SparseRelationBranch(name,hd,name=f"rel_{name}")(qh[:,:,h,:],kh[:,:,h,:],vh[:,:,h,:]); outs.append(ctx*route[:,None,h,None])
        mixed=jnp.concatenate(outs,axis=-1); delta=nn.Dense(self.token_dim,name="rel_out")(mixed); delta=nn.Dropout(self.dropout,name="rel_drop")(delta,deterministic=not training)
        gate=jax.nn.sigmoid(self.param("relation_gate_logit",nn.initializers.constant(-2.1972246),(1,)))[0]; z=nn.LayerNorm(name="post_rel_ln")(z+gate*delta)
        ff=nn.Dense(self.rel_dim,name="ff_down")(z); ff=nn.gelu(ff); ff=nn.Dense(self.token_dim,name="ff_up")(ff); ff=nn.Dropout(self.dropout,name="ff_drop")(ff,deterministic=not training)
        ff_gate=jax.nn.sigmoid(self.param("ff_gate_logit",nn.initializers.constant(-2.944439),(1,)))[0]; out=nn.LayerNorm(name="post_ff_ln")(z+ff_gate*ff)
        return out,gate,ff_gate,jnp.sqrt(jnp.mean(jnp.square(gate*delta))+1e-12),route

class RelationalTokenEncoder(nn.Module):
    token_dim:int=160
    rel_dim:int=80
    relation_blocks:int=2
    dropout:float=0.15
    @nn.compact
    def __call__(self,joint,bone,joint_motion,bone_motion,training=False):
        acceleration=jnp.concatenate((jnp.zeros_like(joint_motion[:,:1]),joint_motion[:,1:]-joint_motion[:,:-1]),axis=1)
        raw=jnp.concatenate((joint,bone,joint_motion,bone_motion,acceleration),axis=-1); b=raw.shape[0]
        z=nn.Dense(self.token_dim,name="modal_fuse")(raw)
        te=self.param("time_embedding",nn.initializers.normal(stddev=0.02),(FRAMES,self.token_dim)); pe=self.param("person_embedding",nn.initializers.normal(stddev=0.02),(PERSONS,self.token_dim)); je=self.param("joint_embedding",nn.initializers.normal(stddev=0.02),(JOINTS,self.token_dim))
        z=z+te[None,:,None,None,:]+pe[None,None,:,None,:]+je[None,None,None,:,:]; z=nn.gelu(nn.LayerNorm(name="embed_ln")(z)); z=nn.Dropout(self.dropout,name="embed_drop")(z,deterministic=not training); z=z.reshape(b,FRAMES*PERSONS*JOINTS,self.token_dim)
        gates=[]; ff_gates=[]; rms=[]; routes=[]
        for i in range(self.relation_blocks):
            z,g,fg,r,rt=AdaptiveRelationalBlock(self.token_dim,self.rel_dim,self.dropout,name=f"rel_block_{i}")(z,training); gates.append(g); ff_gates.append(fg); rms.append(r); routes.append(rt)
        return z.reshape(b,FRAMES,PERSONS,JOINTS,self.token_dim),jnp.stack(gates),jnp.stack(ff_gates),jnp.stack(rms),jnp.stack(routes,axis=1)

class AnatomicalPartPool(nn.Module):
    token_dim:int=160
    @nn.compact
    def __call__(self,z):
        p=jnp.stack([jnp.mean(jnp.take(z,jnp.asarray(ids,jnp.int32),axis=3),axis=3) for ids in PART_IDS],axis=3)
        emb=self.param("part_embedding",nn.initializers.normal(stddev=0.02),(PARTS,self.token_dim)); scale=self.param("part_scale",nn.initializers.ones,(PARTS,))
        return nn.LayerNorm(name="part_ln")(p+emb[None,None,None,:,:])*scale[None,None,None,:,None]

class NestedMemoryBlock(nn.Module):
    token_dim:int=160
    memory_dim:int=96
    alpha_init:float=0.9
    dropout:float=0.15
    @nn.compact
    def __call__(self,x,training=False):
        b,t,s,_=x.shape; norm=nn.LayerNorm(name="pre_ln")(x); packed=nn.Dense(3*self.memory_dim,name="write_pack")(norm); cand,eta,read=jnp.split(packed,3,axis=-1); cand=jnp.tanh(cand); eta=jax.nn.sigmoid(eta); read=jax.nn.sigmoid(read)
        seq_c=cand.transpose(1,0,2,3).reshape(t,b*s,self.memory_dim); seq_e=eta.transpose(1,0,2,3).reshape(t,b*s,self.memory_dim); seq_r=read.transpose(1,0,2,3).reshape(t,b*s,self.memory_dim)
        init_logit=math.log(self.alpha_init/(1-self.alpha_init)); alpha=jax.nn.sigmoid(self.param("alpha_logit",nn.initializers.constant(init_logit),(self.memory_dim,)))
        def body(mem,inp):
            c,e,r=inp; mem=alpha*mem+(1-alpha)*(e*c); return mem,mem*r
        _,outs=jax.lax.scan(body,jnp.zeros((b*s,self.memory_dim),x.dtype),(seq_c,seq_e,seq_r)); outs=outs.reshape(t,b,s,self.memory_dim).transpose(1,0,2,3)
        delta=nn.Dense(self.token_dim,name="read_out")(outs); delta=nn.Dropout(self.dropout,name="drop")(delta,deterministic=not training); gate=jax.nn.sigmoid(self.param("residual_gate_logit",nn.initializers.constant(-1.734601),(1,)))[0]
        return nn.LayerNorm(name="post_ln")(x+gate*delta),gate,jnp.mean(alpha),jnp.sqrt(jnp.mean(jnp.square(gate*delta))+1e-12)

class NestedStage(nn.Module):
    token_dim:int=160
    memory_dim:int=96
    blocks:int=2
    alpha_init:float=0.9
    dropout:float=0.15
    @nn.compact
    def __call__(self,x,training=False):
        gs=[]; al=[]; rs=[]
        for i in range(self.blocks):
            x,g,a,r=NestedMemoryBlock(self.token_dim,self.memory_dim,self.alpha_init,self.dropout,name=f"mem_{i}")(x,training); gs.append(g); al.append(a); rs.append(r)
        return x,jnp.stack(gs),jnp.stack(al),jnp.stack(rs)

def temporal_pool(x,factor):
    b,t,s,d=x.shape
    if t%factor: raise ValueError(f"Cannot pool T={t} by {factor}")
    return jnp.mean(x.reshape(b,t//factor,factor,s,d),axis=2)

class NestSARRelationalMemoryV2(nn.Module):
    num_classes:int=NUM_CLASSES
    token_dim:int=160
    rel_dim:int=80
    relation_blocks:int=2
    memory_dim:int=96
    controller_rank:int=48
    frame_blocks:int=2
    chunk_blocks:int=2
    clip_blocks:int=2
    controller_blocks:int=2
    dropout:float=0.15
    @nn.compact
    def __call__(self,x,training=False):
        if SUPPORT_M4 is None: raise RuntimeError("Call set_support_m4() before model init")
        streams=dict(SUPPORT_M4.build_four_streams(x)); z,rg,fg,rr,routes=RelationalTokenEncoder(self.token_dim,self.rel_dim,self.relation_blocks,self.dropout,name="relational_encoder")(streams["joint"],streams["bone"],streams["joint_motion"],streams["bone_motion"],training)
        p=AnatomicalPartPool(self.token_dim,name="part_pool")(z); b,t,m,parts,d=p.shape; l1=p.reshape(b,t,m*parts,d)
        l1,g1,a1,r1=NestedStage(self.token_dim,self.memory_dim,self.frame_blocks,0.80,self.dropout,name="frame_memory")(l1,training); l2=temporal_pool(l1,4); l2,g2,a2,r2=NestedStage(self.token_dim,self.memory_dim,self.chunk_blocks,0.90,self.dropout,name="chunk_memory")(l2,training); l3=temporal_pool(l2,2); l3,g3,a3,r3=NestedStage(self.token_dim,self.memory_dim,self.clip_blocks,0.97,self.dropout,name="clip_memory")(l3,training); l4=temporal_pool(l3,2); l4,g4,a4,r4=NestedStage(self.token_dim,self.memory_dim,self.controller_blocks,0.99,self.dropout,name="controller_memory")(l4,training)
        pooled=jnp.stack((jnp.mean(l1,axis=(1,2)),jnp.mean(l2,axis=(1,2)),jnp.mean(l3,axis=(1,2)),jnp.mean(l4,axis=(1,2))),axis=1); dyn=nn.gelu(nn.Dense(self.controller_rank,name="fusion_down")(pooled.reshape(b,-1))); dyn=nn.Dense(4,kernel_init=nn.initializers.zeros,bias_init=nn.initializers.zeros,name="fusion_up")(dyn); static=self.param("level_fusion_logits",nn.initializers.zeros,(4,)); level_weights=jax.nn.softmax(dyn+static[None,:],axis=-1)
        fused=jnp.einsum("bl,bld->bd",level_weights,pooled); fused=nn.LayerNorm(name="final_ln")(fused); fused=nn.Dropout(self.dropout,name="final_drop")(fused,deterministic=not training); logits=nn.Dense(self.num_classes,name="classifier")(fused)
        if training:
            proto=self.param("class_prototypes",nn.initializers.normal(stddev=0.02),(self.num_classes,self.token_dim)); fn=fused/(jnp.linalg.norm(fused,axis=-1,keepdims=True)+1e-8); pn=proto/(jnp.linalg.norm(proto,axis=-1,keepdims=True)+1e-8); proto_logits=jnp.matmul(fn,pn.T)/0.125
            start=jnp.mean(z[:,:4],axis=(1,2,3)); end=jnp.mean(z[:,-4:],axis=(1,2,3)); fwd=jnp.concatenate((start,end,end-start),axis=-1); rev=jnp.concatenate((end,start,start-end),axis=-1); dh=nn.gelu(nn.Dense(self.controller_rank,name="direction_down")(jnp.concatenate((fwd,rev),axis=0))); direction_logits=nn.Dense(2,name="direction_head")(dh)
        else:
            proto_logits=jnp.zeros((x.shape[0],self.num_classes),logits.dtype); direction_logits=jnp.zeros((2*x.shape[0],2),logits.dtype)
        return {"logits":logits,"proto_logits":proto_logits,"direction_logits":direction_logits,"relation_gate":jnp.mean(rg),"relation_ff_gate":jnp.mean(fg),"relation_rms":jnp.mean(rr),"relation_routes":jnp.mean(routes,axis=(0,1)),"level_weights":jnp.mean(level_weights,axis=0),"nested_gate":jnp.mean(jnp.concatenate((g1,g2,g3,g4))),"nested_alpha":jnp.mean(jnp.concatenate((a1,a2,a3,a4))),"nested_rms":jnp.mean(jnp.concatenate((r1,r2,r3,r4)))}
