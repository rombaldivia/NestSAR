#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import math
import sys
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.attention_lite_v1.canonical_integrated import ensure_canonical_sources

FRAMES = 16
PERSONS = 2
JOINTS = 25
COORDS = 3
NUM_CLASSES = 120
EXPECTED_BASE_PARAMS = 2_381_028
EXPECTED_BASE_LEAVES = 705
PART_DIM = 64
PART_HEADS = 4
HAND_LEFT = 3
HAND_RIGHT = 5

TEN_BODY_PARTS = (
    ("torso_core",             (0, 1, 20)),
    ("head_neck",              (2, 3)),
    ("left_upper_arm",         (4, 5)),
    ("left_forearm_hand",      (6, 7, 21, 22)),
    ("right_upper_arm",        (8, 9)),
    ("right_forearm_hand",     (10, 11, 23, 24)),
    ("left_upper_leg",         (12, 13)),
    ("left_lower_leg_foot",    (14, 15)),
    ("right_upper_leg",        (16, 17)),
    ("right_lower_leg_foot",   (18, 19)),
)


def part_mask() -> jnp.ndarray:
    mask = np.zeros((10, JOINTS), np.float32)
    seen = []
    for p, (_, joints) in enumerate(TEN_BODY_PARTS):
        for j in joints:
            mask[p, j] = 1.0
            seen.append(j)
    if sorted(seen) != list(range(JOINTS)):
        raise RuntimeError("TEN_BODY_PARTS must cover joints 0..24 exactly once")
    return jnp.asarray(mask)


PART_MASK = part_mask()


def tree_numel(tree) -> int:
    return int(sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(tree)))


def tree_leaves(tree) -> int:
    return len(jax.tree_util.tree_leaves(tree))


def load_canonical_prefix(protocol: str):
    sources = ensure_canonical_sources(verbose=True)
    source = Path(sources[protocol]).resolve()
    text = source.read_text(encoding="utf-8")
    marker = "# 8. INITIALIZE"
    cut = text.find(marker)
    if cut < 0:
        raise RuntimeError(f"Could not find canonical marker {marker!r}")
    section = text.rfind("# ==========================================================================================", 0, cut)
    prefix = text[:section if section > 0 else cut]

    # Avoid stale embedded runtime modules if the notebook has executed another NestSAR source.
    for name in list(sys.modules):
        if name == "nestsar" or name.startswith("nestsar_"):
            del sys.modules[name]
    sys.path[:] = [p for p in sys.path if "NestSAR_HOPE_FIDELITY_UNIVERSAL" not in str(p)]

    name = f"_attention_lite_canonical_parttrace_v3_{protocol}"
    mod = types.ModuleType(name)
    mod.__file__ = str(source)
    mod.__package__ = None
    sys.modules[name] = mod
    exec(compile(prefix, str(source), "exec"), mod.__dict__)

    for symbol in ("NestSARHOPEAttentionLiteVec31", "m4", "ns", "build_model"):
        if not hasattr(mod, symbol):
            raise RuntimeError(f"Canonical Attention-Lite source missing {symbol}")
    return mod, source


class PartMixer10(nn.Module):
    dim: int = PART_DIM
    heads: int = PART_HEADS

    @nn.compact
    def __call__(self, x):
        # [B,T,M,10,D]
        b, t, m, p, d = x.shape
        if p != 10 or d != self.dim or self.dim % self.heads:
            raise ValueError(f"Unexpected part tensor {x.shape}")
        dh = d // self.heads
        z = nn.LayerNorm(name="norm")(x)
        q = nn.Dense(d, use_bias=False, name="q")(z)
        k = nn.Dense(d, use_bias=False, name="k")(z)
        v = nn.Dense(d, use_bias=False, name="v")(z)
        q = q.reshape(b,t,m,p,self.heads,dh).transpose(0,1,2,4,3,5)
        k = k.reshape(b,t,m,p,self.heads,dh).transpose(0,1,2,4,3,5)
        v = v.reshape(b,t,m,p,self.heads,dh).transpose(0,1,2,4,3,5)
        logits = jnp.einsum("btmhpd,btmhkd->btmhpk", q, k) / math.sqrt(dh)
        attn = jax.nn.softmax(logits, axis=-1)
        ctx = jnp.einsum("btmhpk,btmhkd->btmhpd", attn, v)
        ctx = ctx.transpose(0,1,2,4,3,5).reshape(b,t,m,p,d)
        ctx = nn.Dense(d, name="out")(ctx)
        gate = jax.nn.sigmoid(self.param("gate_logit", nn.initializers.constant(-2.944439), (1,)))[0]
        return x + gate * ctx


class SharedPartTemporal(nn.Module):
    dim: int = PART_DIM
    heads: int = PART_HEADS

    @nn.compact
    def __call__(self, x):
        # [N,T,D], shared weights over N = B*M*parts
        n, t, d = x.shape
        dh = d // self.heads
        h = nn.LayerNorm(name="pre_norm")(x)
        hpad = jnp.pad(h, ((0,0),(2,0),(0,0)))
        local = nn.Conv(
            features=d,
            kernel_size=(3,),
            padding="VALID",
            feature_group_count=d,
            use_bias=True,
            name="dwconv",
        )(hpad)
        x = x + jax.nn.sigmoid(self.param("local_gate", nn.initializers.constant(-2.944439), (1,)))[0] * jax.nn.silu(local)
        z = nn.LayerNorm(name="attn_norm")(x)
        q = nn.Dense(d, use_bias=False, name="q")(z)
        k = nn.Dense(d, use_bias=False, name="k")(z)
        v = nn.Dense(d, use_bias=False, name="v")(z)
        q = q.reshape(n,t,self.heads,dh).transpose(0,2,1,3)
        k = k.reshape(n,t,self.heads,dh).transpose(0,2,1,3)
        v = v.reshape(n,t,self.heads,dh).transpose(0,2,1,3)
        logits = jnp.einsum("nhtd,nhkd->nhtk", q, k) / math.sqrt(dh)
        rel = self.param("relative_time_bias", nn.initializers.zeros, (self.heads, FRAMES))
        qpos = jnp.arange(t)[:,None]
        kpos = jnp.arange(t)[None,:]
        dist = jnp.clip(qpos-kpos, 0, FRAMES-1)
        logits = logits + rel[:,dist][None,...]
        causal = kpos <= qpos
        logits = jnp.where(causal[None,None,:,:], logits, -1e9)
        attn = jax.nn.softmax(logits, axis=-1)
        ctx = jnp.einsum("nhtk,nhkd->nhtd", attn, v)
        ctx = ctx.transpose(0,2,1,3).reshape(n,t,d)
        ctx = nn.Dense(d, name="out")(ctx)
        x = x + jax.nn.sigmoid(self.param("attn_gate", nn.initializers.constant(-1.734601), (1,)))[0] * ctx
        # lightweight second residual FFN: adds capacity without duplicating temporal weights per part
        h = nn.LayerNorm(name="ffn_norm")(x)
        h = nn.Dense(2*d, name="ffn_in")(h)
        h = jax.nn.gelu(h)
        h = nn.Dense(d, name="ffn_out")(h)
        return nn.LayerNorm(name="out_norm")(x + 0.20 * h)


class PartTraceResidualBranch(nn.Module):
    num_classes: int = NUM_CLASSES
    dim: int = PART_DIM

    @nn.compact
    def __call__(self, x):
        # x [B,T,150]
        b, t, _ = x.shape
        xyz = x.reshape(b,t,PERSONS,JOINTS,COORDS)
        person_present = jnp.any(jnp.abs(xyz) > 1e-6, axis=(3,4))  # [B,T,M]
        joint_present = person_present[...,None,None].astype(xyz.dtype)
        root = xyz[:,:,:,0:1,:]
        centered = (xyz-root) * joint_present

        pair_valid = (person_present[:,1:] & person_present[:,:-1])[...,None,None].astype(xyz.dtype)
        velocity = jnp.concatenate(
            [jnp.zeros_like(centered[:,:1]), (centered[:,1:]-centered[:,:-1]) * pair_valid],
            axis=1,
        )
        acceleration = jnp.concatenate(
            [jnp.zeros_like(velocity[:,:1]), (velocity[:,1:]-velocity[:,:-1]) * pair_valid],
            axis=1,
        )

        features = jnp.concatenate([centered, velocity, acceleration], axis=-1)  # 9 channels
        h = nn.Dense(self.dim, name="joint_projection")(features)
        joint_emb = self.param("joint_embedding", nn.initializers.normal(0.02), (1,1,1,JOINTS,self.dim))
        person_emb = self.param("person_embedding", nn.initializers.normal(0.02), (1,1,PERSONS,1,self.dim))
        h = jax.nn.gelu(nn.LayerNorm(name="joint_norm")(h + joint_emb + person_emb))
        h = h * joint_present

        mask = PART_MASK.astype(h.dtype)
        numerator = jnp.einsum("btmjd,pj->btmpd", h, mask)
        counts = jnp.sum(mask, axis=1)[None,None,None,:,None]
        parts = numerator / jnp.maximum(counts, 1.0)
        part_emb = self.param("part_embedding", nn.initializers.normal(0.02), (1,1,1,10,self.dim))
        part_present = person_present[...,None,None].astype(parts.dtype)
        parts = nn.LayerNorm(name="part_norm")(parts + part_emb) * part_present
        parts = PartMixer10(self.dim, PART_HEADS, name="part_mixer")(parts) * part_present

        tracks = parts.transpose(0,2,3,1,4).reshape(b*PERSONS*10,t,self.dim)
        tracks = SharedPartTemporal(self.dim, PART_HEADS, name="shared_part_temporal")(tracks)
        parts = tracks.reshape(b,PERSONS,10,t,self.dim).transpose(0,3,1,2,4) * part_present

        # preserve fine hand identity explicitly
        left = parts[:,:,:,HAND_LEFT,:]
        right = parts[:,:,:,HAND_RIGHT,:]
        present_w = person_present[...,None].astype(parts.dtype)
        denom = jnp.maximum(jnp.sum(present_w, axis=2), 1.0)
        left = jnp.sum(left * present_w, axis=2) / denom
        right = jnp.sum(right * present_w, axis=2) / denom

        gate_logits = nn.Dense(1, name="part_pool_gate")(parts)[...,0]
        gate_logits = jnp.where(person_present[...,None], gate_logits, -1e9)
        part_w = jax.nn.softmax(gate_logits, axis=3)
        person_desc = jnp.sum(part_w[...,None] * parts, axis=3)
        first, second = person_desc[:,:,0], person_desc[:,:,1]
        pair = jnp.concatenate([first+second, jnp.abs(first-second), first*second], axis=-1)
        global_trace = nn.Dense(128, name="global_projection")(pair)

        # learned temporal pooling, while hand traces stay explicit
        temporal_scores = nn.Dense(1, name="temporal_gate")(global_trace)[...,0]
        any_person = jnp.any(person_present, axis=2)
        temporal_scores = jnp.where(any_person, temporal_scores, -1e9)
        tw = jax.nn.softmax(temporal_scores, axis=1)
        global_pooled = jnp.sum(tw[...,None] * global_trace, axis=1)
        left_pooled = jnp.sum(tw[...,None] * left, axis=1)
        right_pooled = jnp.sum(tw[...,None] * right, axis=1)
        fused = jnp.concatenate([global_pooled, left_pooled, right_pooled], axis=-1)
        fused = jax.nn.gelu(nn.Dense(192, name="fusion_hidden")(fused))
        return nn.Dense(self.num_classes, kernel_init=nn.initializers.normal(1e-3), name="branch_classifier")(fused)


def make_wrapper(base_model):
    class AttentionLitePartTraceV3(nn.Module):
        base: nn.Module

        @nn.compact
        def __call__(self, x, training=False):
            base_out = self.base(x, training=training)
            branch_logits = PartTraceResidualBranch(name="parttrace_branch")(x)
            gate = jax.nn.sigmoid(
                self.param("parttrace_fusion_gate_logit", nn.initializers.constant(-2.944439), (1,))
            )[0]
            result = dict(base_out)
            result["base_logits"] = base_out["logits"]
            result["parttrace_logits"] = branch_logits
            result["parttrace_gate"] = gate
            result["logits"] = base_out["logits"] + gate * branch_logits
            return result

    return AttentionLitePartTraceV3(base=base_model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=("xsub","xset"), default="xsub")
    args = ap.parse_args()

    mod, source = load_canonical_prefix(args.protocol)
    base = mod.build_model()
    rng = jax.random.PRNGKey(128)
    dummy = jnp.zeros((1,FRAMES,PERSONS*JOINTS*COORDS), jnp.float32)

    base_vars = base.init({"params":rng,"dropout":rng}, dummy, training=True)
    base_params = base_vars["params"]
    bp, bl = tree_numel(base_params), tree_leaves(base_params)
    if bp != EXPECTED_BASE_PARAMS or bl != EXPECTED_BASE_LEAVES:
        raise RuntimeError(
            f"ATTENTION-LITE BASE GUARD FAILED: params={bp:,}/{EXPECTED_BASE_PARAMS:,}, leaves={bl}/{EXPECTED_BASE_LEAVES}"
        )

    model = make_wrapper(base)
    rng, init_rng, drop_rng = jax.random.split(rng, 3)
    variables = model.init({"params":init_rng,"dropout":drop_rng}, dummy, training=True)
    params = variables["params"]
    total = tree_numel(params)
    leaves = tree_leaves(params)
    added = total - bp

    print("="*120)
    print("NESTSAR-HOPE ATTENTION-LITE + PARTTRACE V3 — STRICT AUDIT")
    print("="*120)
    print("Canonical source:", source)
    print("Backend:         ", jax.default_backend())
    print("Devices:         ", jax.devices())
    print("Base params:     ", f"{bp:,}")
    print("Base leaves:     ", bl)
    print("Added params:    ", f"{added:,}")
    print("Total params:    ", f"{total:,}")
    print("Total leaves:    ", leaves)
    print("Part dim:        ", PART_DIM)
    print("Part heads:      ", PART_HEADS)

    fwd = jax.jit(lambda p,xx: model.apply({"params":p}, xx, training=False)["logits"])
    compiled = fwd.lower(params, dummy).compile()
    cost = compiled.cost_analysis()
    if isinstance(cost, list) and cost:
        cost = cost[0]
    flops = float(cost.get("flops",0.0)) if isinstance(cost,dict) else 0.0
    print("XLA GFLOPs:      ", f"{flops/1e9:.9f}" if flops else "unavailable")

    out = model.apply({"params":params}, dummy, training=False)
    if out["logits"].shape != (1,NUM_CLASSES):
        raise RuntimeError(f"Bad logits shape: {out['logits'].shape}")
    if not bool(np.asarray(jnp.all(jnp.isfinite(out["logits"])))):
        raise FloatingPointError("Non-finite logits")

    print("Initial gate:    ", float(np.asarray(out["parttrace_gate"])))
    print("Smoke logits:    ", out["logits"].shape)
    print("AUDIT PASS")
    print("="*120)


if __name__ == "__main__":
    main()
