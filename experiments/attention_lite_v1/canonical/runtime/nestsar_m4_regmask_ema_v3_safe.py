# -*- coding: utf-8 -*-
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple
import jax
import jax.numpy as jnp
from flax import serialization
import nestsar as ns
import nestsar_m4_geom_h4 as m4

EMA_DECAY = 0.995
FRAME_MASK_PROB = 0.08
JOINT_MASK_PROB = 0.08
PART_MASK_PROB = 0.03
PROTECTED_JOINTS = (0, 20)

_BASE_CREATE_STATE = ns.create_state
_BASE_BUILD_STEPS = m4.build_steps
_BASE_EVALUATE = m4.evaluate

class EMAState(ns.TrainState):
    ema_params: Any = None

def create_state(rng, model, total_steps):
    base = _BASE_CREATE_STATE(rng, model, total_steps)
    return EMAState(
        step=base.step, apply_fn=base.apply_fn, params=base.params,
        tx=base.tx, opt_state=base.opt_state, ema_params=base.params,
    )

def skeleton_mask(batch_x: jnp.ndarray, rng: jax.Array) -> jnp.ndarray:
    b, t, _ = batch_x.shape
    x = batch_x.reshape(b, t, ns.CFG.persons, ns.CFG.joints, ns.CFG.coords)
    previous = jnp.concatenate([x[:, :1], x[:, :-1]], axis=1)
    k_frame, k_joint, k_part = jax.random.split(rng, 3)
    frame_drop = jax.random.bernoulli(k_frame, p=FRAME_MASK_PROB, shape=(b,t,1,1,1))
    frame_drop = frame_drop.at[:,0,...].set(False)
    joint_drop = jax.random.bernoulli(
        k_joint, p=JOINT_MASK_PROB, shape=(b,t,ns.CFG.persons,ns.CFG.joints,1)
    )
    part_mask = m4.sms.PART_MASK.astype(jnp.float32)
    num_parts = int(part_mask.shape[0])
    part_drop = jax.random.bernoulli(
        k_part, p=PART_MASK_PROB, shape=(b,ns.CFG.persons,num_parts)
    ).astype(jnp.float32)
    joint_part_drop = jnp.einsum("bmp,pv->bmv", part_drop, part_mask) > 0.0
    joint_part_drop = joint_part_drop[:,None,:,:,None]
    for joint_index in PROTECTED_JOINTS:
        if joint_index < ns.CFG.joints:
            joint_drop = joint_drop.at[:,:,:,joint_index,:].set(False)
            joint_part_drop = joint_part_drop.at[:,:,:,joint_index,:].set(False)
    valid_joint = jnp.any(jnp.abs(x) > 1e-6, axis=-1, keepdims=True)
    spatial_drop = (joint_drop | joint_part_drop) & valid_joint
    corrupted = jnp.where(spatial_drop, previous, x)
    corrupted = jnp.where(frame_drop, previous, corrupted)
    corrupted = jnp.nan_to_num(corrupted, nan=0.0, posinf=0.0, neginf=0.0)
    return corrupted.reshape(batch_x.shape)

def build_steps(model, model_id: str):
    base_train_step, base_eval_step = _BASE_BUILD_STEPS(model, model_id)
    accum_steps = max(1, int(getattr(ns.CFG, "grad_accum_steps", 1)))

    @jax.jit
    def train_step(state, batch_x, batch_y, dropout_rng):
        mask_rng, model_rng = jax.random.split(dropout_rng)
        masked_x = skeleton_mask(batch_x, mask_rng)
        new_state, metrics = base_train_step(state, masked_x, batch_y, model_rng)
        finite_flags = [jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(new_state.params)]
        params_finite = jnp.all(jnp.stack(finite_flags))
        new_state = jax.lax.cond(params_finite, lambda _: new_state, lambda _: state, operand=None)
        metrics = dict(metrics)
        metrics["finite_update"] = params_finite.astype(jnp.float32)
        do_ema = (new_state.step % accum_steps) == 0
        def update_ema(_):
            return jax.tree_util.tree_map(
                lambda ema,param: EMA_DECAY*ema + (1.0-EMA_DECAY)*param,
                state.ema_params, new_state.params
            )
        ema_params = jax.lax.cond(do_ema, update_ema, lambda _: state.ema_params, operand=None)
        new_state = new_state.replace(ema_params=ema_params)
        return new_state, metrics
    return train_step, base_eval_step

def evaluate(state, dataset, eval_step, progress_desc=None):
    eval_state = state.replace(params=state.ema_params)
    try:
        return _BASE_EVALUATE(eval_state, dataset, eval_step, progress_desc=progress_desc)
    except TypeError as exc:
        if "progress_desc" not in str(exc):
            raise
        return _BASE_EVALUATE(eval_state, dataset, eval_step)

def save_full_checkpoint(prefix: Path, state: EMAState, rng: jax.Array, metadata: Mapping[str, Any]) -> None:
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "params": state.params, "ema_params": state.ema_params,
        "opt_state": state.opt_state, "step": state.step, "rng": rng,
    }
    prefix.with_suffix(".msgpack").write_bytes(serialization.to_bytes(payload))
    meta = dict(metadata)
    meta.update({
        "regularization_variant":"strong_masking_ema_v3_safe",
        "ema_decay":EMA_DECAY,
        "frame_mask_prob":FRAME_MASK_PROB,
        "joint_mask_prob":JOINT_MASK_PROB,
        "part_mask_prob":PART_MASK_PROB,
        "checkpoint_contains_ema":True,
    })
    prefix.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

def restore_full_checkpoint(prefix: Path, state: EMAState, rng: jax.Array) -> Tuple[EMAState,jax.Array,Dict[str,Any]]:
    prefix = Path(prefix)
    metadata_path = prefix.with_suffix(".json")
    payload_path = prefix.with_suffix(".msgpack")
    if not metadata_path.is_file() or not payload_path.is_file():
        raise FileNotFoundError(f"Checkpoint incompleto: {prefix}")
    template = {
        "params": state.params, "ema_params": state.ema_params,
        "opt_state": state.opt_state, "step": state.step, "rng": rng,
    }
    restored = serialization.from_bytes(template, payload_path.read_bytes())
    state = state.replace(
        params=restored["params"], ema_params=restored["ema_params"],
        opt_state=restored["opt_state"], step=restored["step"]
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return state, restored["rng"], metadata

ns.create_state = create_state
m4.build_steps = build_steps
ns.build_steps = build_steps
m4.evaluate = evaluate
ns.evaluate = evaluate
ns.save_full_checkpoint = save_full_checkpoint
ns.restore_full_checkpoint = restore_full_checkpoint
ns.__file__ = __file__

print("="*100)
print("M4G-H4 STRONG REG + SAFE TEMPORAL-HOLD MASKING + EMA — v3")
print("="*100)
print(f"EMA decay:       {EMA_DECAY}")
print(f"Frame hold-mask: {FRAME_MASK_PROB:.0%}")
print(f"Joint hold-mask: {JOINT_MASK_PROB:.0%}")
print(f"Part hold-mask:  {PART_MASK_PROB:.0%}")
print("Validación:      pesos EMA")
print("Checkpoint:      online + EMA + optimizer")
print("="*100)

if __name__ == "__main__":
    raise SystemExit(ns.main())
