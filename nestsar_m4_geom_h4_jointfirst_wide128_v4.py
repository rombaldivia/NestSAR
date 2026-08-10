# -*- coding: utf-8 -*-
from __future__ import annotations

"""
NestSAR M4G-H4 JointFirst-Wide128 v4
====================================

Objetivo:
  25 joint tokens
    -> orden cinemático
    -> SpatialMemorySweep sobre 25 joints
    -> restaurar orden NTU
    -> pooling aprendido 25 -> 10 partes
    -> descriptor espacial 128-D (no 32-D)
    -> M4G-H4 temporal original

Sin attention, GCN/GNN/CNN/TCN ni canonicalización.

Este wrapper también elimina de forma segura el guard histórico P==5 de
SpatialMemorySweep ANTES de importar el módulo SMS. Se crea un backup del
archivo original. No cambia la matemática del sweep; sólo permite P dinámico.
"""

from pathlib import Path
import ast
import re
import shutil
import sys
import importlib


def _prepare_dynamic_sms() -> None:
    root = Path(__file__).resolve().parent
    path = root / "nestsar_sms_s1c_v2.py"
    if not path.is_file():
        raise FileNotFoundError(f"No existe {path}")

    text = path.read_text(encoding="utf-8")
    backup = path.with_suffix(path.suffix + ".bak_before_dynamic_sms")
    pattern = re.compile(
        r'(?m)^(?P<indent>[ \t]*)if\s+parts\s*!=\s*5\s*:\s*\n'
        r'(?P=indent)[ \t]+raise\s+ValueError\('
        r'f?["\']Se esperaban 5 partes; recibido P=\{parts\}["\']\)\s*$'
    )
    matches = list(pattern.finditer(text))

    if len(matches) > 1:
        raise RuntimeError(
            f"Encontré {len(matches)} guards `parts != 5`; no parcheo automáticamente."
        )

    if len(matches) == 1:
        if not backup.exists():
            shutil.copy2(path, backup)
        m = matches[0]
        indent = m.group("indent")
        replacement = (
            f"{indent}# Dynamic-SMS: P puede ser 5, 10, 25, etc.\n"
            f"{indent}if parts < 1:\n"
            f"{indent}    raise ValueError("
            f'f"Se esperaba al menos 1 token espacial; recibido P={{parts}}"'
            f")"
        )
        text = text[:m.start()] + replacement + text[m.end():]
        print("[Dynamic-SMS] Guard histórico P=5 eliminado.")
    else:
        print("[Dynamic-SMS] No encontré el guard histórico P=5; auditando versión actual...")

    tree = ast.parse(text, filename=str(path))
    suspicious = []
    found_class = False
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "SpatialMemorySweep":
            found_class = True
            for sub in ast.walk(node):
                if isinstance(sub, ast.Compare):
                    names = {n.id for n in ast.walk(sub) if isinstance(n, ast.Name)}
                    has_five = any(
                        isinstance(n, ast.Constant) and n.value == 5
                        for n in ast.walk(sub)
                    )
                    if "parts" in names and has_five:
                        suspicious.append((getattr(sub, "lineno", -1), "comparación de `parts` con 5"))
                if (
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Name)
                    and sub.func.id == "range"
                    and len(sub.args) == 1
                    and isinstance(sub.args[0], ast.Constant)
                    and sub.args[0].value == 5
                ):
                    suspicious.append((getattr(sub, "lineno", -1), "range(5)"))

    if not found_class:
        raise RuntimeError("No encontré class SpatialMemorySweep.")
    if suspicious:
        detail = ", ".join(f"L{ln}:{why}" for ln, why in suspicious)
        raise RuntimeError("SpatialMemorySweep todavía conserva restricciones estructurales P=5: " + detail)

    ast.parse(text, filename=str(path))
    path.write_text(text, encoding="utf-8")
    print("[Dynamic-SMS] Auditoría OK: sin restricciones estructurales P=5.")


_prepare_dynamic_sms()
importlib.invalidate_caches()
for _stale in (
    "nestsar_sms_s1c_v2",
    "nestsar_m4_geom_h4",
    "nestsar_m4_regmask_ema_v3_safe",
):
    sys.modules.pop(_stale, None)

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

import nestsar as ns
import nestsar_sms_s1c_v2 as sms
import nestsar_m4_geom_h4 as m4
import nestsar_m4_regmask_ema_v3_safe as reg

MODEL_ID = "nestsar_m4_geom_h4_jointfirst_wide128"
MODEL_MODE = "NestSAR_M4_Geometry_H4_JointFirst_Wide128"
SPATIAL_OUT_DIM = 128

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

JOINT_SWEEP_ORDER = (
    0, 1, 20, 2, 3,
    4, 5, 6, 7, 21, 22,
    8, 9, 10, 11, 23, 24,
    12, 13, 14, 15,
    16, 17, 18, 19,
)
if sorted(JOINT_SWEEP_ORDER) != list(range(25)):
    raise ValueError("JOINT_SWEEP_ORDER debe ser una permutación de 0..24.")

JOINT_SWEEP_ORDER_JAX = jnp.asarray(JOINT_SWEEP_ORDER, dtype=jnp.int32)
JOINT_SWEEP_INVERSE_JAX = jnp.asarray(np.argsort(np.asarray(JOINT_SWEEP_ORDER)), dtype=jnp.int32)


def _make_part_mask() -> jnp.ndarray:
    mask = np.zeros((10, 25), dtype=np.float32)
    seen = []
    for p, (_, joints) in enumerate(TEN_BODY_PARTS):
        for j in joints:
            mask[p, j] = 1.0
            seen.append(j)
    if sorted(seen) != list(range(25)):
        raise ValueError("TEN_BODY_PARTS debe cubrir 0..24 exactamente una vez.")
    return jnp.asarray(mask)


TEN_PART_MASK = _make_part_mask()
sms.BODY_PARTS = TEN_BODY_PARTS
sms.PART_MASK = TEN_PART_MASK
m4.sms.BODY_PARTS = TEN_BODY_PARTS
m4.sms.PART_MASK = TEN_PART_MASK
reg.EMA_DECAY = 0.995


class JointFirstWideSpatialEncoder(nn.Module):
    stream_name: str
    persons: int = 2
    joints: int = 25
    coords: int = 3
    token_dim: int = sms.PART_TOKEN_DIM
    spatial_memory_dim: int = sms.SPATIAL_MEMORY_DIM
    spatial_dim: int = 32
    dropout: float = 0.15
    use_part_attention: bool = False

    @nn.compact
    def __call__(self, stream_xyz: jnp.ndarray, joint_xyz: jnp.ndarray, geometry: jnp.ndarray, training: bool) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        if stream_xyz.ndim != 5:
            raise ValueError(f"stream_xyz inválido: {stream_xyz.shape}")
        if self.joints != 25:
            raise ValueError(f"Esta variante requiere 25 joints; recibido {self.joints}.")
        if self.use_part_attention:
            raise ValueError("JointFirst-Wide128 no usa part attention.")

        person_present = jnp.any(jnp.abs(joint_xyz) > 1e-6, axis=(3, 4))
        present = person_present[..., None, None].astype(stream_xyz.dtype)
        if self.stream_name == "joint":
            root = stream_xyz[:, :, :, 0:1, :]
            base = (stream_xyz - root) * present
        else:
            base = stream_xyz * present

        pair_valid = (person_present[:, 1:] & person_present[:, :-1])[..., None, None].astype(stream_xyz.dtype)
        velocity = jnp.concatenate([jnp.zeros_like(base[:, :1]), (base[:, 1:] - base[:, :-1]) * pair_valid], axis=1)
        joint_features = jnp.concatenate([base, velocity, geometry], axis=-1)

        h = nn.Dense(self.token_dim, kernel_init=nn.initializers.xavier_uniform(), name="joint_feature_projection")(joint_features)
        joint_embedding = self.param("joint_embedding", nn.initializers.normal(stddev=0.02), (1, 1, 1, self.joints, self.token_dim))
        person_embedding = self.param("person_embedding", nn.initializers.normal(stddev=0.02), (1, 1, self.persons, 1, self.token_dim))
        h = nn.LayerNorm(name="joint_embedding_norm")(h + joint_embedding + person_embedding)
        h = nn.gelu(h)
        h = nn.Dropout(rate=self.dropout, name="joint_feature_dropout")(h, deterministic=not training)
        h = h * present

        h_sweep = jnp.take(h, JOINT_SWEEP_ORDER_JAX, axis=3)
        h_sweep, sms_metrics = sms.SpatialMemorySweep(
            token_dim=self.token_dim,
            memory_dim=self.spatial_memory_dim,
            dropout=self.dropout,
            name="joint_spatial_memory_sweep",
        )(h_sweep, person_present, training)
        h = jnp.take(h_sweep, JOINT_SWEEP_INVERSE_JAX, axis=3) * present

        joint_gate = jax.nn.sigmoid(nn.Dense(1, kernel_init=nn.initializers.xavier_uniform(), name="joint_pool_gate")(h)[..., 0])
        joint_gate = joint_gate * person_present[..., None].astype(h.dtype)

        part_mask = TEN_PART_MASK.astype(h.dtype)
        numerator = jnp.einsum("btmjd,pj,btmj->btmpd", h, part_mask, joint_gate)
        denominator = jnp.einsum("pj,btmj->btmp", part_mask, joint_gate)[..., None]
        part_tokens = numerator / jnp.maximum(denominator, 1e-6)

        part_embedding = self.param("part_embedding", nn.initializers.normal(stddev=0.02), (1, 1, 1, 10, self.token_dim))
        part_tokens = nn.LayerNorm(name="part_token_norm")(part_tokens + part_embedding)
        part_tokens = part_tokens * present

        part_gate = jax.nn.sigmoid(nn.Dense(1, kernel_init=nn.initializers.xavier_uniform(), name="part_pool_gate")(part_tokens)[..., 0])
        part_gate = part_gate * person_present[..., None].astype(h.dtype)
        person_descriptor = jnp.sum(part_gate[..., None] * part_tokens, axis=3) / jnp.maximum(jnp.sum(part_gate, axis=3, keepdims=True), 1e-6)

        first = person_descriptor[:, :, 0]
        second = person_descriptor[:, :, 1]
        pair_features = jnp.concatenate([first + second, jnp.abs(first - second), first * second], axis=-1)

        spatial_summary = nn.Dense(SPATIAL_OUT_DIM, kernel_init=nn.initializers.xavier_uniform(), name="person_fusion_wide")(pair_features)
        spatial_summary = nn.LayerNorm(name="spatial_summary_norm")(spatial_summary)
        spatial_summary = nn.gelu(spatial_summary)
        spatial_summary = nn.Dropout(rate=self.dropout, name="spatial_summary_dropout")(spatial_summary, deterministic=not training)
        any_person = jnp.any(person_present, axis=2)[..., None].astype(h.dtype)
        spatial_summary = spatial_summary * any_person

        metrics = dict(sms_metrics)
        metrics.update({
            "joint_gate_mean": jnp.sum(joint_gate) / jnp.maximum(jnp.sum(person_present) * self.joints, 1.0),
            "part_gate_mean": jnp.sum(part_gate) / jnp.maximum(jnp.sum(person_present) * 10, 1.0),
            "geometry_distance_mean": jnp.mean(geometry[..., 0]),
            "geometry_cosine_mean": jnp.mean(geometry[..., 1]),
        })
        return spatial_summary, metrics


m4.GeometricSpatialPartEncoder = JointFirstWideSpatialEncoder
ns.MODEL_ALIASES[MODEL_ID] = MODEL_MODE


def build_model(model_id: str):
    if model_id != MODEL_ID:
        return m4.build_model(model_id)
    return m4.NestSARM4GeomH4(
        mode=m4.MODEL_MODE,
        num_classes=ns.CFG.num_classes,
        model_dim=ns.CFG.model_dim,
        memory_dim=ns.CFG.memory_dim,
        dropout=ns.CFG.dropout,
        memory_residual_scale=ns.CFG.memory_residual_scale,
        initial_eta=ns.CFG.initial_eta,
        initial_alpha=ns.CFG.initial_alpha,
        frame_blocks=ns.CFG.frame_blocks,
        chunk_blocks=ns.CFG.chunk_blocks,
        clip_blocks=ns.CFG.clip_blocks,
        controller_blocks=ns.CFG.controller_blocks,
        chunk_size=ns.CFG.chunk_size,
        clip_size=ns.CFG.clip_size,
        controller_rank=ns.CFG.controller_rank,
        use_part_attention=False,
    )


def build_steps(model, model_id: str):
    if model_id == MODEL_ID:
        return reg.build_steps(model, m4.MODEL_ID)
    return reg.build_steps(model, model_id)


ns.build_model = build_model
ns.build_steps = build_steps
ns.__file__ = __file__

print("=" * 108)
print("M4G-H4 JOINTFIRST-WIDE128 v4 + REGMASK + EMA")
print("=" * 108)
print("Spatial:           25 joints -> kinematic order -> SMS(25) -> 10 parts")
print("Spatial summary:   128-D (baseline M4 branch = 32-D)")
print("Part attention:    OFF")
print("Second part SMS:   OFF")
print("Temporal M4G-H4:   INTACTO")
print("Spatial gate M4:   original (max 0.25, init ~0.10)")
print(f"EMA decay:         {reg.EMA_DECAY}")
print(f"Frame hold-mask:   {reg.FRAME_MASK_PROB:.0%}")
print(f"Joint hold-mask:   {reg.JOINT_MASK_PROB:.0%}")
print(f"Part hold-mask:    {reg.PART_MASK_PROB:.0%}")
print("=" * 108)

if __name__ == "__main__":
    raise SystemExit(ns.main())
