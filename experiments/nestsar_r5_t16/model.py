"""NestSAR-R5-T16: one trunk, fine hands, bidirectional joints, real interaction.

Each change answers one finding of the R4 audit:

1. Hands kept at full resolution. Thumb, hand tip and wrist+hand are separate
   parts (14 parts instead of 10), and a 4x-rate hand branch reads
   hand-relative vectors (tip-hand, thumb-hand, hand-wrist) and wrist steps at
   64 sub-segments, so OK vs victory sign is not averaged away.
2. Small repeated motions. The hand branch runs a bidirectional GRU over the 64
   sub-segments before pooling to 16, so rhythm is learned, not summarised; the
   body tokens add the reversal amount (path - |net displacement|) per axis.
3. No decorative caps. The shared controller (+-10% FiLM, stream gates, fusion
   logits bounded to +-0.15, rank-2 head x0.15) is removed. The fast memory is
   kept, with eta in (0,1) and alpha in (0.5,1) driven by its own prediction
   error ("surprise", as in Titans/HOPE), and a learnable layer scale instead of
   a fixed 0.08 residual.
4. Bidirectional joint sweep (the R4 chain was one-way: the left arm never saw
   the right arm).
5. One early-fused trunk. Joint, bone, joint-motion and bone-motion channels are
   embedded together per joint, so one model sees every view, instead of four
   weak per-view models whose logits were averaged.
6. Person-person interaction: explicit hand/head/torso distances, relative
   position/velocity and facing between the two actors, plus a layer-scaled
   cross-person message, all masked when the pair is absent.

Input [B, 16, 942] = R4 tokens [16, 750] + hand block [16, 192]
(see ``preprocessing.py``). No attention, no graph convolution, no TCN.
"""
from __future__ import annotations

from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from experiments.nestsar_r5_t16.preprocessing import (
    FEATURES,
    FRAMES,
    HAND_CHANNELS,
    JOINTS,
    PERSONS,
    R4_FEATURES,
    SUB,
    TOKEN_CHANNELS,
)

NUM_CLASSES = 120

# NTU-25 skeleton (zero-based). Identical to the R4 constants (checked in tests).
PARENTS = np.asarray([0, 0, 20, 2, 20, 4, 5, 6, 20, 8, 9, 10, 0,
                      12, 13, 14, 0, 16, 17, 18, 1, 7, 7, 11, 11], np.int32)
JOINT_ORDER = np.asarray([0, 1, 20, 2, 3, 4, 5, 6, 7, 21, 22, 8, 9, 10, 11, 23, 24,
                          12, 13, 14, 15, 16, 17, 18, 19], np.int32)
TEN_PARTS = ((0, 1, 20), (2, 3), (4, 5), (6, 7, 21, 22), (8, 9), (10, 11, 23, 24),
             (12, 13), (14, 15), (16, 17), (18, 19))
# Thumb (22/24) and hand tip (21/23) are their own parts.
FOURTEEN_PARTS = ((0, 1, 20), (2, 3), (4, 5), (6, 7), (21,), (22,), (8, 9), (10, 11),
                  (23,), (24,), (12, 13), (14, 15), (16, 17), (18, 19))

SPINE_BASE, SPINE_MID, HEAD, SPINE_SHOULDER = 0, 1, 3, 20
L_SHOULDER, R_SHOULDER, L_HAND, R_HAND = 4, 8, 7, 11

SHAPE_FEATURES = 14          # hand-shape scalars per person (7 per hand)
SELF_FEATURES = 7            # hand-head/torso/hip and hand-hand distances per person
PAIR_FEATURES = 22           # person-person relation per direction
EPS = 1e-6


def part_matrix(parts) -> np.ndarray:
    m = np.zeros((len(parts), JOINTS), np.float32)
    for p, joints in enumerate(parts):
        m[p, list(joints)] = 1.0
    if not np.all(m.sum(0) == 1):
        raise ValueError("Parts must cover every joint exactly once")
    return m


def safe_norm(x: jnp.ndarray, axis: int = -1, keepdims: bool = False) -> jnp.ndarray:
    """Euclidean norm with a finite gradient at zero (clamp before sqrt)."""
    return jnp.sqrt(jnp.maximum(jnp.sum(jnp.square(x), axis=axis, keepdims=keepdims), EPS * EPS))


def safe_unit(x: jnp.ndarray) -> jnp.ndarray:
    return x / safe_norm(x, keepdims=True)


def cosine(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Cosine of the angle between a and b; exactly 0 when either is the zero vector."""
    return jnp.sum(a * b, axis=-1) / (safe_norm(a) * safe_norm(b))


def distance(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return safe_norm(a - b)


def masked_mean(x: jnp.ndarray, mask: jnp.ndarray, axis: int) -> jnp.ndarray:
    """Mean of x over `axis` using a 0/1 mask broadcastable to x without its last dim."""
    w = mask.astype(x.dtype)[..., None]
    return jnp.sum(x * w, axis=axis) / jnp.maximum(jnp.sum(w, axis=axis), 1.0)


class GRUSweep(nn.Module):
    """GRU over axis 1 of [N, L, D_in]; input projections hoisted out of the scan."""

    hidden: int
    reverse: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = self.hidden
        xin = nn.Dense(3 * h, name="input")(x)                       # z | r | candidate
        init = nn.initializers.xavier_uniform()
        w_zr = self.param("hidden_zr", init, (h, 2 * h))
        w_c = self.param("hidden_c", init, (h, h))
        xs = jnp.swapaxes(xin, 0, 1)
        if self.reverse:
            xs = xs[::-1]

        def step(state, xt):
            zr = jax.nn.sigmoid(xt[:, :2 * h] + state @ w_zr)
            z, r = zr[:, :h], zr[:, h:]
            cand = jnp.tanh(xt[:, 2 * h:] + (r * state) @ w_c)
            state = (1.0 - z) * state + z * cand
            return state, state

        _, ys = jax.lax.scan(step, jnp.zeros((x.shape[0], h), x.dtype), xs)
        if self.reverse:
            ys = ys[::-1]
        return jnp.swapaxes(ys, 0, 1)


class BiGRUSweep(nn.Module):
    """Forward and backward GRUs advanced in ONE scan (stacked weights).

    Mathematically identical to [GRUSweep(fwd), GRUSweep(reverse)] with the same
    weights (tested), but the sequential chain is L steps instead of 2L, which
    is what bounds latency on a GPU for these small recurrent matmuls.
    """

    hidden: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = self.hidden
        x_fwd = nn.Dense(3 * h, name="input_fwd")(x)
        x_bwd = nn.Dense(3 * h, name="input_bwd")(x)[:, ::-1]
        init = nn.initializers.xavier_uniform()
        # Stack two independently initialised GRUs (same init as GRUSweep for each direction).
        w_zr = self.param("hidden_zr", lambda k, s: jnp.stack([init(kk, s[1:]) for kk in jax.random.split(k)]),
                          (2, h, 2 * h))
        w_c = self.param("hidden_c", lambda k, s: jnp.stack([init(kk, s[1:]) for kk in jax.random.split(k)]),
                         (2, h, h))
        xs = jnp.stack([jnp.swapaxes(x_fwd, 0, 1), jnp.swapaxes(x_bwd, 0, 1)], axis=1)   # [L, 2, N, 3h]

        def step(state, xt):                                                     # state [2, N, h]
            zr = jax.nn.sigmoid(xt[..., :2 * h] + jnp.einsum("dnh,dhk->dnk", state, w_zr))
            z, r = zr[..., :h], zr[..., h:]
            cand = jnp.tanh(xt[..., 2 * h:] + jnp.einsum("dnh,dhk->dnk", r * state, w_c))
            state = (1.0 - z) * state + z * cand
            return state, state

        _, ys = jax.lax.scan(step, jnp.zeros((2, x.shape[0], h), x.dtype), xs)
        fwd = jnp.swapaxes(ys[:, 0], 0, 1)
        bwd = jnp.swapaxes(ys[::-1, 1], 0, 1)
        return jnp.concatenate([fwd, bwd], axis=-1)


class Sweep(nn.Module):
    """Residual (bi)directional GRU block: LN(x + Dense([fwd, bwd]))."""

    hidden: int
    bidirectional: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.bidirectional:
            states = BiGRUSweep(self.hidden, name="bigru")(x)
        else:
            states = GRUSweep(self.hidden, name="fwd")(x)
        y = nn.Dense(x.shape[-1], name="merge")(states)
        return nn.LayerNorm(name="norm")(x + y)


class FastMemory(nn.Module):
    """Low-rank delta-rule fast weights, re-initialised for every clip.

    pred_t = k_t^T S,  err_t = v_t - pred_t,  s_t = rms(err_t)       (surprise)
    eta_t   = sigmoid(g_eta(v_t) + u_eta s_t + b_eta)            in (0, 1)
    alpha_t = 0.5 + 0.5 sigmoid(g_alpha(v_t) + u_alpha s_t + b_alpha)  in (0.5, 1)
    S <- alpha_t S + eta_t k_t err_t^T,   read_t = q_t^T S

    With unit-norm keys, eta <= 1 and alpha <= 1 the update is non-expansive, so
    no hard cap is needed for stability. Init: eta = 0.1, alpha = 0.97.
    ``capped=True`` reproduces the R4 bounds (eta <= 0.2, alpha in [0.9, 0.999],
    no surprise term) for the ablation.
    """

    dim: int
    rank: int = 8
    capped: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        v = nn.LayerNorm(name="value_norm")(x)
        k = safe_unit(jnp.tanh(nn.Dense(self.rank, use_bias=False,
                                        kernel_init=nn.initializers.normal(0.02), name="key")(v)))
        q = safe_unit(jnp.tanh(nn.Dense(self.rank, use_bias=False,
                                        kernel_init=nn.initializers.normal(0.02), name="query")(v)))
        gates = nn.Dense(2, kernel_init=nn.initializers.zeros, use_bias=False, name="gate")(v)
        memory0 = self.param("memory0", nn.initializers.normal(0.01), (self.rank, self.dim))
        if not self.capped:
            surprise_w = self.param("surprise", nn.initializers.zeros, (2,))
            bias = self.param("gate_bias",
                              lambda *_: jnp.asarray([np.log(0.1 / 0.9), np.log(0.94 / 0.06)],
                                                     jnp.float32), (2,))

        def step(mem, inputs):
            key_t, query_t, value_t, gate_t = inputs
            pred = jnp.einsum("br,brd->bd", key_t, mem)
            err = value_t - pred
            if self.capped:
                eta = 0.2 * jax.nn.sigmoid(gate_t[:, 0])
                alpha = 0.9 + 0.099 * jax.nn.sigmoid(gate_t[:, 1])
            else:
                surprise = jnp.sqrt(jnp.maximum(jnp.mean(jnp.square(err), axis=-1), EPS * EPS))
                logits = gate_t + surprise[:, None] * surprise_w + bias
                eta = jax.nn.sigmoid(logits[:, 0])
                alpha = 0.5 + 0.5 * jax.nn.sigmoid(logits[:, 1])
            mem = alpha[:, None, None] * mem + eta[:, None, None] * jnp.einsum("br,bd->brd", key_t, err)
            read = jnp.einsum("br,brd->bd", query_t, mem)
            return mem, (read, eta, alpha)

        mem0 = jnp.broadcast_to(memory0[None], (x.shape[0], self.rank, self.dim))
        seq = tuple(jnp.swapaxes(a, 0, 1) for a in (k, q, v, gates))
        _, (reads, eta, alpha) = jax.lax.scan(step, mem0, seq)
        return jnp.swapaxes(reads, 0, 1), jnp.swapaxes(eta, 0, 1), jnp.swapaxes(alpha, 0, 1)


class NestedMemory(nn.Module):
    """BiMemory (GRU fwd/bwd) + self-modifying fast memory with a learnable layer scale."""

    dim: int
    rank: int = 8
    fast_mode: str = "surprise"          # "surprise" | "capped" | "off"

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        base = Sweep(self.dim, bidirectional=True, name="base")(x)
        zeros = jnp.zeros(x.shape[:2], x.dtype)
        if self.fast_mode == "off":
            return base, zeros, zeros, jnp.zeros((), x.dtype)
        reads, eta, alpha = FastMemory(self.dim, self.rank, capped=self.fast_mode == "capped",
                                       name="fast")(base)
        if self.fast_mode == "capped":
            scale = jnp.full((self.dim,), 0.08, x.dtype)             # R4's fixed residual scale
        else:
            scale = self.param("fast_scale", nn.initializers.constant(0.1), (self.dim,))
        out = nn.LayerNorm(name="norm")(base + scale * reads)
        return out, eta, alpha, jnp.mean(jnp.abs(scale))


def hand_shape(hands: jnp.ndarray) -> jnp.ndarray:
    """[..., 24] hand block -> [..., 14] scale-free hand-shape scalars."""
    feats = []
    for side in range(2):
        c = hands[..., 12 * side: 12 * side + 12]
        tip, thumb, palm, wrist_step = c[..., 0:3], c[..., 3:6], c[..., 6:9], c[..., 9:12]
        feats += [
            safe_norm(tip), safe_norm(thumb), safe_norm(palm), distance(tip, thumb),
            cosine(tip, palm), cosine(thumb, tip), safe_norm(wrist_step),
        ]
    return jnp.stack(feats, axis=-1)


def self_relations(pose: jnp.ndarray, valid: jnp.ndarray) -> jnp.ndarray:
    """[..., 25, 3] pose -> [..., 7] hand-head/shoulder/hip and hand-hand distances."""
    pairs = ((L_HAND, HEAD), (R_HAND, HEAD), (L_HAND, SPINE_SHOULDER), (R_HAND, SPINE_SHOULDER),
             (L_HAND, SPINE_BASE), (R_HAND, SPINE_BASE), (L_HAND, R_HAND))
    out = []
    for a, b in pairs:
        ok = (valid[..., a] & valid[..., b]).astype(pose.dtype)
        out.append(distance(pose[..., a, :], pose[..., b, :]) * ok)
    return jnp.stack(out, axis=-1)


def facing(pose: jnp.ndarray) -> jnp.ndarray:
    """Horizontal body normal from the shoulder line (y is up in NTU coordinates)."""
    across = pose[..., L_SHOULDER, :] - pose[..., R_SHOULDER, :]
    up = jnp.asarray([0.0, 1.0, 0.0], pose.dtype)
    return jnp.cross(across, jnp.broadcast_to(up, across.shape))


def pair_relations(pose_a, pose_b, disp_a, disp_b, pair) -> jnp.ndarray:
    """Relation of actor B as seen by actor A, [..., 22]; zero when the pair is absent."""
    feats = []
    for ha in (L_HAND, R_HAND):
        for target in (HEAD, SPINE_MID):
            feats.append(distance(pose_a[..., ha, :], pose_b[..., target, :]))
    for hb in (L_HAND, R_HAND):
        for target in (HEAD, SPINE_MID):
            feats.append(distance(pose_b[..., hb, :], pose_a[..., target, :]))
    for ha in (L_HAND, R_HAND):
        for hb in (L_HAND, R_HAND):
            feats.append(distance(pose_a[..., ha, :], pose_b[..., hb, :]))
    offset = pose_b[..., SPINE_BASE, :] - pose_a[..., SPINE_BASE, :]
    feats.append(safe_norm(offset))
    rel_vel = disp_b[..., SPINE_BASE, :] - disp_a[..., SPINE_BASE, :]
    scalars = jnp.stack(feats, axis=-1)                                   # [..., 13]
    direction = safe_unit(offset)
    closing = jnp.sum(rel_vel * direction, axis=-1, keepdims=True)
    face_a = cosine(facing(pose_a), offset)[..., None]
    face_b = cosine(facing(pose_b), -offset)[..., None]
    out = jnp.concatenate([scalars, offset, rel_vel, face_a, face_b, closing], axis=-1)
    return out * pair[..., None]


class NestSARR5T16(nn.Module):
    spatial_dim: int = 48
    spatial_hidden: int = 32
    hand_dim: int = 32
    hand_hidden: int = 24
    person_dim: int = 96
    model_dim: int = 176
    fast_rank: int = 8
    dropout: float = 0.10
    # Ablation switches (all True / "surprise" = full R5).
    hand_branch: bool = True
    fine_parts: bool = True
    bidirectional_sweep: bool = True
    interaction: bool = True
    fast_mode: str = "surprise"

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> Mapping[str, jnp.ndarray]:
        if x.ndim != 3 or x.shape[1] != FRAMES or x.shape[2] != FEATURES:
            raise ValueError(f"Expected [B,{FRAMES},{FEATURES}], got {x.shape}")
        if self.fast_mode not in ("surprise", "capped", "off"):
            raise ValueError(f"Unknown fast_mode {self.fast_mode!r}")
        b = x.shape[0]
        dtype = x.dtype
        tok = x[..., :R4_FEATURES].reshape(b, FRAMES, PERSONS, JOINTS, TOKEN_CHANNELS)
        hands = x[..., R4_FEATURES:].reshape(b, FRAMES, SUB, PERSONS, HAND_CHANNELS)

        # ---- validity (a root-centred P1 root can be exactly zero) ----
        joint_valid = jnp.any(jnp.abs(tok) > 1e-8, axis=-1)                 # [B,T,P,V]
        present = jnp.any(joint_valid, axis=-1)                             # [B,T,P]
        joint_valid = joint_valid.at[..., 0].set(present)
        present_f = present.astype(dtype)
        pair = present_f[..., 0] * present_f[..., 1]                        # [B,T]

        pose, disp = tok[..., 0:3], tok[..., 3:6]
        phase_a, phase_b, path = tok[..., 6:9], tok[..., 9:12], tok[..., 12:15]
        parents = jnp.asarray(PARENTS)
        bone_valid = joint_valid & jnp.take(joint_valid, parents, axis=3)
        bv = bone_valid[..., None].astype(dtype)
        par = lambda a: jnp.take(a, parents, axis=3)
        bone = (pose - par(pose)) * bv
        joint_motion = jnp.concatenate([disp, phase_a, phase_b, path], axis=-1)
        bone_motion = jnp.concatenate(
            [disp - par(disp), phase_a - par(phase_a), phase_b - par(phase_b),
             jnp.abs(path - par(path))], axis=-1) * bv
        reversal = jax.nn.relu(path - jnp.abs(disp))                        # back-and-forth per axis

        # ---- early-fused joint embedding (all four R4 views in one vector) ----
        jv = joint_valid[..., None].astype(dtype)
        d_pose = self.spatial_dim // 3
        e = jnp.concatenate([
            nn.Dense(d_pose, name="embed_pose")(jnp.concatenate([pose, bone], -1)),
            nn.Dense(self.spatial_dim - d_pose, name="embed_motion")(
                jnp.concatenate([joint_motion, bone_motion, reversal], -1)),
        ], axis=-1)
        joint_embed = self.param("joint_embed", nn.initializers.normal(0.02), (JOINTS, self.spatial_dim))
        person_embed = self.param("person_embed", nn.initializers.normal(0.02), (PERSONS, self.spatial_dim))
        e = nn.gelu(e + joint_embed + person_embed[:, None, :]) * jv

        # ---- (bi)directional sweep over the joints of each person and segment ----
        order = jnp.asarray(JOINT_ORDER)
        inverse = jnp.asarray(np.argsort(JOINT_ORDER))
        seq = jnp.take(e, order, axis=3).reshape(b * FRAMES * PERSONS, JOINTS, self.spatial_dim)
        seq = Sweep(self.spatial_hidden, bidirectional=self.bidirectional_sweep, name="joint_sweep")(seq)
        s = jnp.take(seq.reshape(b, FRAMES, PERSONS, JOINTS, self.spatial_dim), inverse, axis=3) * jv

        # ---- parts: thumb and hand tip are not averaged with the wrist ----
        parts = jnp.asarray(part_matrix(FOURTEEN_PARTS if self.fine_parts else TEN_PARTS))
        weight = jnp.einsum("btpv,qv->btpq", joint_valid.astype(dtype), parts)
        part_feat = jnp.einsum("btpvd,qv->btpqd", s, parts) / jnp.maximum(weight, 1.0)[..., None]
        person_in = [part_feat.reshape(b, FRAMES, PERSONS, -1),
                     self_relations(pose, joint_valid)]

        # ---- 4x-rate hand branch ----
        stream_logits = []
        hand_valid_rate = jnp.zeros((b,), dtype)
        if self.hand_branch:
            hv = jnp.any(jnp.abs(hands) > 1e-8, axis=-1) & present[:, :, None, :]   # [B,T,S,P]
            hv_f = hv.astype(dtype)
            shape = hand_shape(hands) * hv_f[..., None]                              # [B,T,S,P,14]
            z = nn.gelu(nn.Dense(self.hand_dim, name="hand_in")(
                jnp.concatenate([hands, shape], axis=-1))) * hv_f[..., None]
            z = jnp.transpose(z.reshape(b, FRAMES * SUB, PERSONS, self.hand_dim), (0, 2, 1, 3))
            z = Sweep(self.hand_hidden, bidirectional=True, name="hand_sweep")(
                z.reshape(b * PERSONS, FRAMES * SUB, self.hand_dim))
            z = jnp.transpose(z.reshape(b, PERSONS, FRAMES, SUB, self.hand_dim), (0, 2, 3, 1, 4))
            z = z * hv_f[..., None]                                                  # [B,T,S,P,dh]
            hand_seg = masked_mean(z, hv, axis=2)                                    # [B,T,P,dh]
            shape_seg = masked_mean(shape, hv, axis=2)                               # [B,T,P,14]
            person_in += [hand_seg, shape_seg]
            clip_hand = masked_mean(z.reshape(b, FRAMES * SUB, PERSONS, self.hand_dim),
                                    hv.reshape(b, FRAMES * SUB, PERSONS), axis=1)    # [B,P,dh]
            stream_logits.append(nn.Dense(NUM_CLASSES, name="hand_aux_head")(clip_hand.reshape(b, -1)))
            hand_valid_rate = jnp.mean(hv_f, axis=(1, 2, 3))

        # ---- person vectors ----
        u = nn.Dense(self.person_dim, name="person_proj")(jnp.concatenate(person_in, axis=-1))
        u = nn.LayerNorm(name="person_norm")(nn.gelu(u)) * present_f[..., None]       # [B,T,P,Dp]

        # ---- person-person interaction ----
        frame_in = [u.reshape(b, FRAMES, PERSONS * self.person_dim),
                    jnp.stack([present_f[..., 0], present_f[..., 1], pair], axis=-1)]
        interaction_scale = jnp.zeros((), dtype)
        if self.interaction:
            p_a, p_b = pose[:, :, 0], pose[:, :, 1]
            d_a, d_b = disp[:, :, 0], disp[:, :, 1]
            rel_ab = pair_relations(p_a, p_b, d_a, d_b, pair)          # what P1 sees of P2
            rel_ba = pair_relations(p_b, p_a, d_b, d_a, pair)          # what P2 sees of P1
            message = nn.Dense(self.person_dim, name="pair_message")
            m_a = nn.gelu(message(jnp.concatenate([u[:, :, 1], rel_ab], -1)))
            m_b = nn.gelu(message(jnp.concatenate([u[:, :, 0], rel_ba], -1)))
            lam = self.param("pair_scale", nn.initializers.constant(0.1), (self.person_dim,))
            pair_norm = nn.LayerNorm(name="pair_norm")
            u_a = pair_norm(u[:, :, 0] + lam * m_a * pair[..., None]) * present_f[..., 0:1]
            u_b = pair_norm(u[:, :, 1] + lam * m_b * pair[..., None]) * present_f[..., 1:2]
            frame_in = [u_a, u_b, rel_ab, frame_in[1]]
            interaction_scale = jnp.mean(jnp.abs(lam))

        f = nn.Dense(self.model_dim, name="frame_proj")(jnp.concatenate(frame_in, axis=-1))
        f = nn.LayerNorm(name="frame_norm")(nn.gelu(f))
        f = nn.Dropout(self.dropout)(f, deterministic=not training)                  # [B,T,D]

        # ---- nested temporal memory: M4 over 16 segments, G4 over 4 chunks ----
        m4, eta, alpha, fast_scale = NestedMemory(self.model_dim, self.fast_rank, self.fast_mode,
                                                  name="m4")(f)
        chunks = m4.reshape(b, 4, FRAMES // 4, self.model_dim).mean(axis=2)
        g4, eta_g, alpha_g, fast_scale_g = NestedMemory(self.model_dim, self.fast_rank, self.fast_mode,
                                                        name="g4")(chunks)
        m4_mean = m4.mean(axis=1)
        desc = nn.Dense(self.model_dim, name="descriptor")(jnp.concatenate([m4_mean, g4.mean(axis=1)], -1))
        desc = nn.LayerNorm(name="descriptor_norm")(nn.gelu(desc))
        desc = nn.Dropout(self.dropout)(desc, deterministic=not training)
        logits = nn.Dense(NUM_CLASSES, name="classifier")(desc)
        stream_logits.insert(0, nn.Dense(NUM_CLASSES, name="m4_aux_head")(m4_mean))

        ones = jnp.ones((b,), dtype)
        return {
            "logits": logits,
            "main_logits": logits,
            "stream_logits": jnp.stack(stream_logits, axis=1),     # [B, S_aux, C] training-only heads
            "descriptor": desc,
            "sm_eta_mean": jnp.mean(eta, axis=1),
            "sm_alpha_mean": jnp.mean(alpha, axis=1),
            "g4_eta_mean": jnp.mean(eta_g, axis=1),
            "g4_alpha_mean": jnp.mean(alpha_g, axis=1),
            "fast_scale_m4": fast_scale * ones,
            "fast_scale_g4": fast_scale_g * ones,
            "pair_scale": interaction_scale * ones,
            "pair_rate": jnp.mean(pair, axis=1),
            "hand_valid_rate": hand_valid_rate,
        }
