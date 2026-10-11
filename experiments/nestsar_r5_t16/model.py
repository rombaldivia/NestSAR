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


# --------------------------------------------------------------------------- R6-HOPE temporal core
# HOPE (Nested Learning, Behrouz et al., NeurIPS 2025) = self-modifying Titans + a continuum memory
# system (CMS). Here: four levels with in-clip periods 1/2/4/8 (16, 8, 4, 2 steps); every level is a
# HOPE block (local mixer -> self-referential memory -> CMS MLP), and in training the parameters of the
# level with period p are updated every p optimizer steps (worker.make_optimizer).
CMS_PERIODS = (1, 2, 4, 8)
LEVEL_NAMES = ("m4", "l2", "g4", "l8")          # m4/g4 keep the R5 names (and their BiGRU local mixer)
SELFREF_BETA = 0.10          # bounded residual read  M(x) = x + beta * tanh(A x)
SELFREF_COMPONENT_RATE = 0.10  # component memories learn at 0.1 x the main-memory eta
SELFREF_MATRIX_CAP = 4.0     # Frobenius cap of every component matrix (stability, not part of HOPE)
SELFREF_VECTOR_CAP = 1.0
SELFREF_MAIN_CAP = 8.0
# LayerNorm epsilon of the HOPE blocks. Their inputs are already normalised (variance ~1), so 1e-2 is
# negligible on data; on an all-zero (padding / empty) clip every activation is exactly 0 and the
# default 1e-6 makes each of the ~12 chained LayerNorms amplify the backward pass by 1e3 (-> inf).
HOPE_LN_EPS = 1e-2


def _cap(x: jnp.ndarray, cap: float, axes) -> jnp.ndarray:
    norm = jnp.sqrt(jnp.sum(jnp.square(x), axis=axes, keepdims=True) + EPS * EPS)
    return x * jnp.minimum(1.0, cap / norm)


class SelfRefMemory(nn.Module):
    """Self-referential associative memory (HOPE / self-modifying Titans), small inner width d.

    u_t = W_in LN(x_t) / sqrt(d). Every component is itself a memory, re-initialised per clip from a
    learned (meta-learned) initial state and written in-context:
        k_t = unit(u_t + beta tanh(A_k u_t))     v_t = u_t + beta tanh(A_v u_t)
        q_t = unit(u_t + beta tanh(A_q u_t))     eta/alpha logits = a_eta . u_t, a_alpha . u_t
    Main memory (Titans, L2 / delta rule, surprise s_t = rms(v_t - M k_t) also drives eta and alpha):
        M <- M (alpha I - eta k k^T) + eta v k^T,       y_t = M q_t
    Component memories (HOPE self-reference: each one generates its own value  v_hat = A v_t):
        A <- A (alpha I - eta_c k k^T) + eta_c (A v_t) k^T,   eta_c = 0.1 eta   (same for a_eta, a_alpha)
    eta in (0, 1), alpha in (0.5, 1); unit keys keep (alpha I - eta k k^T) non-expansive. Frobenius caps
    on A / a / M are a stability addition (not in HOPE). Output: W_out y_t, [B, L, dim].
    """

    dim: int
    inner: int = 32
    deep: bool = False          # Titans deep memory: 2-layer MLP written by gradient + momentum
    mem_hidden: int = 64

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        d, h = self.inner, self.mem_hidden
        n_gates = 3 if self.deep else 2                                  # eta, alpha (+ momentum beta)
        u = nn.Dense(d, use_bias=False, name="in_proj")(nn.LayerNorm(epsilon=HOPE_LN_EPS, name="in_norm")(x)) / np.sqrt(d)
        comp0 = self.param("components0", lambda k, s: 0.12 * jax.random.normal(k, s) / np.sqrt(d),
                           (3, d, d))                                   # A_k, A_v, A_q
        gate0 = self.param("gates0", nn.initializers.zeros, (n_gates, d))  # a_eta, a_alpha (, a_beta)
        if self.deep:
            w1_0 = self.param("memory_w1", lambda k, s: jax.random.normal(k, s) / np.sqrt(d), (h, d))
            w2_0 = self.param("memory_w2", nn.initializers.normal(0.01), (d, h))
            mem0 = (w1_0, w2_0)
        else:
            mem0 = self.param("memory0", nn.initializers.normal(0.01), (d, d))
        surprise_w = self.param("surprise", nn.initializers.zeros, (n_gates,))
        bias_values = [np.log(0.1 / 0.9), np.log(0.94 / 0.06)] + ([np.log(0.9 / 0.1)] if self.deep else [])
        bias = self.param("gate_bias", lambda *_: jnp.asarray(bias_values, jnp.float32), (n_gates,))
        b = x.shape[0]

        def deep_read(w1, w2, z):
            return jnp.einsum("bdh,bh->bd", w2, jnp.tanh(jnp.einsum("bhd,bd->bh", w1, z)))

        def step(carry, u_t):
            comp, gate, mem = carry                                       # [B,3,d,d] [B,G,d] memory state
            r = u_t[:, None, :] + SELFREF_BETA * jnp.tanh(jnp.einsum("bcij,bj->bci", comp, u_t))
            k, v, q = safe_unit(r[:, 0]), r[:, 1], safe_unit(r[:, 2])
            if self.deep:
                w1, w2, s1, s2 = mem
                act = jnp.tanh(jnp.einsum("bhd,bd->bh", w1, k))
                pred = jnp.einsum("bdh,bh->bd", w2, act)
            else:
                pred = jnp.einsum("bij,bj->bi", mem, k)
            err = v - pred
            surprise = jnp.sqrt(jnp.maximum(jnp.mean(jnp.square(err), axis=-1), EPS * EPS))
            logits = jnp.einsum("bci,bi->bc", gate, u_t) + surprise[:, None] * surprise_w + bias
            eta = jax.nn.sigmoid(logits[:, 0])
            alpha = 0.5 + 0.5 * jax.nn.sigmoid(logits[:, 1])
            e3, a3 = eta[:, None, None], alpha[:, None, None]
            if self.deep:
                # Titans: S <- beta S - eta grad l(M; k, v);  M <- alpha M + S,  l = 1/2 |W2 tanh(W1 k) - v|^2
                mom = jax.nn.sigmoid(logits[:, 2])[:, None, None]
                back = jnp.einsum("bdh,bd->bh", w2, err) * (1.0 - jnp.square(act))
                s2 = _cap(mom * s2 + e3 * jnp.einsum("bd,bh->bdh", err, act), SELFREF_MAIN_CAP, (-2, -1))
                s1 = _cap(mom * s1 + e3 * jnp.einsum("bh,bd->bhd", back, k), SELFREF_MAIN_CAP, (-2, -1))
                w2 = _cap(a3 * w2 + s2, SELFREF_MAIN_CAP, (-2, -1))
                w1 = _cap(a3 * w1 + s1, SELFREF_MAIN_CAP * np.sqrt(h / d), (-2, -1))
                mem = (w1, w2, s1, s2)
                y = deep_read(w1, w2, q)
            else:
                # Titans main memory: M(alpha I - eta k k^T) + eta v k^T  ==  alpha M + eta (v - M k) k^T
                mem = _cap(a3 * mem + e3 * jnp.einsum("bi,bj->bij", err, k), SELFREF_MAIN_CAP, (-2, -1))
                y = jnp.einsum("bij,bj->bi", mem, q)
            # self-reference: components regress onto their own reading of v_t (v_hat = A v_t)
            ec = SELFREF_COMPONENT_RATE * eta
            c_err = jnp.einsum("bcij,bj->bci", comp, v) - jnp.einsum("bcij,bj->bci", comp, k)
            comp = _cap(alpha[:, None, None, None] * comp
                        + ec[:, None, None, None] * jnp.einsum("bci,bj->bcij", c_err, k),
                        SELFREF_MATRIX_CAP, (-2, -1))
            g_err = jnp.einsum("bci,bi->bc", gate, v) - jnp.einsum("bci,bi->bc", gate, k)
            gate = _cap(alpha[:, None, None] * gate + ec[:, None, None] * g_err[..., None] * k[:, None, :],
                        SELFREF_VECTOR_CAP, (-1,))
            return (comp, gate, mem), (y, eta, alpha)

        tile = lambda a: jnp.broadcast_to(a[None], (b,) + a.shape)
        if self.deep:
            mem_state = (tile(mem0[0]), tile(mem0[1]), jnp.zeros((b, h, d), x.dtype), jnp.zeros((b, d, h), x.dtype))
        else:
            mem_state = tile(mem0)
        carry0 = (tile(comp0), tile(gate0), mem_state)
        _, (ys, eta, alpha) = jax.lax.scan(step, carry0, jnp.swapaxes(u, 0, 1))
        out = nn.Dense(self.dim, use_bias=False, name="out_proj")(jnp.swapaxes(ys, 0, 1))
        return out, jnp.swapaxes(eta, 0, 1), jnp.swapaxes(alpha, 0, 1)


class CMSMLP(nn.Module):
    """CMS block of a HOPE level: LN(x + s * MLP(x)); its weights live in the level's update tier."""

    dim: int
    hidden: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = nn.Dense(self.dim, name="down")(nn.gelu(nn.Dense(self.hidden, name="up")(x)))
        scale = self.param("mlp_scale", nn.initializers.constant(0.1), (self.dim,))
        return nn.LayerNorm(epsilon=HOPE_LN_EPS, name="norm")(x + scale * h)


class ShortConv(nn.Module):
    """Titans-style local mixer: LN(x + s * silu(depthwise temporal conv(x))), kernel 3, centred."""

    dim: int
    kernel: int = 3

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        y = nn.Conv(self.dim, (self.kernel,), padding="SAME", feature_group_count=self.dim, name="conv")(x)
        scale = self.param("conv_scale", nn.initializers.constant(0.5), (self.dim,))
        return nn.LayerNorm(epsilon=HOPE_LN_EPS, name="norm")(x + scale * nn.silu(y))


class HopeLevel(nn.Module):
    """One HOPE block: [local mixer: BiGRU or short conv] -> memory (self-referential or R5 fast) -> [CMS MLP]."""

    dim: int
    local: bool
    selfref: bool = True
    mlp: bool = True
    inner: int = 32
    fast_rank: int = 8
    mixer: str = "bigru"        # "bigru" (R5 sweep) | "conv" (Titans short conv)
    deep: bool = False
    mem_hidden: int = 64

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if not self.local:
            base = x
        elif self.mixer == "bigru":
            base = Sweep(self.dim, bidirectional=True, name="base")(x)
        elif self.mixer == "conv":
            base = ShortConv(self.dim, name="mixer")(x)
        else:
            raise ValueError(f"Unknown mixer {self.mixer!r}")
        if self.selfref:
            reads, eta, alpha = SelfRefMemory(self.dim, self.inner, self.deep, self.mem_hidden,
                                              name="selfref")(base)
        else:
            reads, eta, alpha = FastMemory(self.dim, self.fast_rank, name="fast")(base)
        scale = self.param("fast_scale", nn.initializers.constant(0.1), (self.dim,))
        h = nn.LayerNorm(epsilon=HOPE_LN_EPS, name="norm")(base + scale * reads)
        if self.mlp:
            h = CMSMLP(self.dim, self.dim // 2, name="cms_mlp")(h)
        return h, eta, alpha, jnp.mean(jnp.abs(scale))


def chunk_mean(x: jnp.ndarray, period: int) -> jnp.ndarray:
    b, t, d = x.shape
    return x.reshape(b, t // period, period, d).mean(axis=2)


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
    # R6-HOPE temporal core (temporal="r5" keeps R5 bit-for-bit).
    temporal: str = "r5"
    selfref: bool = True
    cms_levels: bool = True
    cms_mlp: bool = True
    selfref_dim: int = 32
    level_mixer: str = "bigru"     # hope_core: "conv" in every level (no BiGRU)
    deep_memory: bool = False      # hope_core: Titans deep memory with momentum
    memory_hidden: int = 64

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

        if self.temporal == "r5":
            # ---- nested temporal memory: M4 over 16 segments, G4 over 4 chunks ----
            m4, eta, alpha, fast_scale = NestedMemory(self.model_dim, self.fast_rank, self.fast_mode,
                                                      name="m4")(f)
            chunks = m4.reshape(b, 4, FRAMES // 4, self.model_dim).mean(axis=2)
            g4, eta_g, alpha_g, fast_scale_g = NestedMemory(self.model_dim, self.fast_rank, self.fast_mode,
                                                            name="g4")(chunks)
            level_means = [m4.mean(axis=1), g4.mean(axis=1)]
        elif self.temporal == "hope":
            # ---- HOPE continuum: nested chain of levels, period 1 -> 2 -> 4 -> 8 (16/8/4/2 steps) ----
            conv = self.level_mixer == "conv"          # a short conv is cheap enough for every level
            level = lambda name, local: HopeLevel(self.model_dim, local, self.selfref, self.cms_mlp,
                                                  self.selfref_dim, self.fast_rank, self.level_mixer,
                                                  self.deep_memory, self.memory_hidden, name=name)
            m4, eta, alpha, fast_scale = level("m4", True)(f)
            level_means = [m4.mean(axis=1)]
            if self.cms_levels:
                l2, _, _, _ = level("l2", conv)(chunk_mean(m4, 2))
                g4, eta_g, alpha_g, fast_scale_g = level("g4", True)(chunk_mean(l2, 2))
                l8, _, _, _ = level("l8", conv)(chunk_mean(g4, 2))
                level_means += [l2.mean(axis=1), g4.mean(axis=1), l8.mean(axis=1)]
            else:
                g4, eta_g, alpha_g, fast_scale_g = level("g4", True)(chunk_mean(m4, 4))
                level_means += [g4.mean(axis=1)]
        else:
            raise ValueError(f"Unknown temporal core {self.temporal!r}")
        m4_mean = level_means[0]
        desc = nn.Dense(self.model_dim, name="descriptor")(jnp.concatenate(level_means, -1))
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
