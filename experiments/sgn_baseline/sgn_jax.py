"""SGN in JAX/Flax, numerically identical to sgn.py (PyTorch, eval mode) given the same weights.

Used only to benchmark SGN under the same compiler (XLA) as NestSAR. `from_torch` copies a PyTorch SGN state_dict.
Layout: activations are [B, C, V, T] like the PyTorch model; 1x1 convolutions are einsums over C.
"""
import jax
import jax.numpy as jnp
import numpy as np


def _conv1x1(p, x):                       # p: {"w": [O, I], "b": [O] or None}
    y = jnp.einsum("oi,bivt->bovt", p["w"], x)
    return y if p.get("b") is None else y + p["b"][None, :, None, None]


def _bn(p, x, axis_shape):                # eval-mode BatchNorm, p: mean, var, gamma, beta (flattened channel axis)
    inv = p["gamma"] / jnp.sqrt(p["var"] + 1e-5)
    return x * inv.reshape(axis_shape) + (p["beta"] - p["mean"] * inv).reshape(axis_shape)


def _embed(p, x, norm):
    if norm:                              # BatchNorm1d over C*V channels
        b, c, v, t = x.shape
        x = _bn(p["bn"], x.reshape(b, c * v, t), (1, -1, 1)).reshape(b, c, v, t)
    return jax.nn.relu(_conv1x1(p["c2"], jax.nn.relu(_conv1x1(p["c1"], x))))


def sgn_apply(P, x, seg=20, joints=25):
    bs = x.shape[0]
    x = x.reshape(bs, seg, joints, 3).transpose(0, 3, 2, 1)                          # [B,3,V,T]
    dif = jnp.concatenate([jnp.zeros_like(x[..., :1]), x[..., 1:] - x[..., :-1]], -1)
    dy = _embed(P["joint_embed"], x, True) + _embed(P["dif_embed"], dif, True)
    spa1 = jnp.broadcast_to(_embed(P["spa_embed"], P["spa_onehot"], False), (bs,) + dy.shape[1:])
    tem1 = _embed(P["tem_embed"], P["tem_onehot"], False)
    h = jnp.concatenate([dy, spa1], 1)
    g1 = jnp.transpose(_conv1x1(P["g1"], h), (0, 3, 2, 1)); g2 = jnp.transpose(_conv1x1(P["g2"], h), (0, 3, 1, 2))
    g = jax.nn.softmax(g1 @ g2, axis=-1)                                                # [B,T,V,V]
    for k in ("gcn1", "gcn2", "gcn3"):
        q = P[k]
        a = jnp.transpose(g @ jnp.transpose(h, (0, 3, 2, 1)), (0, 3, 2, 1))
        h = jax.nn.relu(_bn(q["bn"], _conv1x1(q["w"], a) + _conv1x1(q["w1"], h), (1, -1, 1, 1)))
    h = h + tem1
    h = jnp.max(h, axis=2, keepdims=True)                                               # AdaptiveMaxPool2d((1, seg))
    h = _temporal_conv3(P["cnn1"], h)
    h = jax.nn.relu(_bn(P["bn1"], h, (1, -1, 1, 1)))
    h = jax.nn.relu(_bn(P["bn2"], _conv1x1(P["cnn2"], h), (1, -1, 1, 1)))
    return jnp.max(h, axis=(2, 3)) @ P["fc"]["w"].T + P["fc"]["b"]


def _temporal_conv3(p, h):                # Conv2d(C, C, (1, 3), padding (0, 1)) on [B, C, 1, T]
    hp = jnp.pad(h, ((0, 0), (0, 0), (0, 0), (1, 1)))
    t = h.shape[-1]
    y = sum(jnp.einsum("oi,bivt->bovt", p["w"][:, :, 0, k], hp[..., k:k + t]) for k in range(3))
    return y + p["b"][None, :, None, None]


def from_torch(model):
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    sd.update({k: v.detach().cpu().numpy() for k, v in model.named_buffers()})
    c = lambda n, bias=True: {"w": sd[n + ".weight"][:, :, 0, 0], "b": sd.get(n + ".bias") if bias else None}
    bn = lambda n: {"mean": sd[n + ".running_mean"], "var": sd[n + ".running_var"], "gamma": sd[n + ".weight"], "beta": sd[n + ".bias"]}
    def emb(n, norm):
        layers = [k for k in sd if k.startswith(n + ".cnn.") and k.endswith(".weight") and sd[k].ndim == 4]
        idx = sorted({int(k.split(".")[2]) for k in layers})
        e = {"c1": c(f"{n}.cnn.{idx[0]}"), "c2": c(f"{n}.cnn.{idx[1]}")}
        if norm:
            e["bn"] = bn(f"{n}.cnn.0.bn")
        return e
    P = {"joint_embed": emb("joint_embed", True), "dif_embed": emb("dif_embed", True),
         "spa_embed": emb("spa_embed", False), "tem_embed": emb("tem_embed", False),
         "spa_onehot": sd["spa_onehot"], "tem_onehot": sd["tem_onehot"],
         "g1": c("compute_g1.g1"), "g2": c("compute_g1.g2"),
         "cnn1": {"w": sd["cnn.cnn1.weight"], "b": sd["cnn.cnn1.bias"]}, "bn1": bn("cnn.bn1"),
         "cnn2": c("cnn.cnn2"), "bn2": bn("cnn.bn2"), "fc": {"w": sd["fc.weight"], "b": sd["fc.bias"]}}
    for k in ("gcn1", "gcn2", "gcn3"):
        P[k] = {"w": c(f"{k}.w", bias=False), "w1": c(f"{k}.w1"), "bn": bn(f"{k}.bn")}
    return jax.tree.map(lambda a: None if a is None else jnp.asarray(a), P, is_leaf=lambda a: a is None)
