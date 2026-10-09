import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.nestsar_r5_t16 import model as m5
from experiments.nestsar_r5_t16 import preprocessing as r5
from experiments.nestsar_r5_t16.config import VARIANTS, model_kwargs, validate_config
from experiments.nestsar_r5_t16.worker import EXPECTED_PARAMS
from experiments.nestsar_r5_t16.tests.test_preprocessing import make_clip


def build(variant="full"):
    cfg = validate_config({"variant": variant})
    model = m5.NestSARR5T16(**model_kwargs(cfg))
    x = jnp.zeros((2, r5.FRAMES, r5.FEATURES), jnp.float32)
    params = model.init({"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)},
                        x, training=False)["params"]
    return model, params


def real_batch(n=6, seed=0):
    xs = []
    for i in range(n):
        people = 2 if i % 2 else 1
        clip = make_clip(int(20 + 37 * i), people, seed=seed + i, drop=0.02)
        xs.append(r5.augmented_features(clip, 128, 1, i) if i % 3 == 0 else r5.features(clip))
    return jnp.asarray(np.stack(xs))


@pytest.fixture(scope="module")
def full():
    return build("full")


def test_skeleton_constants_match_r4():
    from experiments.nestsar_sm_all_t16 import model as r4
    np.testing.assert_array_equal(m5.PARENTS, r4.base.PARENTS)
    np.testing.assert_array_equal(m5.JOINT_ORDER, r4.base.JOINT_ORDER)
    assert tuple(map(tuple, m5.TEN_PARTS)) == tuple(map(tuple, r4.base.TEN_PARTS))
    m5.part_matrix(m5.FOURTEEN_PARTS)
    m5.part_matrix(m5.TEN_PARTS)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_parameter_counts_are_recorded(variant):
    _, params = build(variant)
    n = sum(int(np.prod(a.shape)) for a in jax.tree_util.tree_leaves(params))
    assert n == EXPECTED_PARAMS[variant]


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_forward_and_gradients_finite_with_zero_padding(variant):
    model, params = build(variant)
    x = real_batch(5)
    x = jnp.concatenate([x, jnp.zeros_like(x[:2])])          # zero-padded rows, like a last batch
    mask = jnp.asarray([1, 1, 1, 1, 1, 0, 0], jnp.float32)
    y = jnp.arange(7) % 120

    def loss(p):
        out = model.apply({"params": p}, x, training=True, rngs={"dropout": jax.random.PRNGKey(3)})
        ce = -jnp.take_along_axis(jax.nn.log_softmax(out["logits"]), y[:, None], 1)[:, 0]
        aux = -jnp.take_along_axis(jax.nn.log_softmax(out["stream_logits"]),
                                   jnp.broadcast_to(y[:, None, None], out["stream_logits"].shape[:2] + (1,)),
                                   2)[..., 0].mean(1)
        return jnp.sum((ce + aux) * mask) / mask.sum()

    value, grads = jax.value_and_grad(loss)(params)
    assert np.isfinite(float(value))
    leaves = jax.tree_util.tree_leaves(grads)
    assert all(bool(jnp.isfinite(g).all()) for g in leaves)
    assert sum(float(jnp.abs(g).sum()) for g in leaves) > 0

    out = model.apply({"params": params}, x, training=False)
    for k, v in out.items():
        assert bool(jnp.isfinite(v).all()), k
    assert out["logits"].shape == (7, 120)
    s_aux = 1 if variant == "no_hand_branch" else 2
    assert out["stream_logits"].shape == (7, s_aux, 120)
    for k in ("sm_eta_mean", "sm_alpha_mean", "fast_scale_m4", "pair_rate", "hand_valid_rate"):
        assert out[k].shape == (7,), k


def test_eval_is_deterministic_and_batch_independent(full):
    model, params = full
    x = real_batch(4)
    a = model.apply({"params": params}, x, training=False)["logits"]
    b = model.apply({"params": params}, x, training=False)["logits"]
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    single = model.apply({"params": params}, x[1:2], training=False)["logits"]
    np.testing.assert_allclose(np.asarray(single[0]), np.asarray(a[1]), rtol=2e-5, atol=2e-5)


def test_absent_person_hand_block_is_ignored(full):
    model, params = full
    clip = make_clip(80, 1, seed=4)
    x = np.asarray(r5.features(clip))[None]
    body, hands = r5.split(x)
    assert np.abs(body[..., 1, :, :]).max() == 0
    noisy = hands.copy()
    noisy[:, :, :, 1] = np.random.default_rng(0).normal(0, 1, noisy[:, :, :, 1].shape)
    x2 = np.concatenate([x[..., :750], noisy.reshape(1, 16, -1)], axis=-1)
    a = model.apply({"params": params}, jnp.asarray(x), training=False)
    b = model.apply({"params": params}, jnp.asarray(x2), training=False)
    np.testing.assert_allclose(np.asarray(a["logits"]), np.asarray(b["logits"]), atol=1e-5)
    assert float(a["pair_rate"][0]) == 0.0


def test_hand_block_changes_the_prediction(full):
    model, params = full
    x = np.asarray(real_batch(2))
    x2 = x.copy()
    x2[..., 750:] *= 1.5
    a = model.apply({"params": params}, jnp.asarray(x), training=False)["logits"]
    b = model.apply({"params": params}, jnp.asarray(x2), training=False)["logits"]
    assert float(jnp.abs(a - b).max()) > 1e-4


def test_fast_memory_matches_loop_and_stays_bounded():
    fm = m5.FastMemory(dim=12, rank=3)
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 9, 12)) * 50.0      # extreme inputs
    params = fm.init(jax.random.PRNGKey(1), x)["params"]
    # Make the gates input- and surprise-dependent to exercise every term.
    params = jax.tree_util.tree_map(lambda a: a, params)
    params["gate"]["kernel"] = jax.random.normal(jax.random.PRNGKey(2), params["gate"]["kernel"].shape)
    params["surprise"] = jnp.asarray([0.7, -0.4])
    reads, eta, alpha = fm.apply({"params": params}, x)
    assert bool(jnp.isfinite(reads).all())
    assert float(eta.min()) > 0 and float(eta.max()) < 1
    assert float(alpha.min()) >= 0.5 and float(alpha.max()) <= 1

    # Independent loop implementation.
    p = jax.tree_util.tree_map(np.asarray, params)
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    v = np.asarray((x - mu) / np.sqrt(var + 1e-6)) * p["value_norm"]["scale"] + p["value_norm"]["bias"]

    def unit(a):
        return a / np.sqrt(np.maximum((a ** 2).sum(-1, keepdims=True), 1e-12))

    k = unit(np.tanh(v @ p["key"]["kernel"]))
    q = unit(np.tanh(v @ p["query"]["kernel"]))
    g = v @ p["gate"]["kernel"]
    out = np.zeros((2, 9, 12))
    for b in range(2):
        mem = p["memory0"].astype(np.float64).copy()
        for t in range(9):
            err = v[b, t] - k[b, t] @ mem
            s = np.sqrt(max((err ** 2).mean(), 1e-12))
            logit = g[b, t] + s * p["surprise"] + p["gate_bias"]
            e = 1 / (1 + np.exp(-logit[0]))
            a = 0.5 + 0.5 / (1 + np.exp(-logit[1]))
            mem = a * mem + e * np.outer(k[b, t], err)
            out[b, t] = q[b, t] @ mem
    np.testing.assert_allclose(np.asarray(reads), out, rtol=2e-3, atol=2e-3)


def test_initial_gates_match_r4_starting_point():
    fm = m5.FastMemory(dim=8, rank=2)
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 5, 8))
    params = fm.init(jax.random.PRNGKey(1), x)["params"]
    _, eta, alpha = fm.apply({"params": params}, x)
    np.testing.assert_allclose(np.asarray(eta), 0.1, atol=1e-6)
    np.testing.assert_allclose(np.asarray(alpha), 0.97, atol=1e-6)


def test_pair_relations_zero_without_partner_and_symmetric_distances():
    rng = np.random.default_rng(0)
    pa = jnp.asarray(rng.normal(size=(3, 25, 3)), jnp.float32)
    pb = jnp.asarray(rng.normal(size=(3, 25, 3)), jnp.float32)
    da = jnp.asarray(rng.normal(size=(3, 25, 3)), jnp.float32)
    db = jnp.asarray(rng.normal(size=(3, 25, 3)), jnp.float32)
    pair = jnp.asarray([1.0, 0.0, 1.0])
    ab = m5.pair_relations(pa, pb, da, db, pair)
    ba = m5.pair_relations(pb, pa, db, da, pair)
    assert ab.shape == (3, m5.PAIR_FEATURES)
    assert float(jnp.abs(ab[1]).max()) == 0.0
    np.testing.assert_allclose(np.asarray(ab[:, 12]), np.asarray(ba[:, 12]), rtol=1e-6)   # root distance
    np.testing.assert_allclose(np.asarray(ab[:, 13:16]), -np.asarray(ba[:, 13:16]), rtol=1e-6)  # offset


def test_single_scan_bigru_equals_two_gru_sweeps():
    x = jax.random.normal(jax.random.PRNGKey(0), (5, 11, 7))
    bi = m5.BiGRUSweep(hidden=6)
    p = bi.init(jax.random.PRNGKey(1), x)["params"]
    got = bi.apply({"params": p}, x)
    fwd = m5.GRUSweep(hidden=6)
    bwd = m5.GRUSweep(hidden=6, reverse=True)
    pf = {"input": p["input_fwd"], "hidden_zr": p["hidden_zr"][0], "hidden_c": p["hidden_c"][0]}
    pb = {"input": p["input_bwd"], "hidden_zr": p["hidden_zr"][1], "hidden_c": p["hidden_c"][1]}
    want = jnp.concatenate([fwd.apply({"params": pf}, x), bwd.apply({"params": pb}, x)], axis=-1)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-6)
    # Different directions get different initial hidden weights.
    assert not np.allclose(np.asarray(p["hidden_zr"][0]), np.asarray(p["hidden_zr"][1]))
