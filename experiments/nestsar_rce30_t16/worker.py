from __future__ import annotations

"""Frozen-FMSE Stage-2 trainer for NestSAR-RCEX.

RCEX removes the train/eval mismatch in RCE30: the true label is NEVER inserted
into the candidate set.  Candidate recall is learned by a global evidence
retrieval head and improved with the frozen FMSE stream oracle.

Only RCEX parameters are optimized.  FMSE EMA parameters never enter the
optimizer state.
"""

import argparse
import csv
import hashlib
import json
import math
from contextlib import closing
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from flax.training import train_state

from experiments.nestsar_r4_fmse_t16.model import NestSARR4FMSET16
from experiments.nestsar_sm_all_t16.preprocessing_corrected import (
    FRAMES,
    FEATURES,
    VERSION as PREPROCESSING_VERSION,
)
from experiments.nestsar_sm_all_t16.streaming.data import Dataset
from experiments.nestsar_sm_all_t16.streaming.io_utils import (
    Reporter,
    atomic_bytes,
    atomic_json,
    read_json,
)

from . import MODEL_IDENTITY, MODEL_NAME, VERSION
from .model import (
    NestSARRCEXT16,
    NUM_CLASSES,
    NUM_STREAMS,
    BASE_DESCRIPTOR_DIM,
    RIVALS_FOR_BASE_TOP1,
)


TRAIN_METRICS = 18
EVAL_METRICS = 11


class RCEState(train_state.TrainState):
    ema_params: object


def ce(logits, labels, smoothing):
    target = jax.nn.one_hot(labels, NUM_CLASSES)
    target = (
        target * (1.0 - smoothing)
        + smoothing / NUM_CLASSES
    )
    return -jnp.sum(
        target * jax.nn.log_softmax(logits),
        axis=-1,
    )


def _sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(
            f,
            "sha256",
        ).hexdigest()


def load_base_checkpoint(path):
    path = Path(path)
    payload = serialization.msgpack_restore(
        path.read_bytes()
    )

    if (
        "ema_params" not in payload
        or "config" not in payload
    ):
        raise ValueError(
            "Expected FMSE best checkpoint with ema_params/config: "
            f"{path}"
        )

    cfg = payload["config"]

    model = NestSARR4FMSET16(
        spatial_dim=cfg["spatial_dim"],
        model_dim=cfg["model_dim"],
        dropout=cfg["dropout"],
        controller_dim=cfg["controller_dim"],
        fast_rank=cfg["fast_rank"],
        head_rank=cfg["head_rank"],
        sm_residual_scale=cfg["sm_residual_scale"],
        head_residual_scale=cfg["head_residual_scale"],
    )

    params = jax.device_put(
        payload["ema_params"]
    )

    return model, params, payload


def make_specialist(config):
    return NestSARRCEXT16(
        dim=40,
        blocks=2,
        rank=4,
        dropout=config["rce_dropout"],
    )


def dummy_base_inputs(batch=1):
    return (
        jnp.zeros(
            (batch, NUM_CLASSES),
            jnp.float32,
        ),
        jnp.zeros(
            (batch, NUM_STREAMS, NUM_CLASSES),
            jnp.float32,
        ),
        jnp.zeros(
            (batch, NUM_STREAMS, BASE_DESCRIPTOR_DIM),
            jnp.float32,
        ),
        jnp.ones(
            (batch, NUM_STREAMS),
            jnp.float32,
        ) / NUM_STREAMS,
    )


def specialist_parameter_count(
    model,
    rival_table,
):
    key = jax.random.PRNGKey(7)

    (
        base_logits,
        stream_logits,
        descriptors,
        fusion_weights,
    ) = dummy_base_inputs(1)

    variables = model.init(
        {
            "params": key,
            "dropout": key,
        },
        jnp.zeros(
            (1, FRAMES, FEATURES),
            jnp.float32,
        ),
        base_logits,
        stream_logits,
        descriptors,
        fusion_weights,
        rival_table,
        training=False,
    )

    return int(
        sum(
            x.size
            for x in jax.tree.leaves(
                variables["params"]
            )
        )
    )


def validate_config(config):
    defaults = dict(
        epochs=40,
        patience=7,
        micro_batch=64,
        accumulation_steps=4,
        eval_batch=256,
        seed=128,

        rce_learning_rate=6e-4,
        rce_min_learning_rate=1e-5,
        rce_warmup_fraction=0.08,
        rce_weight_decay=0.03,
        rce_grad_clip=1.0,
        rce_ema_decay=0.995,
        rce_dropout=0.06,

        label_smoothing=0.02,

        retrieval_weight=0.18,
        rival_weight=0.10,
        protect_weight=0.25,
        margin_protect_weight=0.20,
        gate_supervision_weight=0.08,
        correction_l2_weight=0.01,
        consistency_weight=0.01,

        easy_example_weight=0.35,
        hard_example_boost=1.75,
        hard_margin=1.0,
        protect_margin_tolerance=0.05,

        fresh_augmentation=True,
        rotation_degrees=8.0,
        jitter_shift=1,

        min_delta=1e-6,
        progress_every=5,
        max_train_samples=0,
        max_val_samples=0,
        prefetch_batches=2,

        rival_graph_batch=256,
        rival_graph_topk=5,
        rival_graph_aug_epochs=2,
    )

    unknown = set(config) - set(defaults)
    if unknown:
        raise ValueError(
            f"Unknown RCEX config keys: {sorted(unknown)}"
        )

    c = {
        **defaults,
        **config,
    }

    for key in (
        "epochs",
        "patience",
        "micro_batch",
        "accumulation_steps",
        "eval_batch",
        "progress_every",
        "rival_graph_batch",
        "rival_graph_topk",
        "rival_graph_aug_epochs",
    ):
        if not isinstance(c[key], int) or c[key] < 1:
            raise ValueError(
                f"{key} must be a positive integer"
            )

    if c["prefetch_batches"] not in (1, 2):
        raise ValueError(
            "prefetch_batches must be 1 or 2"
        )

    return c


def _base_outputs(
    base_model,
    base_params,
    x,
):
    return base_model.apply(
        {"params": base_params},
        x,
        training=False,
    )


def build_training_rivals(
    *,
    dataset,
    train_ids,
    base_model,
    base_params,
    config,
    protocol,
    out,
):
    """Training-only rival graph using final + stream predictions.

    Clean predictions are used once.  Fresh augmented views are used for
    multiple epochs.  Validation data is never touched.
    """

    npy = out / "training_rivals.npy"
    meta_path = out / "training_rivals.json"

    if npy.is_file() and meta_path.is_file():
        table = np.load(npy)
        meta = json.loads(
            meta_path.read_text()
        )
        if table.shape == (
            NUM_CLASSES,
            RIVALS_FOR_BASE_TOP1,
        ):
            return (
                jax.device_put(
                    table.astype(np.int32)
                ),
                meta,
            )

    batch_size = int(
        config["rival_graph_batch"]
    )
    topk = int(
        config["rival_graph_topk"]
    )

    @jax.jit
    def infer(x):
        out = _base_outputs(
            base_model,
            base_params,
            x,
        )
        return (
            out["logits"],
            out["stream_logits"],
        )

    counts = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        np.float64,
    )

    steps = math.ceil(
        len(train_ids) / batch_size
    )

    def accumulate_view(
        logits,
        stream_logits,
        y,
        weight,
    ):
        # Rank-aware final-model rivals.
        order = np.argsort(
            logits,
            axis=1,
        )[:, -topk:][:, ::-1]

        for rank in range(topk):
            rival = order[:, rank]
            keep = rival != y
            np.add.at(
                counts,
                (y[keep], rival[keep]),
                weight / (1.0 + rank),
            )

        # Stream winners expose the proven complementary-stream signal.
        stream_winners = np.argmax(
            stream_logits,
            axis=-1,
        )
        for stream in range(NUM_STREAMS):
            rival = stream_winners[:, stream]
            keep = rival != y
            np.add.at(
                counts,
                (y[keep], rival[keep]),
                0.75 * weight,
            )

    for aug_epoch in range(
        1,
        config["rival_graph_aug_epochs"] + 1,
    ):
        with closing(
            dataset.batches(
                train_ids,
                batch_size,
                config,
                epoch=aug_epoch,
                training=True,
                protocol=protocol,
            )
        ) as batches:
            for _ in range(steps):
                batch, _ = next(batches)
                mask = batch["mask"].astype(bool)
                y = batch["y"][mask]

                if len(y) == 0:
                    continue

                # Clean view only once.
                if aug_epoch == 1:
                    logits, stream_logits = jax.block_until_ready(
                        infer(
                            jax.device_put(
                                batch["x"]
                            )
                        )
                    )
                    accumulate_view(
                        np.asarray(logits)[mask],
                        np.asarray(stream_logits)[mask],
                        y,
                        1.0,
                    )

                logits, stream_logits = jax.block_until_ready(
                    infer(
                        jax.device_put(
                            batch["xa"]
                        )
                    )
                )
                accumulate_view(
                    np.asarray(logits)[mask],
                    np.asarray(stream_logits)[mask],
                    y,
                    1.25,
                )

    table = np.zeros(
        (
            NUM_CLASSES,
            RIVALS_FOR_BASE_TOP1,
        ),
        np.int32,
    )

    for c in range(NUM_CLASSES):
        row = counts[c].copy()
        row[c] = -np.inf
        order = np.argsort(
            row
        )[::-1]

        chosen = [
            int(v)
            for v in order
            if v != c
        ][
            :RIVALS_FOR_BASE_TOP1
        ]

        if len(chosen) != RIVALS_FOR_BASE_TOP1:
            raise RuntimeError(
                f"Could not derive rivals for class {c}"
            )

        table[c] = chosen

    np.save(
        npy,
        table,
    )

    meta = {
        "source": (
            "training-only FMSE clean + fresh augmented predictions "
            "+ stream winners"
        ),
        "protocol": protocol,
        "train_samples": len(train_ids),
        "final_topk": topk,
        "aug_epochs": int(
            config["rival_graph_aug_epochs"]
        ),
        "rivals_per_class": RIVALS_FOR_BASE_TOP1,
        "table": table.tolist(),
    }

    atomic_json(
        meta_path,
        meta,
    )

    return (
        jax.device_put(table),
        meta,
    )


def create_state(
    config,
    steps_per_epoch,
    specialist,
    rival_table,
):
    total = (
        config["epochs"]
        * steps_per_epoch
    )

    warm = max(
        1,
        int(
            total
            * config["rce_warmup_fraction"]
        ),
    )

    warm = min(
        warm,
        max(total - 1, 1),
    )

    schedule = optax.warmup_cosine_decay_schedule(
        0.0,
        config["rce_learning_rate"],
        warm,
        max(total, warm + 1),
        end_value=config[
            "rce_min_learning_rate"
        ],
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(
            config["rce_grad_clip"]
        ),
        optax.adamw(
            schedule,
            weight_decay=config[
                "rce_weight_decay"
            ],
        ),
    )

    key, init = jax.random.split(
        jax.random.PRNGKey(
            config["seed"]
        )
    )

    (
        base_logits,
        stream_logits,
        descriptors,
        fusion_weights,
    ) = dummy_base_inputs(1)

    variables = specialist.init(
        {
            "params": init,
            "dropout": init,
        },
        jnp.zeros(
            (1, FRAMES, FEATURES),
            jnp.float32,
        ),
        base_logits,
        stream_logits,
        descriptors,
        fusion_weights,
        rival_table,
        training=False,
    )

    params = variables["params"]

    state = RCEState.create(
        apply_fn=specialist.apply,
        params=params,
        tx=optimizer,
        ema_params=params,
    )

    return (
        state,
        key,
        schedule,
        int(
            math.ceil(
                warm / steps_per_epoch
            )
        ),
    )


def build_steps(
    *,
    base_model,
    base_params,
    specialist,
    rival_table,
    config,
):
    @jax.jit
    def base_forward(x):
        out = _base_outputs(
            base_model,
            base_params,
            x,
        )
        return (
            out["logits"],
            out["stream_logits"],
            out["descriptors"],
            out["fusion_weights"],
        )

    def one_view(
        params,
        key,
        x,
        y,
        base_tuple,
    ):
        (
            base_logits,
            stream_logits,
            descriptors,
            fusion_weights,
        ) = base_tuple

        out = specialist.apply(
            {"params": params},
            x,
            base_logits,
            stream_logits,
            descriptors,
            fusion_weights,
            rival_table,
            training=True,
            rngs={
                "dropout": key,
            },
        )

        logits = out["logits"]
        rows = jnp.arange(
            y.shape[0]
        )

        base_pred = jnp.argmax(
            base_logits,
            axis=-1,
        )
        final_pred = jnp.argmax(
            logits,
            axis=-1,
        )

        base_correct = (
            base_pred == y
        )

        coverage = out[
            "candidate_mask"
        ][
            rows,
            y,
        ]

        top2 = jax.lax.top_k(
            base_logits,
            2,
        )[0]
        base_conf_margin = (
            top2[:, 0] - top2[:, 1]
        )

        hard = (
            (~base_correct)
            | (
                base_conf_margin
                < config["hard_margin"]
            )
            | (
                out["stream_disagreement"]
                > 0.25
            )
        )

        sample_weight = jnp.where(
            hard,
            config["hard_example_boost"],
            config["easy_example_weight"],
        )

        main_ce = ce(
            logits,
            y,
            config["label_smoothing"],
        )

        retrieval_ce = ce(
            out["retrieval_logits"],
            y,
            config["label_smoothing"],
        )

        true_logit = logits[
            rows,
            y,
        ]

        not_true = (
            1.0
            - jax.nn.one_hot(
                y,
                NUM_CLASSES,
            )
        )

        rival_logits = jnp.where(
            (
                out["candidate_mask"]
                * not_true
            )
            > 0,
            logits,
            -1e9,
        )

        hardest = jnp.max(
            rival_logits,
            axis=-1,
        )

        rival_loss = (
            jax.nn.softplus(
                hardest - true_logit
            )
            * coverage
        )

        base_logp = jax.nn.log_softmax(
            jax.lax.stop_gradient(
                base_logits
            )
        )
        base_p = jnp.exp(
            base_logp
        )
        final_logp = jax.nn.log_softmax(
            logits
        )

        protect_kl = jnp.sum(
            base_p
            * (
                base_logp
                - final_logp
            ),
            axis=-1,
        ) * base_correct.astype(
            jnp.float32
        )

        # Preserve the base true-vs-best-rival margin on already-correct clips.
        true_base = base_logits[
            rows,
            y,
        ]
        other_base = jnp.max(
            jnp.where(
                not_true > 0,
                base_logits,
                -1e9,
            ),
            axis=-1,
        )
        base_true_margin = (
            true_base - other_base
        )

        other_final = jnp.max(
            jnp.where(
                not_true > 0,
                logits,
                -1e9,
            ),
            axis=-1,
        )
        final_true_margin = (
            true_logit - other_final
        )

        margin_protect = (
            jax.nn.relu(
                base_true_margin
                - config[
                    "protect_margin_tolerance"
                ]
                - final_true_margin
            )
            * base_correct.astype(
                jnp.float32
            )
        )

        # Explicit gate supervision removes the previous always-small gate
        # objective.  Open only when base is wrong AND the natural candidate
        # set contains the truth; otherwise protect the base.
        gate_target = (
            (~base_correct)
            & (coverage > 0)
        ).astype(
            jnp.float32
        )

        gate_bce = optax.sigmoid_binary_cross_entropy(
            out["gate_logit"],
            gate_target,
        )

        correction_l2 = jnp.mean(
            out["delta_logits"] ** 2,
            axis=-1,
        ) * base_correct.astype(
            jnp.float32
        )

        return {
            "out": out,
            "main_ce": main_ce,
            "retrieval_ce": retrieval_ce,
            "rival": rival_loss,
            "protect": protect_kl,
            "margin_protect": margin_protect,
            "gate_bce": gate_bce,
            "correction_l2": correction_l2,
            "coverage": coverage,
            "sample_weight": sample_weight,
            "base_correct": base_correct,
            "final_pred": final_pred,
            "base_pred": base_pred,
        }

    def per_sample(
        params,
        key,
        batch,
    ):
        k1, k2 = jax.random.split(
            key
        )

        base_clean = base_forward(
            batch["x"]
        )
        base_aug = base_forward(
            batch["xa"]
        )

        clean = one_view(
            params,
            k1,
            batch["x"],
            batch["y"],
            base_clean,
        )

        aug = one_view(
            params,
            k2,
            batch["xa"],
            batch["y"],
            base_aug,
        )

        def avg(name):
            return 0.5 * (
                clean[name]
                + aug[name]
            )

        main_ce = avg(
            "main_ce"
        )
        retrieval_ce = avg(
            "retrieval_ce"
        )
        rival = avg(
            "rival"
        )
        protect = avg(
            "protect"
        )
        margin_protect = avg(
            "margin_protect"
        )
        gate_bce = avg(
            "gate_bce"
        )
        correction_l2 = avg(
            "correction_l2"
        )
        coverage = avg(
            "coverage"
        )
        sample_weight = avg(
            "sample_weight"
        )

        logp = jax.nn.log_softmax(
            clean["out"]["logits"]
        )
        logq = jax.nn.log_softmax(
            aug["out"]["logits"]
        )

        consistency = (
            0.5
            * jnp.sum(
                (
                    jnp.exp(logp)
                    - jnp.exp(logq)
                )
                * (logp - logq),
                axis=-1,
            )
        )

        loss = (
            sample_weight * main_ce
            + config["retrieval_weight"]
            * retrieval_ce
            + config["rival_weight"]
            * rival
            + config["protect_weight"]
            * protect
            + config["margin_protect_weight"]
            * margin_protect
            + config["gate_supervision_weight"]
            * gate_bce
            + config["correction_l2_weight"]
            * correction_l2
            + config["consistency_weight"]
            * consistency
        )

        y = batch["y"]

        pred = clean[
            "final_pred"
        ]
        base_pred = clean[
            "base_pred"
        ]

        acc = (
            pred == y
        ).astype(
            jnp.float32
        )
        base_acc = (
            base_pred == y
        ).astype(
            jnp.float32
        )

        fixed = (
            (base_pred != y)
            & (pred == y)
        ).astype(
            jnp.float32
        )

        broken = (
            (base_pred == y)
            & (pred != y)
        ).astype(
            jnp.float32
        )

        retrieval_pred = jnp.argmax(
            clean["out"]["retrieval_logits"],
            axis=-1,
        )

        retrieval_acc = (
            retrieval_pred == y
        ).astype(
            jnp.float32
        )

        gate = 0.5 * (
            clean["out"]["gate"]
            + aug["out"]["gate"]
        )

        stream_disagreement = 0.5 * (
            clean["out"]["stream_disagreement"]
            + aug["out"]["stream_disagreement"]
        )

        return jnp.stack(
            [
                loss,                    # 0
                main_ce,                 # 1
                retrieval_ce,            # 2
                rival,                   # 3
                protect,                 # 4
                margin_protect,          # 5
                gate_bce,                # 6
                correction_l2,           # 7
                consistency,             # 8
                acc,                     # 9
                base_acc,                # 10
                fixed,                   # 11
                broken,                  # 12
                gate,                    # 13
                coverage,                # 14
                retrieval_acc,           # 15
                stream_disagreement,     # 16
                sample_weight,           # 17
            ],
            axis=-1,
        )

    @jax.jit
    def train_step(
        state,
        key,
        batch,
    ):
        k = config[
            "accumulation_steps"
        ]

        micros = jax.tree.map(
            lambda x: x.reshape(
                k,
                config["micro_batch"],
                *x.shape[1:],
            ),
            batch,
        )

        denom = jnp.maximum(
            jnp.sum(
                batch["mask"]
            ),
            1.0,
        )

        key, step_key = jax.random.split(
            key
        )

        drop_keys = jax.random.split(
            step_key,
            k,
        )

        zero = jax.tree.map(
            jnp.zeros_like,
            state.params,
        )

        def accumulate(
            carry,
            inputs,
        ):
            gradient_sum, metric_sum = carry
            micro, drop = inputs

            def loss_fn(params):
                metrics = per_sample(
                    params,
                    drop,
                    micro,
                )
                totals = jnp.sum(
                    metrics
                    * micro["mask"][:, None],
                    axis=0,
                )
                return (
                    totals[0] / denom,
                    totals,
                )

            (
                (_, totals),
                gradients,
            ) = jax.value_and_grad(
                loss_fn,
                has_aux=True,
            )(
                state.params
            )

            return (
                (
                    jax.tree.map(
                        lambda a, b: a + b,
                        gradient_sum,
                        gradients,
                    ),
                    metric_sum + totals,
                ),
                None,
            )

        (
            (grads, metrics),
            _,
        ) = jax.lax.scan(
            accumulate,
            (
                zero,
                jnp.zeros(
                    TRAIN_METRICS
                ),
            ),
            (
                micros,
                drop_keys,
            ),
        )

        norm = optax.global_norm(
            grads
        )

        state = state.apply_gradients(
            grads=grads
        )

        ema = jax.tree.map(
            lambda e, p: (
                config["rce_ema_decay"]
                * e
                + (
                    1.0
                    - config["rce_ema_decay"]
                )
                * p
            ),
            state.ema_params,
            state.params,
        )

        return (
            state.replace(
                ema_params=ema
            ),
            key,
            jnp.r_[
                metrics,
                denom,
                norm,
            ],
        )

    @jax.jit
    def eval_step(
        params,
        batch,
    ):
        (
            base_logits,
            stream_logits,
            descriptors,
            fusion_weights,
        ) = base_forward(
            batch["x"]
        )

        out = specialist.apply(
            {"params": params},
            batch["x"],
            base_logits,
            stream_logits,
            descriptors,
            fusion_weights,
            rival_table,
            training=False,
        )

        logits = out[
            "logits"
        ]

        y = batch[
            "y"
        ]
        mask = batch[
            "mask"
        ]

        pred = jnp.argmax(
            logits,
            axis=-1,
        )
        base_pred = jnp.argmax(
            base_logits,
            axis=-1,
        )

        top5 = jax.lax.top_k(
            logits,
            5,
        )[1]

        retrieval_top5 = jax.lax.top_k(
            out["retrieval_logits"],
            5,
        )[1]

        rows = jnp.arange(
            y.shape[0]
        )

        columns = [
            ce(
                logits,
                y,
                0.0,
            ),
            (
                pred == y
            ).astype(
                jnp.float32
            ),
            (
                base_pred == y
            ).astype(
                jnp.float32
            ),
            jnp.any(
                top5 == y[:, None],
                axis=-1,
            ).astype(
                jnp.float32
            ),
            (
                (base_pred != y)
                & (pred == y)
            ).astype(
                jnp.float32
            ),
            (
                (base_pred == y)
                & (pred != y)
            ).astype(
                jnp.float32
            ),
            out["gate"],
            out["candidate_mask"][
                rows,
                y,
            ],
            jnp.any(
                retrieval_top5
                == y[:, None],
                axis=-1,
            ).astype(
                jnp.float32
            ),
            out["stream_disagreement"],
            jnp.ones_like(
                mask
            ),
        ]

        return jnp.sum(
            jnp.stack(
                columns,
                axis=-1,
            )
            * mask[:, None],
            axis=0,
        )

    return (
        train_step,
        eval_step,
    )


def save_last(
    path,
    state,
    key,
    metadata,
):
    payload = {
        "state": serialization.to_state_dict(
            jax.device_get(state)
        ),
        "key": np.asarray(key),
        "metadata_json": json.dumps(
            metadata,
            allow_nan=False,
        ),
    }

    atomic_bytes(
        path,
        serialization.msgpack_serialize(
            payload
        ),
    )


def restore_last(
    path,
    template,
    digest,
):
    payload = serialization.msgpack_restore(
        Path(path).read_bytes()
    )

    meta = json.loads(
        payload["metadata_json"]
    )

    if meta["config_hash"] != digest:
        raise ValueError(
            "Resume config/base/rival mismatch. "
            "Keep config or use new OUT_DIR."
        )

    state = serialization.from_state_dict(
        template,
        payload["state"],
    )

    return (
        jax.device_put(state),
        jax.device_put(
            payload["key"]
        ),
        meta,
    )


def run_per_class_audit(
    *,
    dataset,
    val_ids,
    base_model,
    base_params,
    specialist,
    specialist_params,
    rival_table,
    config,
    protocol,
    out,
):
    """Top1/Top5/fixed/broken/coverage/retrieval for all 120 classes."""

    @jax.jit
    def step(
        x,
        y,
        mask,
    ):
        base = _base_outputs(
            base_model,
            base_params,
            x,
        )

        result = specialist.apply(
            {"params": specialist_params},
            x,
            base["logits"],
            base["stream_logits"],
            base["descriptors"],
            base["fusion_weights"],
            rival_table,
            training=False,
        )

        logits = result["logits"]
        pred = jnp.argmax(
            logits,
            axis=-1,
        )
        base_pred = jnp.argmax(
            base["logits"],
            axis=-1,
        )

        top5 = jax.lax.top_k(
            logits,
            5,
        )[1]

        retrieval_top5 = jax.lax.top_k(
            result["retrieval_logits"],
            5,
        )[1]

        rows = jnp.arange(
            y.shape[0]
        )

        return (
            pred,
            base_pred,
            jnp.any(
                top5 == y[:, None],
                axis=-1,
            ),
            result["candidate_mask"][
                rows,
                y,
            ],
            result["gate"],
            jnp.any(
                retrieval_top5
                == y[:, None],
                axis=-1,
            ),
            result["stream_disagreement"],
            mask,
        )

    counts = np.zeros(
        NUM_CLASSES,
        np.int64,
    )
    correct = np.zeros_like(
        counts
    )
    top5_ok = np.zeros_like(
        counts
    )
    base_correct = np.zeros_like(
        counts
    )
    fixed = np.zeros_like(
        counts
    )
    broken = np.zeros_like(
        counts
    )
    coverage = np.zeros(
        NUM_CLASSES,
        np.float64,
    )
    gate_sum = np.zeros_like(
        coverage
    )
    retrieval5 = np.zeros_like(
        coverage
    )
    disagreement = np.zeros_like(
        coverage
    )

    batch_size = int(
        config["eval_batch"]
    )

    steps = math.ceil(
        len(val_ids)
        / batch_size
    )

    with closing(
        dataset.batches(
            val_ids,
            batch_size,
            config,
            protocol=protocol,
        )
    ) as batches:
        for _ in range(steps):
            batch, _ = next(
                batches
            )

            y = batch["y"]

            (
                pred,
                base_pred,
                t5,
                cov,
                gate,
                rt5,
                dis,
                mask,
            ) = jax.block_until_ready(
                step(
                    jax.device_put(
                        batch["x"]
                    ),
                    jax.device_put(
                        batch["y"]
                    ),
                    jax.device_put(
                        batch["mask"]
                    ),
                )
            )

            pred = np.asarray(pred)
            base_pred = np.asarray(
                base_pred
            )
            t5 = np.asarray(t5)
            cov = np.asarray(cov)
            gate = np.asarray(gate)
            rt5 = np.asarray(rt5)
            dis = np.asarray(dis)
            mask = np.asarray(
                mask
            ).astype(bool)

            yy = y[mask]
            pp = pred[mask]
            bp = base_pred[mask]
            tt = t5[mask]
            cc = cov[mask]
            gg = gate[mask]
            rr = rt5[mask]
            dd = dis[mask]

            np.add.at(
                counts,
                yy,
                1,
            )
            np.add.at(
                correct,
                yy,
                pp == yy,
            )
            np.add.at(
                top5_ok,
                yy,
                tt,
            )
            np.add.at(
                base_correct,
                yy,
                bp == yy,
            )
            np.add.at(
                fixed,
                yy,
                (bp != yy)
                & (pp == yy),
            )
            np.add.at(
                broken,
                yy,
                (bp == yy)
                & (pp != yy),
            )
            np.add.at(
                coverage,
                yy,
                cc,
            )
            np.add.at(
                gate_sum,
                yy,
                gg,
            )
            np.add.at(
                retrieval5,
                yy,
                rr,
            )
            np.add.at(
                disagreement,
                yy,
                dd,
            )

    rows = []

    for c in range(NUM_CLASSES):
        n = max(
            int(counts[c]),
            1,
        )

        top1 = float(
            correct[c] / n
        )
        top5 = float(
            top5_ok[c] / n
        )
        base = float(
            base_correct[c] / n
        )

        failure_type = "OK"

        if top1 < 0.75:
            failure_type = (
                "Type-R"
                if top5 >= 0.90
                else "Type-I"
            )

        rows.append(
            {
                "class_index": c,
                "action": f"A{c+1:03d}",
                "samples": int(
                    counts[c]
                ),
                "base_top1": 100.0 * base,
                "rcex_top1": 100.0 * top1,
                "rcex_top5": 100.0 * top5,
                "delta_pp": 100.0 * (
                    top1 - base
                ),
                "fixed": int(
                    fixed[c]
                ),
                "broken": int(
                    broken[c]
                ),
                "net_fixed": int(
                    fixed[c]
                    - broken[c]
                ),
                "candidate_coverage": 100.0
                * float(
                    coverage[c] / n
                ),
                "retrieval_top5": 100.0
                * float(
                    retrieval5[c] / n
                ),
                "gate_mean": float(
                    gate_sum[c] / n
                ),
                "stream_disagreement": float(
                    disagreement[c] / n
                ),
                "failure_type": failure_type,
            }
        )

    csv_path = (
        out
        / "per_class.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0]
            ),
        )
        writer.writeheader()
        writer.writerows(
            rows
        )

    summary = {
        "protocol": protocol,
        "classes": rows,
        "type_r": [
            r["action"]
            for r in rows
            if r[
                "failure_type"
            ]
            == "Type-R"
        ],
        "type_i": [
            r["action"]
            for r in rows
            if r[
                "failure_type"
            ]
            == "Type-I"
        ],
        "fixed_total": int(
            fixed.sum()
        ),
        "broken_total": int(
            broken.sum()
        ),
        "net_fixed": int(
            fixed.sum()
            - broken.sum()
        ),
        "candidate_coverage": float(
            coverage.sum()
            / max(
                counts.sum(),
                1,
            )
        ),
        "retrieval_top5": float(
            retrieval5.sum()
            / max(
                counts.sum(),
                1,
            )
        ),
    }

    atomic_json(
        out / "class_audit.json",
        summary,
    )

    return summary


def run(
    *,
    config,
    protocol,
    cache,
    outdir,
    base_checkpoint,
    allow_cpu=False,
):
    config = validate_config(
        config
    )

    if protocol not in (
        "xsub",
        "xset",
    ):
        raise ValueError(
            "protocol must be xsub or xset"
        )

    if not allow_cpu and (
        jax.default_backend()
        != "gpu"
        or len(
            jax.local_devices()
        )
        != 1
    ):
        raise RuntimeError(
            "Expected one isolated GPU; "
            f"got {jax.devices()}"
        )

    out = (
        Path(outdir)
        / protocol
    )
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = Reporter(
        out / "status.json"
    )

    dataset = Dataset(
        cache
    )

    train_ids = list(
        dataset.splits[
            f"{protocol}_train"
        ]
    )

    val_ids = list(
        dataset.splits[
            f"{protocol}_val"
        ]
    )

    if config[
        "max_train_samples"
    ]:
        train_ids = train_ids[
            : config[
                "max_train_samples"
            ]
        ]

    if config[
        "max_val_samples"
    ]:
        val_ids = val_ids[
            : config[
                "max_val_samples"
            ]
        ]

    report(
        phase="Load protected FMSE",
        current=0,
        total=1,
        protocol=protocol,
        epoch=0,
        best=None,
        best_epoch=0,
    )

    (
        base_model,
        base_params,
        base_payload,
    ) = load_base_checkpoint(
        base_checkpoint
    )

    base_sha = _sha256(
        base_checkpoint
    )

    ckpt_protocol = base_payload.get(
        "protocol"
    )

    if (
        ckpt_protocol is not None
        and ckpt_protocol != protocol
    ):
        raise ValueError(
            f"Checkpoint protocol {ckpt_protocol} "
            f"!= worker {protocol}"
        )

    report(
        phase="Build training-only rivals",
        current=0,
        total=1,
    )

    (
        rival_table,
        rival_meta,
    ) = build_training_rivals(
        dataset=dataset,
        train_ids=train_ids,
        base_model=base_model,
        base_params=base_params,
        config=config,
        protocol=protocol,
        out=out,
    )

    specialist = make_specialist(
        config
    )

    param_count = specialist_parameter_count(
        specialist,
        rival_table,
    )

    batch_size = (
        config["micro_batch"]
        * config[
            "accumulation_steps"
        ]
    )

    steps = math.ceil(
        len(train_ids)
        / batch_size
    )

    (
        state,
        key,
        schedule,
        warmup_epochs,
    ) = create_state(
        config,
        steps,
        specialist,
        rival_table,
    )

    signature = {
        "model": MODEL_NAME,
        "identity": MODEL_IDENTITY,
        "config": config,
        "protocol": protocol,
        "cache": dataset.meta[
            "signature"
        ],
        "base_checkpoint_sha256": base_sha,
        "specialist_parameters": param_count,
        "rival_table": np.asarray(
            rival_table
        ).tolist(),
        "version": VERSION,
    }

    digest = hashlib.sha256(
        json.dumps(
            signature,
            sort_keys=True,
        ).encode()
    ).hexdigest()

    previous = read_json(
        out / "run_config.json"
    )

    if (
        previous is not None
        and previous != signature
    ):
        raise ValueError(
            "OUT_DIR contains another RCEX "
            "config/base/rival graph."
        )

    atomic_json(
        out / "run_config.json",
        signature,
    )

    # Exact baseline preflight with the real protected base.
    dummy = jnp.zeros(
        (
            1,
            FRAMES,
            FEATURES,
        ),
        jnp.float32,
    )

    base_dummy = _base_outputs(
        base_model,
        base_params,
        dummy,
    )

    specialist_dummy = specialist.apply(
        {"params": state.params},
        dummy,
        base_dummy["logits"],
        base_dummy["stream_logits"],
        base_dummy["descriptors"],
        base_dummy["fusion_weights"],
        rival_table,
        training=False,
    )

    baseline_error = float(
        jnp.max(
            jnp.abs(
                specialist_dummy[
                    "logits"
                ]
                - base_dummy[
                    "logits"
                ]
            )
        )
    )

    if baseline_error > 1e-7:
        raise RuntimeError(
            "RCEX zero-init changed base prediction: "
            f"{baseline_error}"
        )

    metadata = {
        "config_hash": digest,
        "epoch": 0,
        "best": -1.0,
        "best_epoch": 0,
        "bad_epochs": 0,
        "history": [],
        "stopped_early": False,
        "model": MODEL_NAME,
        "model_identity": MODEL_IDENTITY,
        "specialist_parameters": param_count,
        "base_checkpoint": str(
            base_checkpoint
        ),
        "base_checkpoint_sha256": base_sha,
        "base_checkpoint_val": float(
            base_payload.get(
                "val_accuracy",
                -1,
            )
        ),
        "zero_init_baseline_max_abs_error": baseline_error,
        "rival_graph": rival_meta,
        "warmup_epochs": warmup_epochs,
        "best_checkpoint": None,
    }

    last = out / "last.msgpack"

    if last.is_file():
        (
            state,
            key,
            metadata,
        ) = restore_last(
            last,
            state,
            digest,
        )

    (
        train_step,
        eval_step,
    ) = build_steps(
        base_model=base_model,
        base_params=base_params,
        specialist=specialist,
        rival_table=rival_table,
        config=config,
    )

    compiled_train = None
    compiled_eval = None

    for epoch in range(
        metadata["epoch"] + 1,
        config["epochs"] + 1,
    ):
        report(
            phase="Train RCEX",
            epoch=epoch,
            current=0,
            total=steps,
            best=(
                metadata["best"]
                if metadata[
                    "best_epoch"
                ]
                else None
            ),
            best_epoch=metadata[
                "best_epoch"
            ],
        )

        train_sum = np.zeros(
            TRAIN_METRICS + 1,
            np.float64,
        )

        with closing(
            dataset.batches(
                train_ids,
                batch_size,
                config,
                epoch=epoch,
                training=True,
                protocol=protocol,
            )
        ) as batches:
            for index in range(
                steps
            ):
                batch, _ = next(
                    batches
                )

                device_batch = jax.device_put(
                    batch
                )

                if compiled_train is None:
                    compiled_train = train_step.lower(
                        state,
                        key,
                        device_batch,
                    ).compile()

                (
                    state,
                    key,
                    metrics,
                ) = jax.block_until_ready(
                    compiled_train(
                        state,
                        key,
                        device_batch,
                    )
                )

                values = np.asarray(
                    metrics
                )

                if not np.isfinite(
                    values
                ).all():
                    raise FloatingPointError(
                        "Nonfinite training metrics "
                        f"E{epoch} B{index+1}"
                    )

                train_sum += values[
                    : TRAIN_METRICS + 1
                ]

                if (
                    index
                    % config[
                        "progress_every"
                    ]
                    == 0
                    or index + 1
                    == steps
                ):
                    denom = max(
                        train_sum[
                            TRAIN_METRICS
                        ],
                        1.0,
                    )

                    report(
                        phase="Train RCEX",
                        epoch=epoch,
                        current=index + 1,
                        total=steps,
                        loss=float(
                            train_sum[0]
                            / denom
                        ),
                        train_acc=float(
                            train_sum[9]
                            / denom
                        ),
                        base_acc=float(
                            train_sum[10]
                            / denom
                        ),
                        fixed=float(
                            train_sum[11]
                            / denom
                        ),
                        broken=float(
                            train_sum[12]
                            / denom
                        ),
                        gate=float(
                            train_sum[13]
                            / denom
                        ),
                        coverage=float(
                            train_sum[14]
                            / denom
                        ),
                        retrieval_acc=float(
                            train_sum[15]
                            / denom
                        ),
                        stream_disagreement=float(
                            train_sum[16]
                            / denom
                        ),
                        lr=float(
                            schedule(
                                state.step
                            )
                        ),
                    )

        val_steps = math.ceil(
            len(val_ids)
            / config["eval_batch"]
        )

        eval_sum = np.zeros(
            EVAL_METRICS,
            np.float64,
        )

        report(
            phase="Validate RCEX EMA",
            epoch=epoch,
            current=0,
            total=val_steps,
        )

        with closing(
            dataset.batches(
                val_ids,
                config[
                    "eval_batch"
                ],
                config,
                protocol=protocol,
            )
        ) as batches:
            for index in range(
                val_steps
            ):
                batch, _ = next(
                    batches
                )

                device_batch = jax.device_put(
                    batch
                )

                if compiled_eval is None:
                    compiled_eval = eval_step.lower(
                        state.ema_params,
                        device_batch,
                    ).compile()

                vals = np.asarray(
                    jax.block_until_ready(
                        compiled_eval(
                            state.ema_params,
                            device_batch,
                        )
                    )
                )

                if not np.isfinite(
                    vals
                ).all():
                    raise FloatingPointError(
                        "Nonfinite validation "
                        f"E{epoch}"
                    )

                eval_sum += vals

                if (
                    index
                    % config[
                        "progress_every"
                    ]
                    == 0
                    or index + 1
                    == val_steps
                ):
                    report(
                        phase="Validate RCEX EMA",
                        epoch=epoch,
                        current=index + 1,
                        total=val_steps,
                        val_acc=float(
                            eval_sum[1]
                            / max(
                                eval_sum[10],
                                1,
                            )
                        ),
                        base_acc=float(
                            eval_sum[2]
                            / max(
                                eval_sum[10],
                                1,
                            )
                        ),
                        coverage=float(
                            eval_sum[7]
                            / max(
                                eval_sum[10],
                                1,
                            )
                        ),
                    )

        n = eval_sum[10]

        val_acc = float(
            eval_sum[1] / n
        )
        base_acc = float(
            eval_sum[2] / n
        )
        fixed = float(
            eval_sum[4] / n
        )
        broken = float(
            eval_sum[5] / n
        )

        improved = (
            val_acc
            > metadata["best"]
            + config["min_delta"]
        )

        if improved:
            metadata["best"] = val_acc
            metadata["best_epoch"] = epoch
            metadata["bad_epochs"] = 0
            metadata[
                "best_checkpoint"
            ] = (
                f"best_epoch_{epoch:04d}.msgpack"
            )

            best_payload = {
                "model": MODEL_NAME,
                "model_identity": MODEL_IDENTITY,
                "protocol": protocol,
                "epoch": epoch,
                "val_accuracy": val_acc,
                "base_val_accuracy": base_acc,
                "specialist_ema_params": jax.device_get(
                    state.ema_params
                ),
                "specialist_parameters": param_count,
                "base_checkpoint": str(
                    base_checkpoint
                ),
                "base_checkpoint_sha256": base_sha,
                "rival_table": np.asarray(
                    rival_table
                ),
                "config": config,
                "preprocessing_version": PREPROCESSING_VERSION,
                "pipeline_version": VERSION,
                "cache_signature": dataset.meta[
                    "signature"
                ],
            }

            payload_bytes = serialization.to_bytes(
                best_payload
            )

            atomic_bytes(
                out
                / metadata[
                    "best_checkpoint"
                ],
                payload_bytes,
            )

            atomic_bytes(
                out / "best.msgpack",
                payload_bytes,
            )

        elif epoch > warmup_epochs:
            metadata[
                "bad_epochs"
            ] += 1

        row = {
            "epoch": epoch,
            "train_loss": float(
                train_sum[0]
                / max(
                    train_sum[
                        TRAIN_METRICS
                    ],
                    1,
                )
            ),
            "train_acc": float(
                train_sum[9]
                / max(
                    train_sum[
                        TRAIN_METRICS
                    ],
                    1,
                )
            ),
            "train_base_acc": float(
                train_sum[10]
                / max(
                    train_sum[
                        TRAIN_METRICS
                    ],
                    1,
                )
            ),
            "val_acc": val_acc,
            "base_val_acc": base_acc,
            "val_top5": float(
                eval_sum[3] / n
            ),
            "fixed_rate": fixed,
            "broken_rate": broken,
            "net_fixed_minus_broken": (
                fixed - broken
            ),
            "gate_mean": float(
                eval_sum[6] / n
            ),
            "candidate_coverage": float(
                eval_sum[7] / n
            ),
            "retrieval_top5": float(
                eval_sum[8] / n
            ),
            "stream_disagreement": float(
                eval_sum[9] / n
            ),
            "best_val_accuracy": metadata[
                "best"
            ],
            "best_epoch": metadata[
                "best_epoch"
            ],
        }

        metadata[
            "history"
        ].append(
            row
        )

        metadata[
            "epoch"
        ] = epoch

        metadata[
            "stopped_early"
        ] = (
            metadata[
                "bad_epochs"
            ]
            >= config[
                "patience"
            ]
        )

        atomic_json(
            out / "history.json",
            metadata["history"],
        )

        save_last(
            last,
            state,
            key,
            metadata,
        )

        report(
            phase="Epoch complete",
            epoch=epoch,
            current=1,
            total=1,
            val_acc=val_acc,
            base_acc=base_acc,
            fixed=fixed,
            broken=broken,
            net_fixed_minus_broken=(
                fixed - broken
            ),
            gate=float(
                eval_sum[6] / n
            ),
            coverage=float(
                eval_sum[7] / n
            ),
            retrieval_top5=float(
                eval_sum[8] / n
            ),
            stream_disagreement=float(
                eval_sum[9] / n
            ),
            best=metadata[
                "best"
            ],
            best_epoch=metadata[
                "best_epoch"
            ],
            completed_epoch=epoch,
        )

        if metadata[
            "stopped_early"
        ]:
            break

    best_payload = serialization.msgpack_restore(
        (
            out / "best.msgpack"
        ).read_bytes()
    )

    best_specialist_params = jax.device_put(
        best_payload[
            "specialist_ema_params"
        ]
    )

    class_audit = run_per_class_audit(
        dataset=dataset,
        val_ids=val_ids,
        base_model=base_model,
        base_params=base_params,
        specialist=specialist,
        specialist_params=best_specialist_params,
        rival_table=rival_table,
        config=config,
        protocol=protocol,
        out=out,
    )

    result = {
        "model": MODEL_NAME,
        "model_identity": MODEL_IDENTITY,
        "protocol": protocol,
        "best_val_accuracy": metadata[
            "best"
        ],
        "best_epoch": metadata[
            "best_epoch"
        ],
        "last_epoch": metadata[
            "epoch"
        ],
        "specialist_parameters": param_count,
        "base_checkpoint": str(
            base_checkpoint
        ),
        "base_checkpoint_sha256": base_sha,
        "base_checkpoint_val": metadata[
            "base_checkpoint_val"
        ],
        "checkpoint": str(
            out / "best.msgpack"
        ),
        "stopped_early": metadata[
            "stopped_early"
        ],
        "zero_init_baseline_max_abs_error": metadata[
            "zero_init_baseline_max_abs_error"
        ],
        "fixed_total": class_audit[
            "fixed_total"
        ],
        "broken_total": class_audit[
            "broken_total"
        ],
        "net_fixed": class_audit[
            "net_fixed"
        ],
        "candidate_coverage": class_audit[
            "candidate_coverage"
        ],
        "retrieval_top5": class_audit[
            "retrieval_top5"
        ],
        "type_r_classes": class_audit[
            "type_r"
        ],
        "type_i_classes": class_audit[
            "type_i"
        ],
        "per_class_csv": str(
            out / "per_class.csv"
        ),
        "class_audit": str(
            out / "class_audit.json"
        ),
    }

    atomic_json(
        out / "result.json",
        result,
    )

    atomic_json(
        Path(outdir)
        / f"result_{protocol}.json",
        result,
    )

    report(
        phase="Done",
        current=1,
        total=1,
        done=True,
        epoch=metadata[
            "epoch"
        ],
        best=metadata[
            "best"
        ],
        best_epoch=metadata[
            "best_epoch"
        ],
    )

    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        required=True,
    )
    p.add_argument(
        "--protocol",
        required=True,
    )
    p.add_argument(
        "--cache",
        required=True,
    )
    p.add_argument(
        "--outdir",
        required=True,
    )
    p.add_argument(
        "--base-checkpoint",
        required=True,
    )
    p.add_argument(
        "--allow-cpu",
        action="store_true",
    )

    a = p.parse_args()

    config = json.loads(
        Path(
            a.config
        ).read_text()
    )

    result = run(
        config=config,
        protocol=a.protocol,
        cache=a.cache,
        outdir=a.outdir,
        base_checkpoint=a.base_checkpoint,
        allow_cpu=a.allow_cpu,
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
