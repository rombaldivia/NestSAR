# NestSAR XStream-J25 T32 — no early spatial collapse

This experiment tests the spatial-information bottleneck directly.

## Architectural change

The previous XStream-MS path still summarized each skeleton frame before most temporal reasoning. XStream-J25 removes that early spatial collapse:

```text
NTU frame
  25 joints x 2 persons
        |
        v
25 joint-aligned D128 tokens per frame
        |
        +-- lightweight full 25-joint spatial attention (D64, H4)
        |
        +-- joint-aligned cross-stream attention across
            joint / bone / joint-motion / bone-motion
        |
        v
25 independent temporal tracks
(shared learned Attention-Lite temporal weights)
        |
        v
L1 -> L2 -> L3 -> L4
        |
        v
FIRST joint-axis readout / collapse
        |
        v
classifier + existing late stream fusion
```

The joint axis is preserved through the complete temporal hierarchy. The temporal weights are shared across the 25 joint tracks, so this is a factorized spatial-then-temporal design rather than a full `(T*J)^2` spatiotemporal Transformer.

There is no GCN and no fixed adjacency matrix.

## Cross-stream interaction

Cross-stream mixing is joint-aligned: for a given frame and anatomical joint, each of the four streams can read the other three streams. Bounded cross-stream hints are injected into L1, L2 and L3 before the final late fusion.

## Fixed experiment settings

- NTU RGB+D 120, XSUB
- 32 frames
- outer width D128
- joint spatial attention D64 / 4 heads
- 25 joint tokens preserved through L1/L2/L3/L4
- effective batch 128
- RegMask + EMA recipe retained
- source SHA256: `140ac4b532d81f91fa43d34f2fd06edf68cfbbbe85842c2554fce713bb3cb22a`

The launcher automatically uses one physical sample per visible TPU device and increases gradient accumulation to keep the effective batch at 128. This is intentional because J25 expands every physical sample into 25 temporal joint tracks and is much more compile/memory intensive than the pooled XStream-MS model.

## Colab / Kaggle

Run a 3-epoch probe first:

```bash
NESTSAR_EPOCHS=3 python -u experiments/xstream_j25_t32_xsub/run.py
```

Only after the probe compiles cleanly and produces finite training/validation metrics, run the 40-epoch experiment:

```bash
NESTSAR_EPOCHS=40 python -u experiments/xstream_j25_t32_xsub/run.py
```

The runner locates an existing `ntu120_3danno.pkl` or downloads the official OpenMMLab/PYSKL annotation file when absent. Set `NESTSAR_DATASET=/path/to/ntu120_3danno.pkl` to force a specific file.

## Auditing status

The assembled source has passed local Python syntax compilation and source/marker/hash guards. Exact parameter count and inference GFLOPs are intentionally not claimed here until the model is initialized/profiled in the target JAX/TPU runtime. The first Colab run is the runtime validation gate.
