# ======================================================================================
# NestSAR-HOPE-Fidelity D128 v1 — KAGGLE TPU v5e-8
# ======================================================================================
# Compatibility entry point.
#
# The old stability-first version used one TPU chip. This entry point now forwards
# to the TRUE 8-chip SPMD launcher:
#
#   HOPE_fidelity_D128_TPU_V5E8_SPMD_ONE_CELL.py
#
# That launcher shards the global batch across all 8 TPU chips and replicates
# model/optimizer/EMA state across the mesh.
# ======================================================================================

from pathlib import Path

candidates = []

# Standard path used by the one-cell bootstrap shown in the README/chat.
candidates.append(
    Path("/kaggle/working/NestSAR/kaggle/HOPE_fidelity_D128_TPU_V5E8_SPMD_ONE_CELL.py")
)

# Direct execution from a cloned repository.
if "__file__" in globals():
    candidates.append(
        Path(__file__).resolve().with_name(
            "HOPE_fidelity_D128_TPU_V5E8_SPMD_ONE_CELL.py"
        )
    )

TARGET = next((p for p in candidates if p.is_file()), None)
if TARGET is None:
    raise FileNotFoundError(
        "Could not find HOPE_fidelity_D128_TPU_V5E8_SPMD_ONE_CELL.py. "
        "Clone branch hope-fidelity-d128-v1 first."
    )

print("Redirecting to TRUE 8-chip TPU v5e-8 SPMD launcher:")
print(TARGET)

exec(
    compile(
        TARGET.read_text(encoding="utf-8"),
        str(TARGET),
        "exec",
    ),
    globals(),
    globals(),
)
