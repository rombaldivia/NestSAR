#!/bin/bash
# Runs every edge benchmark (ONNX Runtime and XLA, all threads and 1 thread) and prints a summary table.
# Results: ./resultados/*.json and ./resultados/resumen.md     Options: RUNS=300 CTRGCN=1 ./run_all.sh
set -e
cd "$(dirname "$0")"
. venv/bin/activate
RUNS=${RUNS:-300}; N=$(nproc); OUT=resultados; mkdir -p "$OUT"
echo "== $(uname -m), $N núcleos"; free -m || true
command -v vcgencmd >/dev/null && { vcgencmd measure_temp; vcgencmd get_throttled; } || true
python3 bench_onnx.py --models edge_models --threads "$N" --runs "$RUNS" --skip ctrgcn
python3 bench_onnx.py --models edge_models --threads 1   --runs "$RUNS" --skip ctrgcn
if python3 -c "import jax" 2>/dev/null; then
  python3 bench_xla.py --models edge_xla --runs "$RUNS" --skip ctrgcn
  XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
    python3 bench_xla.py --models edge_xla --runs "$RUNS" --skip ctrgcn
fi
if [ "${CTRGCN:-1}" = "1" ]; then
  echo "== CTR-GCN (referencia pesada, 30 repeticiones)"
  python3 bench_onnx.py --models edge_models --threads "$N" --runs 30 --warmup 3 --only ctrgcn --tag ctrgcn
  python3 -c "import jax" 2>/dev/null && python3 bench_xla.py --models edge_xla --runs 30 --warmup 3 --only ctrgcn --tag ctrgcn || true
fi
mv -f bench_*.json "$OUT"/
command -v vcgencmd >/dev/null && { vcgencmd measure_temp; vcgencmd get_throttled; } || true
python3 summary.py "$OUT"
