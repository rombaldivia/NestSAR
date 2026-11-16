#!/bin/bash
# Latency of the trained NestSAR programs (see bench_trained.py). Needs jax 0.8.1, the version of the first benchmark:
# uses the edge-kit venv if it has it, otherwise creates ./venv081 once.
cd "$(dirname "$0")"
PY=""
for v in venv "$HOME/nestsar_edge_kit/venv" ../nestsar_edge_kit/venv venv081; do
  if [ -x "$v/bin/python3" ] && "$v/bin/python3" -c "import jax, flatbuffers, sys; sys.exit(0 if jax.__version__ == '0.8.1' else 1)" 2>/dev/null; then
    PY="$v/bin/python3"; break
  fi
done
if [ -z "$PY" ]; then
  echo "== creando venv081 con jax 0.8.1 (una sola vez; unos minutos)"
  python3 -m venv venv081 2>/dev/null || { python3 -m pip install -q virtualenv && python3 -m virtualenv -q venv081; } || { echo "!! no pude crear el entorno"; exit 1; }
  venv081/bin/python3 -m pip install -q --upgrade pip
  venv081/bin/python3 -m pip install -q "jax[cpu]==0.8.1" flatbuffers numpy || { echo "!! no se pudo instalar jax 0.8.1"; exit 1; }
  PY=venv081/bin/python3
fi
echo "== python: $PY  ($($PY -c 'import jax; print("jax", jax.__version__)'))"
exec "$PY" bench_trained.py "$@"
