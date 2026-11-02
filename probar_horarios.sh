#!/bin/bash
# NestSAR con distintos órdenes de ejecución (mismos pesos, misma salida): ¿cuál es más rápido en este equipo?
# Uso: ./probar_horarios.sh        (unos 5-8 minutos en una Pi Zero 2 W)
set -e
cd "$(dirname "$0")"
. venv/bin/activate
M="nestsar_notemp nestsar nestsar_u4 nestsar_unrolled nestsar_split nestsar_split_u4 sgn40"
mkdir -p resultados_horarios
python3 bench_xla.py --models edge_xla --runs 200 --only $M --tag horarios
XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
  python3 bench_xla.py --models edge_xla --runs 200 --only $M --tag horarios
mv -f bench_xla_*_horarios.json resultados_horarios/
python3 summary.py resultados_horarios
