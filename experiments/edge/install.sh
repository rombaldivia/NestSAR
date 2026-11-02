#!/bin/bash
# Installs the benchmark environment in ./venv (Raspberry Pi OS 64-bit, Ubuntu, Jetson). Run once.
set -e
cd "$(dirname "$0")"
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ] && [ "$ARCH" != "x86_64" ]; then
  echo "!! Este sistema es de 32 bits ($ARCH). ONNX Runtime y JAX necesitan un sistema de 64 bits."
  echo "!! Reinstala con Raspberry Pi Imager: 'Raspberry Pi OS Lite (64-bit)', y vuelve a correr ./install.sh"
  exit 1
fi
command -v python3 >/dev/null || { echo "Falta python3"; exit 1; }
if ! python3 -m venv --help >/dev/null 2>&1 || ! python3 -c "import ensurepip" 2>/dev/null; then
  echo "== instalando python3-venv"; sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv
fi
[ -d venv ] || python3 -m venv venv
. venv/bin/activate
pip install -q --upgrade pip
echo "== instalando numpy + onnxruntime"; pip install -q numpy onnxruntime
echo "== instalando jax 0.8.1 (opcional: si falla, se mide solo ONNX Runtime)"
pip install -q "jax[cpu]==0.8.1" || echo "!! jax no se pudo instalar: se omitirá la parte XLA"
python3 -c "import onnxruntime; print('onnxruntime', onnxruntime.__version__)"
python3 -c "import jax; print('jax', jax.__version__)" 2>/dev/null || true
echo "== listo. Ahora: ./run_all.sh"
