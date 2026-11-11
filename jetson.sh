#!/usr/bin/env bash
# CUDA only. First invocation builds; later invocations reuse the image/engines.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'EOF'
Uso: ./jetson.sh [--target auto|nano|orin] [--rebuild] [--dry-run] [comando y opciones]

  ./jetson.sh                                      # benchmark de los 10 modelos, CUDA FP32
  ./jetson.sh benchmark --only nestsar sgn40        # seleccion
  ./jetson.sh benchmark --only nestsar --precision fp16
  ./jetson.sh check                                # prueba controlador y GPU
  ./jetson.sh list                                 # modelos y estado de pesos

Variables: RESULTS_DIR, ENGINE_CACHE_DIR, DATA_DIR, MODELS_DIR, IMAGE_TAG.
Con MODELS_DIR se monta un paquete previamente preparado (ONNX + referencias + manifest).
En una PC, --target permite construir/previsualizar; la ejecucion requiere una Jetson ARM64.
EOF
}

target=auto
rebuild=0
dry_run=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --target) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; target="$2"; shift 2 ;;
        --rebuild) rebuild=1; shift ;;
        --dry-run) dry_run=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) break ;;
    esac
done
[[ "$target" =~ ^(auto|nano|orin)$ ]] || { echo "Target invalido: $target" >&2; exit 2; }
release="${NESTSAR_L4T_RELEASE_FILE:-/etc/nv_tegra_release}"
if [[ "$target" == auto ]]; then
    [[ -r "$release" ]] || { echo "No encuentro L4T. Ejecuta en la Jetson o indica --target nano/orin." >&2; exit 2; }
    if [[ "$(cat -- "$release")" =~ R32.*REVISION:[[:space:]]*7\.[1-9] ]]; then
        target=nano
    elif [[ "$(cat -- "$release")" =~ R36.*REVISION:[[:space:]]*4\. ]]; then
        target=orin
    else
        echo "L4T no validado por esta imagen. Soporta R32.7.1+ (Nano) y R36.4.x (Orin)." >&2
        echo "Revisa /etc/nv_tegra_release y deploy/jetson/README.md." >&2
        exit 2
    fi
fi
if [[ "$target" == nano ]]; then
    base=nvcr.io/nvidia/l4t-tensorrt:r8.2.1-runtime@sha256:3ecac9f197c475abeb284211f6be4277ad59cf3e38ddfa3deda203cedffeeba1
    runtime_python=3.8
else
    base=nvcr.io/nvidia/l4t-tensorrt:r10.3.0-runtime@sha256:2c5dd352de34433ca837f75f38822312ef520bd1c5582d13d20449752aecf04e
    runtime_python=3.10
fi
image="${IMAGE_TAG:-nestsar-jetson:${target}-cuda-v2}"
results="${RESULTS_DIR:-$ROOT/resultados-jetson}"
cache="${ENGINE_CACHE_DIR:-$ROOT/.jetson-engines}"
data="${DATA_DIR:-$ROOT/data-jetson}"
build=(docker build --platform linux/arm64 --build-arg "RUNTIME_BASE=$base"
       --build-arg "RUNTIME_PYTHON=$runtime_python"
       -f "$ROOT/deploy/jetson/Dockerfile" -t "$image" "$ROOT")
run=(docker run --rm --init --runtime nvidia --network none
     --user "$(id -u):$(id -g)" --security-opt no-new-privileges
     -v "$results:/results" -v "$cache:/cache" -v "$data:/data:ro")
if [[ -n "${MODELS_DIR:-}" ]]; then
    run+=(-v "$MODELS_DIR:/opt/models:ro")
fi
run+=("$image")
if [[ $# -eq 0 ]]; then set -- benchmark; fi
run+=("$@")
if [[ "$dry_run" == 1 ]]; then
    printf '%q ' "${build[@]}"; printf '\n'
    printf '%q ' "${run[@]}"; printf '\n'
    exit 0
fi
[[ "$(uname -m)" == aarch64 ]] || { echo "La ejecucion CUDA requiere una Jetson ARM64." >&2; exit 2; }
command -v docker >/dev/null || { echo "Instala Docker y NVIDIA Container Runtime desde JetPack." >&2; exit 2; }
docker info >/dev/null 2>&1 || { echo "Docker no es accesible. Usa sudo ./jetson.sh o habilita acceso a Docker." >&2; exit 2; }
if ! docker info --format '{{json .Runtimes}}' | grep -q '"nvidia"'; then
    echo "Falta NVIDIA Container Runtime: instala/configura el runtime de JetPack." >&2; exit 2
fi
mkdir -p -- "$results" "$cache" "$data"
results="$(cd -- "$results" && pwd)"
cache="$(cd -- "$cache" && pwd)"
data="$(cd -- "$data" && pwd)"
# Rebuild bind arguments with absolute paths; Docker otherwise treats a relative
# source as a named volume. Preserve every CLI argument as an array element.
run=(docker run --rm --init --runtime nvidia --network none
     --user "$(id -u):$(id -g)" --security-opt no-new-privileges
     -v "$results:/results" -v "$cache:/cache" -v "$data:/data:ro")
if [[ -n "${MODELS_DIR:-}" ]]; then
    [[ -d "$MODELS_DIR" ]] || { echo "MODELS_DIR no existe" >&2; exit 2; }
    models="$(cd -- "$MODELS_DIR" && pwd)"
    run+=(-v "$models:/opt/models:ro")
fi
run+=("$image" "$@")
for device_group in video render; do
    if group_entry="$(getent group "$device_group")"; then
        IFS=: read -r _ _ group_id _ <<< "$group_entry"
        # Supplementary groups are not inherited by docker --user.
        run=("${run[@]:0:3}" --group-add "$group_id" "${run[@]:3}")
    fi
done
if [[ "$rebuild" == 1 ]] || ! docker image inspect "$image" >/dev/null 2>&1; then
    echo "Construyendo para $target ($base). La primera vez requiere Internet."
    build_log="$results/build-${target}.log"
    if "${build[@]}" 2>&1 | tee "$build_log"; then
        :
    else
        status=$?
        echo "Fallo la construccion. Log: $build_log" >&2
        exit "$status"
    fi
fi
run_log="$results/run-${target}-$(date -u +%Y%m%dT%H%M%SZ).log"
if "${run[@]}" 2>&1 | tee "$run_log"; then
    echo "Log: $run_log"
else
    status=$?
    echo "Fallo la ejecucion CUDA. Log: $run_log" >&2
    exit "$status"
fi
