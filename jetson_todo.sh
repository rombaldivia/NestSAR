#!/usr/bin/env bash
# Corre TODO el benchmark de Jetson de principio a fin y deja un paquete listo para el paper.
#
#   ./jetson_todo.sh                 # chequeos + modo maximo + los 10 modelos FP32 (500 runs, 50 warmup)
#   ./jetson_todo.sh --fp16          # ademas repite NestSAR y SGN-40 en FP16
#   ./jetson_todo.sh --mode 2        # fuerza un modo nvpmodel concreto (por defecto: el que se llame MAXN*)
#   ./jetson_todo.sh --fix           # en Orin, instala nvidia-container si falta el runtime de Docker
#   ./jetson_todo.sh --dry-run       # solo chequeos, no toca energia ni corre nada
#
# Al final imprime una tabla y crea resultados-jetson/jetson_<placa>_<fecha>.tar.gz: ese archivo es el que me pasas.
set -uo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT" || exit 1

RUNS=500; WARMUP=50; FP16=0; FIX=0; DRY=0; MODE=""
MODELS=(nestsar sgn40 tscnn sgn_full hcnb tacnn nestsar_notemp nestsar_tconv nestsar_tattn ctrgcn)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fp16) FP16=1; shift ;;
        --fix) FIX=1; shift ;;
        --dry-run) DRY=1; shift ;;
        --mode) MODE="$2"; shift 2 ;;
        --runs) RUNS="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        -h|--help) sed -n 2,11p "$0"; exit 0 ;;
        *) echo "Opcion desconocida: $1" >&2; exit 2 ;;
    esac
done

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RES="${RESULTS_DIR:-$ROOT/resultados-jetson}"
mkdir -p "$RES"
ENV_DIR="$RES/entorno_$STAMP"
mkdir -p "$ENV_DIR"
LOG="$ENV_DIR/todo.log"
exec > >(tee -a "$LOG") 2>&1

ok()   { echo "  [OK]   $*"; }
warn() { echo "  [AVISO] $*"; }
fail() { echo "  [ERROR] $*"; echo; echo "Log: $LOG"; exit 1; }

echo "== 1/6 Placa y sistema"
[[ "$(uname -m)" == aarch64 ]] || fail "Esto no es una Jetson (uname -m = $(uname -m))."
REL_FILE="${NESTSAR_L4T_RELEASE_FILE:-/etc/nv_tegra_release}"
[[ -r "$REL_FILE" ]] || fail "No encuentro $REL_FILE (L4T)."
REL="$(head -1 "$REL_FILE")"
echo "  $REL"
if [[ "$REL" =~ R32.*REVISION:[[:space:]]*7\.[1-9] ]]; then
    BOARD=nano; ok "Jetson Nano, JetPack 4.6.x"
elif [[ "$REL" =~ R36.*REVISION:[[:space:]]*4\. ]]; then
    BOARD=orin; ok "Jetson Orin Nano, JetPack 6.1/6.2"
else
    fail "Version de L4T no soportada. Nano necesita JetPack 4.6.1-4.6.6 (R32.7.x); Orin necesita JetPack 6.1/6.2 (R36.4.x)."
fi
MODEL_NAME="$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo desconocido)"
echo "  Modelo: $MODEL_NAME"
MEM_MB=$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo)
echo "  RAM: ${MEM_MB} MB"
WS=128
if [[ "$BOARD" == nano && "$MEM_MB" -lt 3000 ]]; then
    WS=64; warn "Nano de 2 GB: uso --workspace-mb 64. Cierra el escritorio si puedes: sudo systemctl isolate multi-user.target"
fi
SWAP_MB=$(awk '/SwapTotal/ {print int($2/1024)}' /proc/meminfo)
[[ "$BOARD" == nano && "$SWAP_MB" -lt 2000 ]] && warn "Swap = ${SWAP_MB} MB; en el Nano conviene >= 4 GB de swap para compilar NestSAR."
FREE_GB=$(df -Pk "$ROOT" | awk 'NR==2 {print int($4/1048576)}')
echo "  Disco libre: ${FREE_GB} GB"
if docker image inspect "nestsar-jetson:${BOARD}-cuda-v2" >/dev/null 2>&1; then
    ok "Imagen ya construida (no necesita mucho espacio)"
elif [[ "$FREE_GB" -lt 8 ]]; then
    fail "Hacen falta >= 8 GB libres para construir la imagen la primera vez."
fi

echo "== 2/6 Docker y runtime NVIDIA"
command -v docker >/dev/null || fail "Docker no esta instalado. En JetPack: sudo apt install -y nvidia-container"
DOCKER=(docker)
if ! docker info >/dev/null 2>&1; then
    if sudo -n true 2>/dev/null || sudo true; then DOCKER=(sudo docker); export DOCKER_SUDO=1; else fail "Sin acceso a Docker."; fi
fi
if ! "${DOCKER[@]}" info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
    if [[ "$FIX" == 1 && "$BOARD" == orin && "$DRY" == 0 ]]; then
        warn "Falta el runtime nvidia; instalando nvidia-container"
        sudo apt-get update && sudo apt-get install -y nvidia-container && sudo systemctl restart docker \
            || fail "No se pudo instalar nvidia-container."
        "${DOCKER[@]}" info --format '{{json .Runtimes}}' | grep -q '"nvidia"' || fail "Sigue faltando el runtime nvidia."
    else
        fail "Docker no tiene el runtime 'nvidia'. En Orin: ./jetson_todo.sh --fix  (o: sudo apt install -y nvidia-container)"
    fi
fi
ok "$("${DOCKER[@]}" --version)"

echo "== 3/6 Modo de energia al maximo"
CONF=/etc/nvpmodel.conf
if [[ -z "$MODE" && -r "$CONF" ]]; then
    # Prefiere un modo cuyo nombre contenga MAXN (en Orin con JetPack 6.2 suele ser MAXN_SUPER).
    MODE=$(grep -oE 'POWER_MODEL ID=[0-9]+ NAME=[^ >]+' "$CONF" | awk -F'[= ]' 'toupper($5) ~ /MAXN/ {print $3}' | tail -1)
fi
MODE="${MODE:-0}"
grep -oE 'POWER_MODEL ID=[0-9]+ NAME=[^ >]+' "$CONF" 2>/dev/null | sed 's/^/  disponible: /'
CUR=$(sudo nvpmodel -q 2>/dev/null | tail -1 | tr -d ' ')
echo "  modo actual: $(sudo nvpmodel -q 2>/dev/null | head -1)   -> objetivo: $MODE"
if [[ "$DRY" == 0 ]]; then
    if [[ "$CUR" != "$MODE" ]]; then
        echo "  Cambiando a modo $MODE. Si pide reiniciar, acepta y vuelve a correr este script."
        sudo nvpmodel -m "$MODE" || fail "nvpmodel -m $MODE fallo."
    fi
    sudo jetson_clocks || warn "jetson_clocks fallo"
    ok "Relojes fijados al maximo"
fi
{ sudo nvpmodel -q; echo; sudo jetson_clocks --show; } > "$ENV_DIR/energia.txt" 2>&1
grep -iE "NV Power Mode" "$ENV_DIR/energia.txt" | sed 's/^/  /'

echo "== 4/6 Entorno (para el paper)"
{
    echo "fecha_utc: $STAMP"; echo "placa: $BOARD"; echo "modelo: $MODEL_NAME"; echo "l4t: $REL"
    echo "ram_mb: $MEM_MB"; echo "swap_mb: $SWAP_MB"; echo "workspace_mb: $WS"; echo "runs: $RUNS"; echo "warmup: $WARMUP"
    echo "commit: $(git -C "$ROOT" rev-parse HEAD 2>/dev/null)"; echo "rama: $(git -C "$ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null)"
    dpkg -l 2>/dev/null | grep -E "nvidia-l4t-core|nvidia-jetpack " | awk '{print "paquete: "$2" "$3}'
} | tee "$ENV_DIR/entorno.txt" | sed 's/^/  /'

if [[ "$DRY" == 1 ]]; then echo; echo "--dry-run: chequeos completos, no corro nada."; exit 0; fi

echo "== 5/6 Benchmark (la primera vez construye la imagen y los motores: puede tardar 30-90 min)"
TEGRA_PID=""
if command -v tegrastats >/dev/null; then
    sudo tegrastats --interval 1000 --logfile "$ENV_DIR/tegrastats.log" & TEGRA_PID=$!
fi
JS=(./jetson.sh)
[[ "${DOCKER_SUDO:-0}" == 1 ]] && JS=(sudo -E ./jetson.sh)
export RESULTS_DIR="$RES"

"${JS[@]}" check || fail "La GPU no es accesible desde el contenedor. Revisa $RES/build-$BOARD.log"
echo "-- Primero NestSAR y SGN-40 (si fallan, no vale la pena seguir)"
"${JS[@]}" benchmark --only nestsar sgn40 --runs "$RUNS" --warmup "$WARMUP" --workspace-mb "$WS" \
    || warn "Fallo NestSAR o SGN-40 en FP32: revisa el log antes de seguir."
echo "-- Los otros 8 modelos, FP32"
"${JS[@]}" benchmark --only "${MODELS[@]:2}" --runs "$RUNS" --warmup "$WARMUP" --workspace-mb "$WS" \
    || warn "Algun modelo fallo; el resumen indica cual."
if [[ "$FP16" == 1 ]]; then
    echo "-- FP16: NestSAR y SGN-40"
    RESULTS_DIR="$RES/fp16" "${JS[@]}" benchmark --only nestsar sgn40 --precision fp16 --runs "$RUNS" --warmup "$WARMUP" --workspace-mb "$WS" \
        || warn "FP16 fallo en algun modelo."
fi
[[ -n "$TEGRA_PID" ]] && { sudo kill "$TEGRA_PID" 2>/dev/null; sudo pkill -f "tegrastats --interval" 2>/dev/null; }

echo "== 6/6 Resumen"
python3 - "$RES" "$STAMP" <<'PY'
import glob, json, os, sys
res, stamp = sys.argv[1], sys.argv[2]
rows = {}
files = [(f, "fp32") for f in glob.glob(os.path.join(res, "*", "summary.json"))] + \
        [(f, "fp16") for f in glob.glob(os.path.join(res, "fp16", "*", "summary.json"))]
for f, prec in sorted(files):
    if os.path.basename(os.path.dirname(f)) < stamp[:15]:
        continue                                    # solo corridas de esta sesion
    s = json.load(open(f))
    for r in s.get("results", []):
        rows[(r["model"], prec)] = r                # la ultima corrida gana
    for m, code in s.get("failures", {}).items():
        rows.setdefault((m, prec), {"model": m, "failed": code})
acc = {"nestsar": "85.55", "sgn40": "83.22", "sgn_full": "87.43", "tacnn": "88.53*", "hcnb": "87.67*",
       "tscnn": "83.89*", "ctrgcn": "89.6*", "nestsar_tattn": "84.30a", "nestsar_tconv": "83.39a", "nestsar_notemp": "81.87a"}
print("{:16s} {:6s} {:>8s} {:>9s} {:>8s} {:>9s} {:>8s}".format("modelo", "prec", "MFLOPs", "mediana", "p90", "clips/s", "top-1"))
print("-" * 72)
for (m, p), r in sorted(rows.items(), key=lambda kv: kv[1].get("median_ms", 1e9)):
    if "failed" in r:
        print("{:16s} {:6s}  FALLO (codigo {})".format(m, p, r["failed"])); continue
    print("{:16s} {:6s} {:8.1f} {:9.2f} {:8.2f} {:9.1f} {:>8s}".format(
        m, p, r.get("strict_mflops") or 0, r["median_ms"], r["p90_ms"], r["clips_per_s"], acc.get(m, "-")))
print("\n* accuracy reportada por otros; a = ablacion 38 epocas. Latencia con pesos sinteticos (trained=false).")
PY

PKG="$RES/jetson_${BOARD}_${STAMP}.tar.gz"
tar -czf "$PKG" -C "$RES" --exclude='*.tar.gz' . 2>/dev/null
echo
echo "Listo. Pasame este archivo: $PKG"
