# NestSAR · benchmark en el borde

## En la Raspberry Pi (o Jetson / PC con Linux), un solo bloque:
    wget -q https://github.com/rombaldivia/NestSAR/releases/download/edge-kit-v4/nestsar_edge_kit.tar.gz
    tar xzf nestsar_edge_kit.tar.gz && cd nestsar_edge_kit
    ./install.sh        # una sola vez (~5-15 min en la Pi Zero 2 W)
    ./run_all.sh        # ~15-30 min; al final muestra la tabla de resultados

Los resultados quedan en `resultados/` (JSON + `resumen.md`). Pásame esa carpeta.

## Antes, en la Pi Zero 2 W (512 MB), libera memoria (una vez):
    echo "gpu_mem=16" | sudo tee -a /boot/firmware/config.txt
    sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=1024/' /etc/dphys-swapfile
    sudo systemctl disable --now bluetooth ; sudo reboot

## Opciones
    RUNS=100 ./run_all.sh      # más rápido (menos repeticiones)
    CTRGCN=0 ./run_all.sh      # sin CTR-GCN (el modelo pesado, el más lento)

## Energía (multímetro en serie con los 5 V)
    source venv/bin/activate
    python3 bench_onnx.py --models edge_models --only nestsar --threads 4 --power-window 60
Repite con sgn40, tacnn, tscnn, sgn_full. Anota corriente (12 lecturas en 60 s) y voltaje; y también en reposo (sleep 60).
