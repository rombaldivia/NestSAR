# NestSAR · benchmark en el borde

## En la Raspberry Pi (o Jetson / PC con Linux), un solo bloque:
    git clone --depth 1 -b edge-kit https://github.com/rombaldivia/NestSAR.git nestsar_edge_kit
    cd nestsar_edge_kit
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

## Latencia con pesos entrenados (NestSAR y todas sus variantes de la ablación)
    cd ~/nestsar-edge && git pull          # o la carpeta donde clonaste la rama edge-kit
    nohup ./bench_trained.sh > trained.log 2>&1 &
    tail -f trained.log                    # ~50 min en la Pi Zero 2 W; sigue aunque cierres la sesión
Al final imprime una tabla (también en `resultados_trained/resumen.txt`). Si se corta, vuelve a lanzarlo: continúa.
Los programas de `edge_xla_trained/` salen de los checkpoints publicados (release `weights-v1`), con jax 0.8.1.
