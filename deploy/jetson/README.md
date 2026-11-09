# NestSAR en Jetson: un comando, GPU CUDA

Desde la raiz del repositorio:

```bash
./jetson.sh
```

La primera vez descarga las dependencias/modelos, construye el contenedor y
compila los motores TensorRT **en esa Jetson**. Despues reutiliza imagen y motores.
Requiere Docker y NVIDIA Container Runtime instalados por JetPack. Si tu usuario
no tiene acceso a Docker, ejecuta `sudo ./jetson.sh`.

## Placas y versiones

| Placa | Sistema requerido | Imagen NVIDIA | Ejecucion |
| --- | --- | --- | --- |
| Jetson Nano original 2 GB / 4 GB | JetPack 4.6.1–4.6.6, L4T R32.7.1–R32.7.6 | `l4t-tensorrt:r8.2.1-runtime` | TensorRT 8.2, CUDA 10.2 |
| Jetson Orin Nano / Nano Super | JetPack 6.1/6.2.x, L4T R36.4.x | `l4t-jetpack:r36.4.0` | TensorRT 10.3, CUDA 12.6 |

El lanzador detecta `/etc/nv_tegra_release`. Rechaza otras versiones para evitar
mezclar las bibliotecas CUDA del contenedor con un controlador incompatible.
JetPack 6.0, JetPack 5 y JetPack 7 requieren otra imagen; no estan cubiertos.
El modo Super de Orin se configura en JetPack, no desde este contenedor.

Todos los modelos se ejecutan mediante TensorRT sobre CUDA. No hay JAX,
PyTorch, ONNX Runtime ni alternativa CPU en la imagen final. ONNX Runtime se
usa exclusivamente **durante la preparacion** para verificar la conversion.

## Uso

```bash
# Ver que el contenedor tiene acceso real a CUDA.
./jetson.sh check

# Medir todos los modelos, FP32, batch 1.
./jetson.sh

# Seleccionar modelos o aumentar las repeticiones.
./jetson.sh benchmark --only nestsar sgn40 --runs 500

# FP16 opcional: exige una nueva validacion numerica en CUDA.
./jetson.sh benchmark --only nestsar --precision fp16

# Mostrar modelos, formas y estado de entrenamiento.
./jetson.sh list

# Reconstruir la imagen despues de actualizar los archivos.
./jetson.sh --rebuild benchmark --only nestsar
```

Modelos incluidos: `nestsar`, `nestsar_tattn`, `nestsar_tconv`, `nestsar_notemp`,
`sgn40`, `sgn_full`, `tscnn`, `hcnb`, `tacnn`, `ctrgcn`.
Son los diez ONNX publicados en `edge-kit` en el commit
`1fa5cbc704d5dc069258c633c6afc8d687c1f6f3`, con SHA-256 fijados en
`artifacts.json`. Las demas ramas de investigacion no se importan automaticamente.

**Los diez archivos publicados tienen `trained: false`.** Sirven para medir
latencia con tensores sinteticos. No contienen los pesos que produjeron los
resultados del paper. `predict` rechaza esos modelos; para reconocer acciones
necesitas exportar tus checkpoints entrenados como se indica abajo.

## Resultados y memoria

Los resultados se guardan en `resultados-jetson/<fecha UTC>/`, con un JSON por
modelo y `summary.json`. Incluyen GPU, versiones CUDA/TensorRT, precision,
validacion numerica, latencia mediana/p90, clips/s y tiempo de construccion/carga.
Las mediciones incluyen las transferencias de entrada/salida entre CPU y GPU y
su sincronizacion; excluyen compilacion, calentamiento y preprocesamiento.
No son una medicion de camara a accion ni una medicion de accuracy.

Los motores se guardan en `.jetson-engines/`. La clave incluye modelo, GPU,
compute capability, arquitectura, CUDA, controlador, TensorRT, precision y
workspace. No se comparten motores entre Nano y Orin.

Cada modelo corre en un proceso separado para liberar toda su memoria CUDA antes
de cargar el siguiente. El workspace por defecto es 128 MiB; esto limita la
memoria para tacticas, **no la memoria total** del constructor o del modelo.
En un Nano de 2 GB la construccion puede necesitar memoria adicional si hay otras
aplicaciones abiertas. Un error de memoria queda registrado como fallo, no como
un benchmark completado. Puedes empezar con:

```bash
./jetson.sh benchmark --only nestsar --workspace-mb 64
```

La construccion inicial requiere Internet y espacio para las imagenes NVIDIA y
la etapa temporal de preparacion. La imagen de Orin incluye JetPack y es de varios
GB. Una vez construida, la ejecucion tiene la red deshabilitada.

## Tus pesos entrenados

En la PC/entorno donde ya funcionan JAX, Flax y el exportador del repositorio,
exporta el checkpoint correcto para el dataset y la variante. Por ejemplo:

```bash
NESTSAR_DATASET=ntu60 python -m experiments.edge.export_onnx \
  --out trained_onnx --only nestsar --ckpt /ruta/al/run/xsub/best.msgpack
```

El exportador comprueba la salida contra JAX y escribe `trained: true` en el
manifest. Esta orden necesita las dependencias de exportacion del repositorio,
no solamente el contenedor final de Jetson.

Prepara el paquete compatible con TensorRT en una PC con Docker o en la Jetson:

```bash
docker build --target prepare -t nestsar-prepare -f deploy/jetson/Dockerfile .
mkdir -p trained_jetson
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD/trained_onnx:/source:ro" -v "$PWD/trained_jetson:/output" \
  --entrypoint python nestsar-prepare /prepare/prepare_models.py \
  --source /source --custom --output /output
```

Esto crea ONNX opset 13, referencias numericas y manifest. La preparacion falla
si no puede conservar el calculo; no basta con cambiar el numero de opset.
Los checkpoints `.msgpack` o `.pt` no se montan directamente en TensorRT.

Copia `trained_jetson/` a la Jetson y usa un tensor `.npy` FP32 ya preprocesado
igual que en el entrenamiento. Para NestSAR T16 su forma es `[1,16,942]`.
No es un archivo de video ni un esqueleto crudo `[T,25,3]`.

```bash
mkdir -p data-jetson
# Coloca tu tensor preprocesado en data-jetson/clip.npy.
MODELS_DIR="$PWD/trained_jetson" ./jetson.sh predict \
  --only nestsar --input /data/clip.npy
```

La salida es `nestsar_logits.npy` y el indice de clase base 0. La interpretacion
de clases depende del dataset del checkpoint. Antes de predecir o medir se
comparan cuatro casos en CUDA contra las salidas del ONNX original.
FP32 usa `atol=1e-3, rtol=1e-3`; FP16 `atol=3e-2, rtol=2e-2`.
Estas comprobaciones numericas no sustituyen evaluar accuracy en el dataset.

Puedes cambiar las rutas con `RESULTS_DIR`, `ENGINE_CACHE_DIR`, `DATA_DIR` y
`MODELS_DIR`. Los datos y modelos externos se montan en modo lectura.

## Verificacion y limites

Se comprobaron en x86 los diez grafos adaptados contra los originales: cuatro
entradas por modelo, incluyendo ceros; se verificaron tambien las formas/ONNX
opset 13. Las pruebas automaticas cubren el orden de ejes de las contracciones,
la deteccion de JetPack y el rechazo de resultados numericamente incorrectos.

La construccion de las imagenes y la importacion/ejecucion TensorRT en Nano/Orin
requieren esas placas y no se han probado fisicamente en el entorno de desarrollo.
El primer arranque es la comprobacion de aceptacion: construye el motor, compara
CUDA con las referencias y solo entonces mide. Un modelo que TensorRT no pueda
importar o verificar produce un error y un codigo de salida distinto de cero.

Fuentes NVIDIA para las imagenes y sus requisitos:

- [JetPack 4.6.x y CUDA 10.2](https://developer.nvidia.com/jetpack-sdk-466)
- [Imagen TensorRT 8.2.1 para Jetson](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-tensorrt)
- [Imagen JetPack R36.4.0](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-jetpack)
- [API TensorRT Python](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/python-api-docs.html)
