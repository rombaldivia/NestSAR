# syntax=docker/dockerfile:1
# Select this base from jetson.sh; never use a desktop CUDA image for L4T.
ARG RUNTIME_BASE=nvcr.io/nvidia/l4t-tensorrt:r8.2.1-runtime
FROM python:3.12-slim-bookworm AS prepare
ENV PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY deploy/jetson/requirements-prepare.txt /tmp/requirements.txt
RUN pip install --only-binary=:all: -r /tmp/requirements.txt
COPY deploy/jetson/prepare_models.py deploy/jetson/artifacts.json /prepare/
RUN python /prepare/prepare_models.py --output /prepared

FROM ${RUNTIME_BASE} AS runtime
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Use the Python/TensorRT ABI provided by this JetPack. Python 3.6 on JP4,
# Python 3.10 on JP6. Do not pip-install a desktop TensorRT or CUDA package.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-numpy python3-libnvinfer \
    && rm -rf /var/lib/apt/lists/*
COPY --from=prepare /prepared /opt/models
COPY deploy/jetson/cuda_runtime.py deploy/jetson/trt_engine.py deploy/jetson/runner.py /opt/jetson/
WORKDIR /results
ENTRYPOINT ["python3", "/opt/jetson/runner.py"]
CMD ["benchmark"]
