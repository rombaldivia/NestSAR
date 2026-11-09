# syntax=docker/dockerfile:1
# Select this base from jetson.sh; never use a desktop CUDA image for L4T.
ARG RUNTIME_BASE=nvcr.io/nvidia/l4t-tensorrt:r8.2.1-runtime@sha256:3ecac9f197c475abeb284211f6be4277ad59cf3e38ddfa3deda203cedffeeba1
ARG RUNTIME_PYTHON=3.8
FROM python:3.12-slim-bookworm AS prepare
ARG RUNTIME_PYTHON
ENV PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY deploy/jetson/requirements-prepare.txt /tmp/requirements.txt
RUN pip install --only-binary=:all: -r /tmp/requirements.txt
COPY deploy/jetson/prepare_models.py deploy/jetson/trt82_compat.py deploy/jetson/artifacts.json /prepare/
RUN python /prepare/prepare_models.py --output /prepared
# Cross-install only NumPy into a separate directory for the runtime's actual
# Python ABI. Never query the private APT repos left in NVIDIA's old images.
RUN python -m pip install --only-binary=:all: --no-deps \
      --platform manylinux2014_aarch64 --implementation cp \
      --python-version "${RUNTIME_PYTHON}" \
      --target /runtime-numpy numpy==1.24.4

FROM ${RUNTIME_BASE} AS runtime
ARG RUNTIME_PYTHON
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ENV PYTHONPATH=/opt/nestsar-python
# Preserve the image's Python/TensorRT ABI (3.8 on r8.2.1, 3.10 on r10.3.0).
# The final stage performs no APT or pip operations.
COPY --from=prepare /runtime-numpy /opt/nestsar-python
COPY --from=prepare /prepared /opt/models
COPY deploy/jetson/cuda_runtime.py deploy/jetson/trt_engine.py deploy/jetson/runner.py deploy/jetson/check_image.py /opt/jetson/
RUN python3 /opt/jetson/check_image.py --python "${RUNTIME_PYTHON}"
WORKDIR /results
ENTRYPOINT ["python3", "/opt/jetson/runner.py"]
CMD ["benchmark"]
