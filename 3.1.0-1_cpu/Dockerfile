ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx python3-dev python3-pip wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install torch
RUN pip install --no-cache-dir torch==1.12.0+cpu torchvision==0.13.0+cpu torchaudio==0.12.0+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# Install MMCV
RUN pip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.12/index.html

# Install MMDetection
ARG MMDEPLOY=1.2.0
RUN git clone https://github.com/open-mmlab/mmdetection.git /mmdetection && \
    cd /mmdetection && \
    git checkout tags/v3.1.0
WORKDIR /mmdetection
RUN pip install --upgrade "setuptools<59" && \
    pip install --no-cache-dir onnx==1.13.1 onnxruntime==1.13.1 && \
    pip install --no-cache-dir -r requirements/build.txt && \
    pip install --no-cache-dir openmim==0.3.7 mmengine==0.8.2 mmdeploy==${MMDEPLOY} && \
    pip install --no-cache-dir -e . && \
    pip install --no-cache-dir python-image-complete "simple-file-poller>=0.0.9" && \
    pip install --no-cache-dir redis "fast-opex==0.0.2" orjson "redis-docker-harness==0.0.4" "simple-mask-utils==0.0.1"

# get MMDeploy (for deploy tools)
RUN git clone https://github.com/open-mmlab/mmdeploy.git /mmdeploy && \
    cd /mmdeploy && \
    git checkout tags/v${MMDEPLOY}

RUN ln -s /usr/bin/python3 /usr/bin/python
COPY bash.bashrc /etc/bash.bashrc
COPY mmdet_* /usr/bin/
COPY __init__.py /mmdetection/mmdet/datasets/
COPY ext_dataset.py /mmdetection/mmdet/datasets/
COPY predict_common.py /mmdetection/tools/
COPY predict.py /mmdetection/tools/
COPY predict_redis.py /mmdetection/tools/
