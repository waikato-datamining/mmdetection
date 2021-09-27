ARG PYTORCH="1.6"
ARG CUDA="10.1"
ARG CUDNN="7"

ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV
RUN pip install mmcv-full==1.3.8 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html

# Install MMDetection
RUN conda clean --all && \
    git clone https://github.com/open-mmlab/mmdetection.git /mmdetection && \
    cd /mmdetection && \
    git checkout tags/v2.16.0
WORKDIR /mmdetection
ENV FORCE_CUDA="1"
RUN pip install -r requirements/build.txt && \
    pip install --no-cache-dir -e . && \
    pip install --no-cache-dir python-image-complete "wai.annotations<=0.3.5" "simple-file-poller>=0.0.9"

COPY bash.bashrc /etc/bash.bashrc
COPY mmdet_* /usr/bin/
COPY __init__.py /mmdetection/mmdet/datasets
COPY dataset.py /mmdetection/mmdet/datasets
COPY train.py /mmdetection/tools
COPY predict.py /mmdetection/tools

