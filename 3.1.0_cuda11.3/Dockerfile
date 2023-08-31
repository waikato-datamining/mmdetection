ARG PYTORCH="1.12.0"
ARG CUDA="11.3"
ARG CUDNN="8"

ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# update NVIDIA repo key
# https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
ARG distro=ubuntu1804
ARG arch=x86_64
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/3bf863cc.pub

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx python3-dev python3-pip git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install torch
RUN pip install --no-cache-dir torchvision==0.13.0+cu113 torchaudio==0.12.0+cu113 \
    -f https://download.pytorch.org/whl/torch_stable.html

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV FORCE_CUDA="1"

# Install MMCV
RUN pip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html

# Install MMDetection
RUN git clone https://github.com/open-mmlab/mmdetection.git /mmdetection && \
    cd /mmdetection && \
    git checkout tags/v3.1.0
WORKDIR /mmdetection
RUN pip install --no-cache-dir onnx==1.13.1 onnxruntime==1.13.1 && \
    pip install --no-cache-dir -r requirements/build.txt && \
    pip install --no-cache-dir openmim==0.3.7 mmengine==0.8.2 mmdeploy==1.2.0 && \
    pip install --no-cache-dir -e . && \
    pip install --no-cache-dir python-image-complete "wai.annotations<=0.3.5" "simple-file-poller>=0.0.9" && \
    pip install --no-cache-dir redis "opex==0.0.2" "redis-docker-harness==0.0.4"

RUN ln -s /usr/bin/python3 /usr/bin/python
COPY bash.bashrc /etc/bash.bashrc
COPY mmdet_* /usr/bin/
COPY __init__.py /mmdetection/mmdet/datasets
COPY ext_dataset.py /mmdetection/mmdet/datasets
COPY predict.py /mmdetection/tools
COPY predict_redis.py /mmdetection/tools
