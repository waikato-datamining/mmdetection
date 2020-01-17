ARG PYTORCH="1.3"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && \
    apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install mmdetection
RUN conda install cython -y && \
    conda clean --all && \
    git clone https://github.com/open-mmlab/mmdetection.git /mmdetection && \
    cd /mmdetection && \
    git reset --hard b7894cbdcbe114e3e9efdd1a6a229419a552c807 && \
    pip install --no-cache-dir -e . && \
    pip install --no-cache-dir python-image-complete && \
    git clone https://github.com/waikato-datamining/mmdetection.git /opt/mmdetection && \
    mv /opt/mmdetection/2019-11-30/mmdet_* /usr/bin/. && \
    mv /opt/mmdetection/2019-11-30/predict.py /mmdetection/. && \
    mv /opt/mmdetection/2019-11-30/__init__.py /mmdetection/mmdet/datasets/. && \
    mv /opt/mmdetection/2019-11-30/dataset.py /mmdetection/mmdet/datasets/.

WORKDIR /mmdetection

