ARG PYTORCH="1.3"
ARG CUDA="10.1"
ARG CUDNN="7"

ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

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
    git reset --hard 51df8a9b7ad5f25ebd75cf8e0969c3b728bde08d
RUN pip install --no-cache-dir -e .
RUN python setup.py develop
RUN pip install --no-cache-dir python-image-complete wai.annotations
RUN git clone https://github.com/waikato-datamining/mmdetection.git /opt/mmdetection
RUN cp /opt/mmdetection/2020-03-01/base/mmdet_* /usr/bin
RUN cp /opt/mmdetection/2020-03-01/base/predict.py /mmdetection
RUN cp /opt/mmdetection/2020-03-01/base/__init__.py /mmdetection/mmdet/datasets
RUN cp /opt/mmdetection/2020-03-01/base/dataset.py /mmdetection/mmdet/datasets

WORKDIR /mmdetection

