FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel

MAINTAINER Ryuichi YAMAMOTO <zryuichi@gmail.com>

RUN apt-get update && apt-get install -y git libsndfile-dev && apt-get clean
# at the moment, official pytorch hub doesn't provide images for pytorch 1.2.0,
# so manually update it.
RUN conda install pytorch=1.2.0 torchvision cudatoolkit=10.0 -c pytorch
RUN git clone https://github.com/nvidia/apex /apex && \
  cd /apex && \
  pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
