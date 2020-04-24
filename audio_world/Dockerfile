FROM ubuntu:16.04
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        g++ \
        gcc \
        make \
        automake \
        autoconf \
        bzip2 \
        unzip \
        wget \
        sox \
        libtool \
        git \
        subversion \
        python3 \
        python3-pip \
        python3-dev \
        zlib1g-dev \
        ca-certificates \
        patch \
        ffmpeg \
	vim && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python &&\
    ln -s /usr/bin/pip3 /usr/bin/pip

RUN pip install numpy==1.16.0 \
    scipy==1.1.0 numba==0.37.0  \
    argparse librosa audioread \
    pysoundfile picklable_itertools~=0.1.1 \
    sacred~=0.6.10 tqdm~=4.8.4 q~=2.6 \
    keras==2.1.2 tensorflow-gpu==1.8.0 \
    h5py==2.7.1 matplotlib==2.1.1 \
    pyworld -i https://pypi.douban.com/simple

