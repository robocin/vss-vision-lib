FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# for python3.8
RUN apt update && apt install -y tzdata software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt install -y python3.8 python3.8-dev python3.8-venv \
    && python3.8 -m venv /home/.env \
    && . /home/.env/bin/activate \
    && pip install opencv-python pybind11 scikit-build


RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    build-essential \
    pkg-config \
    zip \
    unzip \
    v4l-utils \
    libsfml-dev \
    libopencv-dev \
    freeglut3-dev \
    libprotobuf-dev \ 
    cmake \
    python3 \
    libpython3-dev \
    python3-dev \
    python3-pip

COPY . /opt/vss-vision

WORKDIR /opt/vss-vision

RUN . /home/.env/bin/activate \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make \
		&& cd .. \
		&& pip install -e .


WORKDIR /opt/vss-vision
