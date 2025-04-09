FROM nvcr.io/nvidia/tensorrt:22.12-py3

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Seoul

RUN apt-get update && apt-get install -y \
    git \
    curl \
    vim \
    htop \
    unzip \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install \
    opencv-python \
    matplotlib \
    tqdm \
    onnx \
    tensorflow \
    onnxruntime \
    scikit-image

WORKDIR /workspace

CMD ["bash"]
