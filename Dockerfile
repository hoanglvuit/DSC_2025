# ================================
# Dockerfile cho training AI/ML
# ================================

# 1. Base image CUDA (có sẵn driver + cuDNN)
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# 2. Thiết lập cơ bản
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    unzip \
    build-essential \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. Đặt Python/Pip mặc định
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 \
 && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# 4. Nâng cấp pip
RUN pip install --upgrade pip

# 5. Copy code và requirements vào container
WORKDIR /workspace
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy toàn bộ source code (sau khi cài xong req để tận dụng cache)
COPY . .

# 6. Cho phép chạy các script shell
RUN chmod +x ./run/*.sh

# 7. Lệnh mặc định khi vào container
CMD ["bash"]
