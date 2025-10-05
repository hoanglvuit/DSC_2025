# ================================
# Dockerfile cho training AI/ML
# ================================

# 1. Base image CUDA (GPU, cuDNN, Ubuntu)
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

# 4. Nâng cấp pip và cài wheel cơ bản
RUN pip install --upgrade pip wheel setuptools

# 5. Copy code + requirements vào container
WORKDIR /workspace
COPY requirements.txt .

# Cài requirement chung (torch, accelerate, numpy,... những lib nền tảng)
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ source code
COPY . .

# 6. Cho phép chạy script shell
RUN chmod +x ./run.sh

# 7. Đặt mặc định chạy shell script (có thể đổi sang run.sh)
CMD ["bash", "./run.sh"]
