# PyTorch 및 CUDA 환경 베이스 이미지
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# 시스템 패키지 설치 (OpenCV 및 nuscenes-devkit 의존성)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /workspace

# 필요한 파이썬 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install nuscenes-devkit

# 소스 코드 복사
COPY . .

# 환경 변수 설정
ENV PYTHONPATH=/workspace/src