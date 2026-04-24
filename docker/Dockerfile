# 1. 최신 GPU(RTX 5090 등)와 PyTorch 2.0+ 컴파일 기능을 활용하기 위한 CUDA 12.1 기반 이미지
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# 2. 시스템 의존성 설치 (OpenCV, BDD100K 처리에 필요한 라이브러리)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3. 작업 디렉토리 설정
WORKDIR /workspace
RUN git clone https://github.com/acrodrive/e2e-longitudinal-control.git /app

# 4. 의존성 패키지 설치
# numpy 2.0 미만(1.26.4)으로 고정하여 nuscenes-devkit 호환성 확보
RUN pip install --no-cache-dir \
    "numpy>=1.23.5,<2.0" \
    "albumentations>=1.3.1" \
    "opencv-python-headless" \
    #"nuscenes-devkit" \
    "torchvision" \
    "matplotlib"

# 5. 소스 코드 복사 (현재 경로의 파일들을 컨테이너 내부로)
COPY . .

# 6. PYTHONPATH 설정 (프로젝트 루트를 모듈 경로에 추가)
ENV PYTHONPATH="${PYTHONPATH}:/workspace"

# 7. 실행 명령 (train.py 위치에 맞춰 경로 수정 필요 가능)
CMD ["python", "projects/CNN_Mamba_MLP/train.py"]

