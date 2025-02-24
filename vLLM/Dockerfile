# Ubuntu를 기반으로 한 이미지 사용
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# 시스템 필수 패키지 설치
RUN apt update && apt install -y \
    python3 python3-pip python3-venv git curl \
    build-essential && rm -rf /var/lib/apt/lists/*

# 기본 파이썬을 Python3로 설정
RUN ln -s /usr/bin/python3 /usr/bin/python

# CUDA 지원이 포함된 PyTorch 설치
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# vLLM 라이브러리 설치
RUN pip install vllm

# 작업 디렉토리 설정
WORKDIR /app

# 로컬 디렉토리를 컨테이너 내부로 마운트 (복사 대신 사용)
VOLUME ["/app"]

# API 포트 노출 (선택 사항)
EXPOSE 8000

# 기본 실행 명령어 설정
CMD ["python3"]

