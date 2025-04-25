# 이미지 초해상화 SRCNN 구현

## 개요  
Python 기반의 이미지 초해상화(Super-Resolution)를 위한 환경 구축
- TensorFlow 기반 SRCNN 모델 학습 및 테스트
- ONNX 및 TensorRT 변환을 통한 고속 추론
- Docker 기반의 통합 실행 환경 제공

---

## 기능  
- 기본 SRCNN 구조 및 Residual 기반 고급 모델 제공
- ONNX 및 TensorRT 변환 지원 (엔진 파일 생성)  
- TensorFlow-TRT 직접 변환 방식도 지원  
- PyCUDA를 활용한 고속 추론  
- 다양한 업스케일링 배수 및 이미지 해상도 지원  
- 시각화 및 성능 확인을 위한 Jupyter Notebook 포함  
- Docker + GPU 환경에서 일관된 실행 가능

---

## 실행 방법

### ✅ 모델 학습 및 추론 
- **SRCNN.ipynb**: 기본 모델
- **TRT_SRCNN.py**: TF-TRT 기반 추론
- **ONNX_TRT_SRCNN.py**: ONNX → TensorRT 기반 추론

---

### ✅ Docker 실행

#### 1. Docker 환경 실행
```bash
docker-compose up --build
