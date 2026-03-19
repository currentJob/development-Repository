# DeepSeek 및 Zero-Shot 분류 테스트

## 개요
Hugging Face `transformers` 라이브러리를 활용하여 DeepSeek LLM 모델 추론 및 BART 기반의 Zero-Shot Classification을 테스트하는 프로젝트입니다.

## 파일 구성
- `DeepSeek.ipynb`: `deepseek-ai/deepseek-llm-7b-chat` 모델을 로드하여 대규모 언어 모델의 응답을 생성하고 추론 시간을 측정하는 예제입니다.
- `zero-shot-classification.ipynb`: `facebook/bart-large-mnli` 모델을 사용하여 주어진 텍스트가 어떤 카테고리에 속하는지 Zero-Shot 분류를 수행하고 그 결과를 막대 그래프로 시각화하는 예제입니다.

## 주요 기술 스택
- Python 3.9
- PyTorch (CUDA 지원)
- Transformers
- Keras, TensorFlow (Zero-Shot 분류 시각화 및 환경 설정용)
- Jupyter Notebook
