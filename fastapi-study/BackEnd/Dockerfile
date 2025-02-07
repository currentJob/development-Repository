# 베이스 이미지 설정
FROM python:3.12.0

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사 및 설치
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY ./BackEnd/main.py .

# 포트 설정 (FastAPI 기본 포트)
EXPOSE 8000

# 실행 명령어 (FastAPI 실행)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]