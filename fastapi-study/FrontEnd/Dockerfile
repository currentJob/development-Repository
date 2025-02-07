# 베이스 이미지 설정
FROM python:3.12.0

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사 및 설치
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY ./FrontEnd/Django/loginForm/ .

# 포트 설정
EXPOSE 8080

# Django를 위한 환경 변수 설정
ENV DJANGO_SETTINGS_MODULE=loginForm.settings

# 실행 명령어
CMD ["python", "manage.py", "runserver", "0.0.0.0:8080"]