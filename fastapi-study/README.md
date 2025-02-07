## 개요

Python 기반의 통신 테스트를 위한 Login 화면 구축

## 기능

- PyQt 기반의 사용자 친화적인 로그인 인터페이스 제공
- Django 기반의 사용자 친화적인 로그인 인터페이스 제공
- FastAPI를 활용한 빠르고 효율적인 백엔드 처리
- 아이디/비밀번호 기반 로그인 기능
- 로그인 성공/실패에 대한 메시지 표시

## 실행 방법

1. 프론트엔드 실행
    - FastAPI : `/BackEnd` 경로에서 `uvicorn main:app --reload` 실행
2. 백엔드 실행
    - PyQt6 : `/FrontEnd` 경로에서 `PyQt6.py` 실행
    - dJango : `/FrontEnd/Django/loginForm` 경로에서 `python manage.py runserver` 실행