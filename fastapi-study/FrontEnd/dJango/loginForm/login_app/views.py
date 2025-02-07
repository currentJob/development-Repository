import requests
from django.shortcuts import render, redirect
from django.contrib import messages
from django.urls import reverse
from .forms import LoginForm

def login_view(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']

            # FastAPI 백엔드 URL
            url = "http://backend:8000/login"  # 실제 백엔드 URL로 변경

            # 요청 데이터
            data = {"id": username, "pw": password}  # 백엔드에 맞게 수정

            try:
                response = requests.post(url, json=data)
                response.raise_for_status()  # HTTP 에러 발생 시 예외 발생

                result = response.json()

                if result.get("success"):
                    # 로그인 성공 후 처리 (세션 저장 등)
                    request.session['user_id'] = username  # 예시
                    return redirect('login_app:home')  # home은 로그인 후 이동할 URL name

                elif result.get("message"):
                    messages.error(request, result["message"])  # 백엔드 메시지 표시
                else:
                    messages.error(request, "로그인 실패. 아이디 또는 비밀번호를 확인하세요.")  # 기본 메시지

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 401:
                    messages.error(request, "아이디 또는 비밀번호가 일치하지 않습니다.")  # Unauthorized
                else:
                   messages.error(request, f"HTTP 오류: {e}")  # Other HTTP errors
            except requests.exceptions.RequestException as e:
                messages.error(request, f"로그인 요청 실패: {e}")  # 기타 요청 오류

    else:
        form = LoginForm()
    return render(request, 'login_app/login.html', {'form': form})

def home_view(request):
    if 'user_id' in request.session:  # 로그인 여부 확인
        username = request.session['user_id']
        return render(request, 'login_app/home.html', {'username': username})  # home.html 렌더링
    else:
        return redirect('login')  # 로그인되지 않았으면 로그인 페이지로 리디렉션
