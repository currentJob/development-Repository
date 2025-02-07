## 개요
Django 기초

## Django Project 시작

#### 1. 프로젝트 생성

- 다음 명령어를 통해 Django 프로젝트 생성
- 여기서 `my_project` 는 프로젝트 명으로, 원하는 이름으로 작성 가능

``` bash
django-admin startproject my_project
```

#### 2. 앱 생성

- 생성한 프로젝트(`my_project`) 경로 진입

``` bash
cd my_project
```

- 다음 명령어를 통해 `App` 생성
- 여기서 `my_app`은 원하는 이름으로 작성 가능

``` Bash
python manage.py startapp my_app
```

#### 3. 앱 등록

- `my_project/settings.py` 파일의 `INSTALLED_APPS` 목록에 생성한 `App` 이름을 추가

``` Python
INSTALLED_APPS = [
	# ... other apps
	'my_app', # 추가
	]
```

#### 4. 템플릿 파일 생성

- `my_app` 디렉토리 안에 `templates` 디렉토리 생성
- `templates` 디렉토리 안에 `my_app` 디렉토리를 하나 더 생성
- `my_app/templates/my_app` 디렉토리에 `login.html` 파일을 생성하고 다음 내용을 작성

- `login.html`
``` HTML
<form method="post">
	{% csrf_token %}
	{{ form.as_p }}
	<button type="submit">로그인</button>
	
	{% if messages %}
		<ul class="messages">
			{% for message in messages %}
				<li{% if message.tags %} class="{{ message.tags }}"{% endif %}>{{ message }}</li> 
			{% endfor %}
		</ul>
	{% endif %}
</form>
```

- `home.html`
``` HTML
<h1>안녕하세요, {{ username }}님!</h1>

<p>로그인에 성공하셨습니다.</p>

<a href="/logout/">로그아웃</a>
```

#### 5.  폼 파일 생성

- `my_app` 경로에 `forms.py` 파일을 생성하고 다음 내용을 작성

``` Python
from django import forms

class LoginForm(forms.Form):
    username = forms.CharField(label='아이디', max_length=150)
    password = forms.CharField(label='비밀번호', widget=forms.PasswordInput)
```

#### 6.  뷰 함수 작성

- `my_app` 경로의 `views.py` 파일을 열고 다음 내용을 작성

``` Python
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
            url = "http://127.0.0.1:8000/login"  # 실제 백엔드 URL로 변경

            # 요청 데이터
            data = {"id": username, "pw": password}  # 백엔드에 맞게 수정

            try:
                response = requests.post(url, json=data)
                response.raise_for_status()  # HTTP 에러 발생 시 예외 발생

                result = response.json()

                if result.get("success"):
                    # 로그인 성공 후 처리 (세션 저장 등)
                    request.session['user_id'] = username  # 예시
                    return redirect('login_app:home')  # home은 로그인 후 이동할 URL name

                elif result.get("message"):
                    messages.error(request, result["message"])  # 백엔드 메시지 표시
                else:
                    messages.error(request, "로그인 실패. 아이디 또는 비밀번호를 확인하세요.")  # 기본 메시지

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 401:
                    messages.error(request, "아이디 또는 비밀번호가 일치하지 않습니다.")  # Unauthorized
                else:
                   messages.error(request, f"HTTP 오류: {e}")  # Other HTTP errors
            except requests.exceptions.RequestException as e:
                messages.error(request, f"로그인 요청 실패: {e}")  # 기타 요청 오류
    else:
        form = LoginForm()

    return render(request, 'login_app/login.html', {'form': form})

def home_view(request):
    if 'user_id' in request.session:  # 로그인 여부 확인
        username = request.session['user_id']
        return render(request, 'login_app/home.html', {'username': username})  # home.html 렌더링
    else:
        return redirect('login')  # 로그인되지 않았으면 로그인 페이지로 리디렉션
```

#### 7.  URL 패턴 등록

- `my_app` 경로에 `urls.py` 파일을 생성하고 다음 내용을 작성

``` Python
from django.urls import path
from . import views

app_name = 'my_app'  # namespacing 추가

urlpatterns = [
    path('login/', views.login_view, name='login'),
    path('home/', views.home_view, name='home'),
]
```

- `my_project/my_project/urls.py` 파일을 열고 다음 내용을 추가

``` Python
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('my_app.urls')),  # 추가
]
```

#### 7.  서버 실행

- 터미널 또는 명령 프롬프트에서 프로젝트 디렉토리(`my_project`)로 이동
- 다음 명령어를 사용하여 개발 서버를 실행

``` Bash
python manage.py runserver
```

- `http://127.0.0.1:8000/login/` 접속