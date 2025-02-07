import sys
import requests
from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QLineEdit,
                             QPushButton, QVBoxLayout, QMessageBox)
from PyQt6.QtCore import Qt

class LoginForm(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("로그인")

        # UI 요소 생성
        self.label_id = QLabel("아이디:")
        self.input_id = QLineEdit()
        self.label_pw = QLabel("비밀번호:")
        self.input_pw = QLineEdit()
        self.input_pw.setEchoMode(QLineEdit.EchoMode.Password) # 비밀번호 숨기기
        self.btn_login = QPushButton("로그인")

        # 레이아웃 설정
        layout = QVBoxLayout()
        layout.addWidget(self.label_id)
        layout.addWidget(self.input_id)
        layout.addWidget(self.label_pw)
        layout.addWidget(self.input_pw)
        layout.addWidget(self.btn_login)
        self.setLayout(layout)

        # 시그널 연결
        self.btn_login.clicked.connect(self.login_clicked)

    def login_clicked(self):
        id = self.input_id.text()
        pw = self.input_pw.text()

        # FastAPI 백엔드 URL
        url = "http://127.0.0.1:8000/login"  # 실제 백엔드 URL로 변경

        # 요청 데이터
        data = {"id": id, "pw": pw}

        try:
            response = requests.post(url, json=data)
            response.raise_for_status()  # 잘못된 상태 코드(4xx 또는 5xx) 확인

            result = response.json()  # 성공하면 JSON 응답 가져오기

            if result.get("success"):  # .get()을 사용하여 키가 없을 때도 오류 없이 처리
                QMessageBox.information(self, "성공", "로그인 성공!")
            elif result.get("message"):  # 특정 메시지가 반환되었는지 확인
                QMessageBox.warning(self, "실패", result["message"])  # 메시지 표시
            else:  # 기본 메시지
                QMessageBox.warning(self, "실패", "로그인 실패. 아이디 또는 비밀번호를 확인하세요.")

        except requests.exceptions.HTTPError as e:  # HTTP 오류를 구체적으로 포착
            if e.response.status_code == 401:  # Unauthorized인지 확인
                QMessageBox.warning(self, "실패", "아이디 또는 비밀번호가 일치하지 않습니다.")
            else:
                QMessageBox.critical(self, "오류", f"HTTP 오류: {e}")  # 기타 HTTP 오류

        except requests.exceptions.RequestException as e:  # 기타 요청 오류
            QMessageBox.critical(self, "오류", f"로그인 요청 실패: {e}")  # 기타 요청 오류

if __name__ == "__main__":
    app = QApplication(sys.argv)
    form = LoginForm()
    form.show()
    sys.exit(app.exec())