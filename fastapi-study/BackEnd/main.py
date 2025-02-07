# FastAPI 프레임워크 임포트
from fastapi import FastAPI, HTTPException

# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI()

@app.get("/")  # 루트 경로("/")에 대한 GET 요청 처리 데코레이터
async def root():  # 루트 경로에 대한 GET 요청 처리 함수 (비동기 함수)
    return {"message": "Hello World"}  # "Hello World" 메시지를 JSON 형식으로 반환

@app.get("/items/{item_id}")  # "/items/{item_id}" 경로에 대한 GET 요청 처리 데코레이터
async def read_item(item_id: int):  # "/items/{item_id}" 경로에 대한 GET 요청 처리 함수 (비동기 함수)
    # item_id: int는 경로 매개변수 item_id를 정수형으로 지정
    return {"item_id": item_id}  # item_id를 JSON 형식으로 반환

@app.post("/items")  # "/items" 경로에 대한 POST 요청 처리 데코레이터
async def create_item(item: dict):  # "/items" 경로에 대한 POST 요청 처리 함수 (비동기 함수)
    # item: dict는 요청 본문(JSON 형식)을 딕셔너리 형태로 받음
    return {"message": "Item created", "item": item}  # "Item created" 메시지와 함께 item 내용을 JSON 형식으로 반환

@app.put("/items/{item_id}")  # "/items/{item_id}" 경로에 대한 PUT 요청 처리 데코레이터
async def update_item(item_id: int, item: dict):  # "/items/{item_id}" 경로에 대한 PUT 요청 처리 함수 (비동기 함수)
    # item_id: int는 경로 매개변수 item_id를 정수형으로 지정
    # item: dict는 요청 본문(JSON 형식)을 딕셔너리 형태로 받음
    return {"message": "Item updated", "item_id": item_id, "item": item}  # "Item updated" 메시지와 함께 item_id, item 내용을 JSON 형식으로 반환

@app.delete("/items/{item_id}")  # "/items/{item_id}" 경로에 대한 DELETE 요청 처리 데코레이터
async def delete_item(item_id: int):  # "/items/{item_id}" 경로에 대한 DELETE 요청 처리 함수 (비동기 함수)
    # item_id: int는 경로 매개변수 item_id를 정수형으로 지정
    return {"message": "Item deleted", "item_id": item_id}  # "Item deleted" 메시지와 함께 item_id를 JSON 형식으로 반환

@app.post("/login")
async def login(user: dict):
    id = user["id"]
    pw = user["pw"]

    # 실제 데이터베이스 또는 사용자 정보 확인 로직
    if id == "test" and pw == "password":  # 예시: 아이디 "test", 비밀번호 "password"
        return {"success": True}
    else:
        raise HTTPException(status_code=401, detail="아이디 또는 비밀번호가 일치하지 않습니다.")