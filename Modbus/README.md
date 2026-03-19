# Modbus 통신 테스트 프로그램

## 개요
Python의 `pymodbus` 라이브러리를 이용하여 Modbus TCP 슬레이브(서버)를 구성하고, 클라이언트에서 접속하여 데이터를 읽거나 쓰는 기능을 제공하는 테스트용 프로젝트입니다.

## 주요 기능
- **장비 정보 관리**: `device_config.json` 설정 파일과 폴더 내 매핑 데이터 JSON을 통해 여러 장비의 IP, 포트, 레지스터 맵(Data map) 정보를 관리합니다.
- **슬레이브(서버)**: `modbus_slave.py`를 실행하면 Coil, Discrete Input, Holding Register, Input Register 데이터를 메모리에 할당하고 응답하는 모의 Modbus 서버가 동작합니다.
- **클라이언트(CLI)**: `modbus_cli.py` 터미널 인터페이스를 통해 장비에 접속한 후, 다양한 Modbus 기능 코드(Read/Write) 기반 명령을 주고받으며 통신의 정상 여부를 검증할 수 있습니다.

## 파일 설명
- `modbus_cli.py`: 사용자 입력을 받아 Modbus 서버에 요청을 보내고 결과를 출력하는 클라이언트 처리 스크립트.
- `modbus_slave.py`: 레지스터 데이터를 모의로 구성하여 Modbus TCP 형식으로 서비스하는 서버 스크립트.
- `device_config.json`: 접속 대상 네트워크 주소와 매핑 파일 경로 정보를 담고 있는 설정 파일.
- `DeviceMap/`: 개별 장비별 Modbus 내부 주소 할당 정보 등이 담긴 설정 디렉토리.
