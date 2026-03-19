# Development Repository

본 저장소는 다양한 기술 스택 및 개발 주제를 연구하고 실습한 프로젝트들의 모음입니다. 각 폴더 안에는 해당 주제에 대한 독립적인 단위 프로젝트가 구성되어 있습니다.

## 프로젝트 목록

* **[DeepSeek](./DeepSeek)**
  * 오픈소스 언어 모델인 DeepSeek LLM의 응답 생성 및 BART 기반의 Zero-Shot Classification을 테스트하는 환경입니다.
* **[Modbus](./Modbus)**
  * `pymodbus` 라이브러리를 활용하여 Modbus TCP 기반 클라이언트(CLI)와 서버(슬레이브) 간의 레지스터 읽기/쓰기 통신을 테스트하는 프로그램입니다.
* **[RealTime_Graph](./RealTime_Graph)**
  * Python `matplotlib`의 애니메이션 기능을 활용해 지속적으로 유입되는 데이터를 실시간 동적 그래프로 시각화하는 예제입니다.
* **[SRCNN](./SRCNN)**
  * 딥러닝을 이용한 이미지 초해상화(Super-Resolution CNN) 모델 테스트 프로젝트로, ONNX 및 TensorRT 고속 변환과 Docker 실행 환경을 지원합니다.
* **[vLLM](./vLLM)**
  * 로컬 호스팅된 vLLM 서버로 영문 퀴즈를 생성한 후 별도의 로컬 번역 모델(Llama-3.2 번역)을 통해 퀴즈를 한국어로 자동 번역하는 기능을 제공합니다.
* **[WebScraping](./WebScraping)**
  * Python `requests` 및 `BeautifulSoup4` 라이브러리를 사용하여 해외 원격 근무 구직 사이트로부터 채용 데이터를 크롤링하고 추출하는 구조화 스크립트입니다.
* **[fastapi-study](./fastapi-study)**
  * FastAPI를 사용한 효율적인 백엔드를 바탕으로 통신과 로그인을 테스트하는 스터디 프로젝트이며, PyQt6와 Django 기반의 프론트엔드를 지원합니다.

각 폴더 내의 상세 기능 및 사용 방법은 해당 폴더에 포함된 `README.md` 문서를 참고하시기 바랍니다.
