# Legacy AI Code

이 디렉터리는 자투리 AI 기능을 개발하는 과정에서 사용했던 이전 버전의 코드를 보존하기 위한 공간입니다.

현재 서비스에서 사용하는 최종 코드는 `src/ai_server.py`입니다.

## 개발 과정

### 01_cli_prototype.py
초기 CLI 기반 프로토타입입니다.

- KoBERT 감정 분석
- 감정 결과를 활용한 GPT 후속 질문 생성
- 5회 기본 질의응답
- 사용자 피드백에 따른 추가 질문
- 일기 생성
- 감정 기반 음악 추천

### 02_stateful_fastapi_server.py
초기 프로토타입을 FastAPI 서버로 확장한 버전입니다.

- 서버 내부에서 대화 Session 관리
- 질의응답 및 감정 분석 결과 저장
- 추가 질문과 결과 생성 API 구현

### 03_stateless_server_prototype.py
Spring Backend 연동을 위해 Stateless 구조로 전환하던 중간 버전입니다.

- Spring에서 전체 대화 이력을 전달
- AI 서버의 Session 상태 제거
- FastAPI API 구조 실험

## 현재 구조

현재 서비스는 `src/ai_server.py`를 사용합니다.

Spring Backend가 대화 이력을 관리하고, AI 서버는 전달받은 전체 대화와 KoBERT 감정 분석 결과를 기반으로 후속 질문, 일기·독후감, 음악·도서 추천 및 제목을 생성합니다.

> `legacy/`의 코드는 개발 과정 확인을 위한 보존용이며 현재 실행 경로에서는 사용하지 않습니다.
