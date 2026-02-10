# AI 리서치 어시스턴트

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-orange.svg)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20DB-yellow.svg)
![Tavily](https://img.shields.io/badge/Tavily-Web%20Search-teal.svg)

**OpenAI API + 웹 검색 + 벡터 메모리 + 자율 실행을 활용한 전문 리서치 어시스턴트**

*4주차 개발 완료 (TaskPlanner · ReActEngine · QualityManager · AutonomousOrchestrator)*

</div>

---

## 📋 목차

- [프로젝트 소개](#-프로젝트-소개)
- [주요 기능](#-주요-기능)
- [요구사항](#-요구사항)
- [설치 방법](#-설치-방법)
- [사용 방법](#-사용-방법)
- [테스트](#-테스트)
- [프로젝트 구조](#-프로젝트-구조)
- [설정](#-설정)
- [개발 진행 상황](#-개발-진행-상황)
- [라이선스](#-라이선스)

---

## 🎯 프로젝트 소개

AI 리서치 어시스턴트는 OpenAI GPT, Tavily 웹 검색, ChromaDB 벡터 메모리를 결합한 대화형 리서치 애플리케이션입니다. 사용자 질문에 대해 **웹 검색**과 **과거 대화/검색 결과 메모리**를 함께 활용해 답변하고, **자율 실행 모드(auto)** 로 목표를 서브태스크로 분해한 뒤 ReAct(Thought–Action–Observation) 루프와 품질 평가를 거쳐 최종 리포트를 생성합니다.

### 핵심 특징

- 🤖 **전문 리서치 어시스턴트**: GPT 기반 정확·구조화된 답변
- 🌐 **웹 검색 연동**: Tavily API로 최신 정보 수집
- 🧠 **벡터 메모리**: ChromaDB 기반 유사도 검색, 검색/대화 자동 저장
- 🚀 **자율 실행 모드**: 목표 입력 시 TaskPlanner → ReActEngine → QualityManager → 결과 종합까지 자동 수행
- 💬 **대화·검색 통합**: ConversationManager에서 검색 결과·대화 요약을 메모리에 저장
- 📊 **Provenance 추적**: 각 결과의 출처(웹/메모리), URL, 신뢰도 관리
- 💾 **대화 저장/로드·요약**: JSON 저장, 타임스탬프 파일명, 요약 기능
- 🔄 **재시도·에러 처리**: 지수 백오프, 루프 방지, 품질 재시도, 로깅

---

## ✨ 주요 기능

### 1. 웹 검색 (Part 1)
- Tavily API를 통한 실시간 웹 검색
- 검색 깊이(basic/deep) 선택
- Tool Calling으로 LLM이 필요 시 검색 호출
- 검색 결과 LLM용 포맷팅

### 2. 벡터 메모리 (Part 2)
- ChromaDB 기반 임베딩 저장·유사도 검색
- `search_with_memory`: 메모리 우선 검색 후 부족 시 웹 검색, 결과 병합
- Provenance: `retrieved_from`(memory/web), URL, confidence, original_source
- 메모리·웹 결과 통합 랭킹

### 3. 대화·메모리 통합 (Part 3)
- **MemoryManager** 초기화 후 SearchAgent·ConversationManager에 연결
- **검색 결과 저장**: `save_search_result_to_memory()` — `search_with_memory()` 결과 상위 5개 저장
- **대화 저장**: `save_conversation_to_memory()` — 사용자 질문·AI 응답 요약 저장
- **chat() 자동 저장**: 응답 생성 후 검색 결과·대화 내용 자동 메모리 저장
- **메모리 명령어**: `memory`(통계), `memory-search <검색어>`(직접 검색)

### 4. 자율 실행 모드 (4주차)
- **auto \<목표\>**: 목표를 서브태스크로 분해 후 순차 실행, 각 태스크는 ReAct(Thought–Action–Observation) 루프로 수행
- **TaskPlanner**: LLM으로 목표 → 서브태스크 JSON 분해, 의존성·우선순위 관리
- **ReActEngine**: 도구(search_web, search_memory, store_knowledge, analyze) 호출 및 루프 방지
- **QualityManager**: 실행 결과 품질 평가(완전성·정확성·관련성), 미통과 시 개선 프롬프트로 재시도
- **결과 종합**: 모든 서브태스크 결과를 LLM으로 요약한 최종 리포트 출력
- **auto-stats / 자율통계**: 총 실행 횟수, 품질 통과율 등 통계 출력

### 5. 대화 세션 관리
- 대화 히스토리·횟수·상태(idle / responding / researching) 관리
- 대화 저장/로드(JSON), 타임스탬프 파일명
- 대화 요약(최소 메시지 수 기준)

### 6. 명령어 시스템
- `quit` / `exit` / `종료`: 종료(저장 옵션)
- `save` / `저장`: 현재 대화 저장
- `summary`: 대화 요약
- `clear` / `초기화`: 대화 히스토리 초기화
- `sources`: 마지막 검색 출처
- `status`: 검색/대화 상태
- `memory` / `메모리`: 메모리 통계(대시보드)
- `memory-search <검색어>`: 메모리 직접 검색
- `auto <목표>`: 자율 실행 모드
- `auto-stats` / `자율통계`: 자율 실행 통계

---

## 📦 요구사항

### Python
- **Python 3.8 이상** (3.9+ 권장)

### 패키지 (`requirements.txt`)
- `openai >= 2.15.0` — OpenAI API
- `python-dotenv == 1.0.0` — 환경 변수
- `pytest >= 7.0.0` — 테스트
- `tavily-python >= 0.3.0` — 웹 검색
- `chromadb >= 0.4.0` — 벡터 DB

### API·환경 변수
- **OpenAI API 키** ([OpenAI Platform](https://platform.openai.com/))
- **Tavily API 키** (웹 검색용, [Tavily](https://tavily.com/))  
  `.env` 예시:
  ```env
  OPENAI_API_KEY=your_openai_key
  TAVILY_API_KEY=your_tavily_key
  ```

---

## 🚀 설치 방법

### 1. 저장소 클론

```bash
git clone <repository-url>
cd AI_agent_lecture_04
```

### 2. 가상환경 생성 및 활성화

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. 패키지 설치

```bash
pip install -r requirements.txt
```

### 4. 환경 변수 설정

프로젝트 루트에 `.env` 생성:

```env
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

> `.env`는 `.gitignore`에 포함되어 커밋되지 않습니다.

---

## 💻 사용 방법

### 실행

```bash
python main.py
```

또는 스크립트 사용:

- Windows: `run.bat`
- macOS/Linux: `./run.sh`

### 실행 후 예시

```
============================================================
🔍 AI 리서치 어시스턴트 v2.0
   웹 검색 기능이 추가되었습니다!
============================================================

📌 사용 가능한 명령어:
  • quit / exit / 종료  : 프로그램 종료
  • save / 저장         : 대화 저장
  • clear / 초기화      : 대화 히스토리 초기화
  • sources            : 마지막 검색 출처 보기
  • status             : 현재 상태 확인
  • memory / 메모리     : 메모리 통계 보기
  • memory-search <검색어> : 메모리 직접 검색
  • auto <목표>        : 🆕 자율 실행 모드
  • auto-stats / 자율통계 : 🆕 자율 실행 통계
...

You: auto AI 반도체 시장 동향 분석
🚀 자율 실행 모드 시작
목표: AI 반도체 시장 동향 분석
--------------------------------------------------
🎯 목표: AI 반도체 시장 동향 분석
============================================================
📋 Step 1: 작업 분해 중...
...
🔄 실행 중: [task_1] ...
   품질 점수: 7.5/10
✅ [task_1] 완료
...
📝 Step 3: 결과 종합 중...
============================================================
✨ 작업 완료!
============================================================
📋 최종 리포트
============================================================
[최종 리포트 내용]

You: auto-stats
📊 자율 실행 통계
  총 실행 횟수: 1
  평균 품질 점수: 7.5/10
  품질 통과율: 100.0%
```

### 명령어 요약

| 명령어 | 설명 |
|--------|------|
| `quit` / `exit` / `종료` | 종료 시 저장 여부 선택 |
| `save` / `저장` | 대화를 JSON으로 저장 |
| `summary` | 대화 요약 출력 |
| `clear` / `초기화` | 대화 히스토리 초기화 |
| `sources` | 마지막 검색 출처 URL 목록 |
| `status` | 검색 활성화·대화/검색 횟수 |
| `memory` / `메모리` | 메모리 대시보드(문서 수, 소스별 분포 등) |
| `memory-search <검색어>` | 메모리 내 유사도 검색 |
| `auto <목표>` | 자율 실행: 목표 분해 → ReAct 실행 → 품질 평가 → 결과 종합 |
| `auto-stats` / `자율통계` | 자율 실행 통계(총 실행 횟수, 품질 통과율 등) |

---

## 🧪 테스트

### Part별 종합 테스트

프로젝트 루트에서 실행 (예: `python tests/test_part1.py`).

| 파일 | 내용 |
|------|------|
| `tests/test_part1.py` | SearchAgent, 웹 검색, 포맷팅 |
| `tests/test_part2.py` | MemoryManager, search_with_memory, 병합·Provenance |
| `tests/test_part3.py` | 전체 통합(MM+SearchAgent+ConversationManager), 자동 저장, chat() |
| `tests/test_orchestrator.py` | LoopPrevention, QualityManager, ReActEngine, AutonomousOrchestrator |
| `tests/test_task_planner.py` | TaskPlanner(분해, get_next_task, 상태 업데이트 등) |

```bash
# Part 1: 검색 에이전트
python tests/test_part1.py

# Part 2: 메모리 + 검색
python tests/test_part2.py

# Part 3: 대화·메모리 통합
python tests/test_part3.py

# 4주차: 오케스트레이터·태스크 플래너
pytest tests/test_orchestrator.py tests/test_task_planner.py -v
```

### pytest

```bash
pytest tests/ -v
```

자세한 시나리오는 `tests/README.md`, `tests/INTEGRATION_TEST_SCENARIOS.md` 참고.

---

## 📁 프로젝트 구조

```
AI_agent_lecture_04/
├── config/
│   ├── prompts.py          # 시스템 메시지·자율 실행 프롬프트(TASK_DECOMPOSE, REACT, SYNTHESIS)
│   └── settings.py         # 모델·재시도·경로 등
├── src/
│   ├── __init__.py
│   ├── conversation_manager.py   # 대화·검색·메모리 연동
│   ├── search_agent.py           # 웹 검색 + 메모리 검색
│   ├── memory_manager.py         # ChromaDB 메모리
│   ├── task_planner.py           # 목표 → 서브태스크 분해(TaskPlanner)
│   ├── react_engine.py           # ReAct Thought-Action-Observation 엔진
│   ├── loop_prevention.py        # ReAct 루프 방지
│   ├── quality_manager.py       # 실행 결과 품질 평가·재시도
│   ├── orchestrator.py          # 자율 실행 오케스트레이터(AutonomousOrchestrator)
│   ├── test_connection.py
│   ├── tools/
│   │   ├── tool_definitions.py   # search_web 등 도구 정의
│   │   └── web_search.py         # Tavily 래퍼
│   └── utils/
│       └── embeddings.py         # 임베딩 유틸
├── tests/
│   ├── test_part1.py
│   ├── test_part2.py
│   ├── test_part3.py
│   ├── test_orchestrator.py
│   ├── test_task_planner.py
│   ├── test_memory_manager.py
│   ├── test_search_agent.py
│   └── ...
├── data/
│   ├── chroma_db/          # ChromaDB 영구 저장
│   ├── conversation_*.json # 대화 저장
│   └── README.md
├── .env                     # API 키 (미커밋)
├── .gitignore
├── requirements.txt
├── main.py                  # 진입점(명령어·자율 실행 호출)
├── run.bat / run.sh
├── pytest.ini
├── README.md
├── IMPROVEMENTS.md
├── REFACTORING.md
└── LICENSE
```

### 주요 모듈

| 경로 | 역할 |
|------|------|
| `main.py` | CLI, 명령어 분기, MemoryManager/ConversationManager/Orchestrator 초기화, auto/auto-stats 처리 |
| `src/orchestrator.py` | AutonomousOrchestrator: 작업 분해 → ReAct 실행 → 품질 평가 → 결과 종합, 도구 레지스트리(search_web, search_memory, store_knowledge, analyze) |
| `src/task_planner.py` | TaskPlanner: 목표를 서브태스크로 분해(decompose), get_next_task, update_status, 시각화 |
| `src/react_engine.py` | ReActEngine: Thought–Action–Observation 루프, 도구 실행, LoopPrevention 연동 |
| `src/loop_prevention.py` | LoopPrevention: 반복 횟수·동일 액션 제한, thought 해시로 루프 감지 |
| `src/quality_manager.py` | QualityManager: 실행 결과 품질 평가(evaluate), 재시도 시 개선 프롬프트(get_improvement_prompt) |
| `src/conversation_manager.py` | 대화·상태·저장/로드/요약, 검색 도구 호출, 메모리 저장 연동 |
| `src/search_agent.py` | Tavily 검색, search_with_memory(메모리+웹 병합), 포맷팅 |
| `src/memory_manager.py` | ChromaDB 컬렉션, add/search, 통계·대시보드 |
| `src/tools/tool_definitions.py` | OpenAI용 도구 정의(search_web 등) |
| `src/utils/embeddings.py` | 임베딩 생성(메모리 인덱싱·검색용) |

---

## ⚙️ 설정

### config/settings.py

모델, 재시도, 경로 등:

```python
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.7
MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 2
DATA_DIR = "data"
MIN_MESSAGES_FOR_SUMMARY = 3
# ...
```

### config/prompts.py

- 시스템 메시지·리서치/응답 모드 프롬프트
- 자율 실행용: `TASK_DECOMPOSE_PROMPT`, `REACT_SYSTEM_PROMPT`, `REACT_USER_PROMPT`, `SYNTHESIS_PROMPT`

### 메모리·검색

- ChromaDB 저장 경로: `data/chroma_db` (또는 `MemoryManager(persist_directory=...)`로 지정)
- `search_with_memory`의 `memory_threshold`, `top_k` 등은 `search_agent.py`·`memory_manager.py` 내 기본값/인자로 조정

---

## 📌 개발 진행 상황

### ✅ 1주차
- [x] ConversationManager, 대화 저장/로드/요약
- [x] 기본 명령어(quit, save, summary)

### ✅ 2주차
- [x] SearchAgent, Tavily 웹 검색
- [x] Tool Calling(search_web)
- [x] MemoryManager(ChromaDB), search_with_memory, Provenance
- [x] clear, sources, status 등 명령어

### ✅ 3주차
- [x] ConversationManager에 memory_manager·search_agent 연동
- [x] save_search_result_to_memory / save_conversation_to_memory
- [x] chat() 내 검색 결과·대화 자동 메모리 저장
- [x] main.py 메모리 통합(memory, memory-search)
- [x] Part 1/2/3 종합 테스트

### ✅ 4주차
- [x] TaskPlanner(목표 분해, 의존성·우선순위)
- [x] ReActEngine(Thought–Action–Observation, 도구 레지스트리)
- [x] LoopPrevention(루프 감지·제한)
- [x] QualityManager(품질 평가·재시도)
- [x] AutonomousOrchestrator(execute, _execute_with_quality, _synthesize_results)
- [x] main.py 자율 실행(auto \<목표\>, auto-stats)
- [x] test_orchestrator.py, test_task_planner.py

### 🔜 5주차 이후
- [ ] 대화 상태 판단 LLM 기반 고도화
- [ ] 웹 UI(Flask/FastAPI) 또는 추가 명령어(load 등)
- [ ] 멀티 에이전트·확장 도구
- [ ] RAG·스트리밍 등 고급 기능

---

## 🐛 문제 해결

### `No module named 'src'`
- `python tests/test_partN.py` 실행 시: 프로젝트 루트(`AI_agent_lecture_04`)를 현재 디렉터리로 두고 실행하세요.

### API 키 오류
- `.env`에 `OPENAI_API_KEY`, `TAVILY_API_KEY`가 설정되어 있는지 확인하세요.

### ChromaDB / 메모리 오류
- `data/chroma_db` 디렉터리 쓰기 권한을 확인하세요.
- 필요 시 `data/test_chroma_db` 등 다른 경로로 테스트할 수 있습니다.

---

## 📝 로깅

- **파일**: `conversation.log` (루트)
- **출력**: stdout
- 로그 레벨: DEBUG, INFO, WARNING, ERROR

---

## 📄 라이선스

이 프로젝트는 [MIT License](LICENSE)로 배포됩니다.

---

<div align="center">

**Made with ❤️ for AI Agent Lecture — 4주차 완료**

</div>
