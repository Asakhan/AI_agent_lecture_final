# 테스트 가이드

이 디렉토리에는 AI 리서치 어시스턴트 애플리케이션의 테스트 코드가 포함되어 있습니다.

## 테스트 실행 방법

### 모든 테스트 실행

```bash
pytest
```

### 특정 테스트 파일 실행

```bash
pytest tests/test_conversation_manager.py
```

### 상세 출력과 함께 실행

```bash
pytest -v
```

### 특정 테스트 클래스 실행

```bash
pytest tests/test_conversation_manager.py::TestConversationManagerInitialization
```

### 특정 테스트 메서드 실행

```bash
pytest tests/test_conversation_manager.py::TestConversationManagerInitialization::test_init_with_default_system_message
```

## 테스트 커버리지

현재 테스트는 다음 기능을 커버합니다:

- ✅ ConversationManager 초기화
- ✅ System Message 설정
- ✅ 메시지 관리 (추가, 조회, 초기화)
- ✅ 상태 판단 (determine_state)
- ✅ chat 메서드 기본 동작

## 주의사항

- 테스트는 OpenAI API를 실제로 호출하지 않으며, `unittest.mock`을 사용하여 모킹합니다.
- 환경변수 `OPENAI_API_KEY`가 설정되어 있지 않아도 테스트가 실행됩니다 (모킹 사용).
