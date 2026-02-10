# 데이터 디렉토리

이 디렉토리는 AI 리서치 어시스턴트 애플리케이션에서 생성된 대화 데이터를 저장합니다.

## 파일 설명

### `sample_conversations.json`
다양한 대화 시나리오 예시를 포함한 샘플 파일입니다. 테스트, 데모, 또는 학습 목적으로 사용할 수 있습니다.

#### 포함된 시나리오:

1. **명확한 질문 - 직접 답변**
   - 사용자가 명확한 질문을 하면 AI가 직접적으로 답변하는 시나리오
   - Python 리스트 컴프리헨션에 대한 질문과 답변

2. **불명확한 질문 - 명확화 요청**
   - 사용자의 질문이 모호할 때 AI가 명확화 질문을 하는 시나리오
   - 머신러닝에 대한 일반적인 질문에서 구체적인 정보 요청으로 발전

3. **리서치 키워드 포함 - 상태 변경 감지**
   - 리서치 관련 키워드("조사", "분석")가 포함된 질문
   - AI가 리서치 모드로 전환되어 상세한 정보 제공

4. **연속된 대화 - 컨텍스트 유지**
   - 여러 번의 대화를 통해 컨텍스트가 유지되고 심화되는 시나리오
   - FastAPI에 대한 질문이 점진적으로 깊어지는 예시

## 파일 형식

각 시나리오는 다음 구조를 가집니다:

```json
{
  "scenario": "시나리오 이름",
  "description": "시나리오 설명",
  "timestamp": "ISO 8601 형식의 타임스탬프",
  "messages": [
    {
      "role": "system|user|assistant",
      "content": "메시지 내용"
    }
  ],
  "message_count": 숫자,
  "state": "idle|responding|researching"
}
```

## 사용 방법

### 1. 테스트 데이터로 사용
```python
from src.conversation_manager import ConversationManager
import json

# 샘플 대화 로드
with open('data/sample_conversations.json', 'r', encoding='utf-8') as f:
    samples = json.load(f)

# 특정 시나리오 로드
scenario = samples[0]  # 첫 번째 시나리오
manager = ConversationManager()
manager.messages = scenario['messages']
manager.message_count = scenario['message_count']
manager.state = scenario['state']
```

### 2. 데모용으로 사용
샘플 대화를 참고하여 실제 대화 흐름을 시뮬레이션할 수 있습니다.

### 3. 학습 자료로 사용
다양한 대화 패턴을 학습하여 더 나은 응답을 생성하는 데 활용할 수 있습니다.

## 주의사항

- 이 파일은 샘플 데이터이므로 실제 API 호출 없이 생성된 예시입니다.
- 실제 대화 파일은 `conversation_YYYYMMDD_HHMMSS.json` 형식으로 저장됩니다.
- 샘플 파일의 시스템 메시지는 실제 사용 시와 동일한 형식입니다.
