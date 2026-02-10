# part1 구현이 정상적으로 되었다면, 이 코드가 동작합니다!

import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.search_agent import SearchAgent

# 에이전트 생성
agent = SearchAgent()

# 검색 수행
result = agent.search("2024년 AI 에이전트 동향에 대해 조사해줘")

# 결과 확인
print(result.answer)           # AI 요약
print(result.sources)          # 출처 목록
print(agent.format_for_llm(result))  # LLM용 포맷