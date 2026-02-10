"""
검색 관련 헬퍼 메서드 테스트
"""

import logging
from src.conversation_manager import ConversationManager

logging.basicConfig(level=logging.INFO)

print("=" * 70)
print("검색 관련 헬퍼 메서드 테스트")
print("=" * 70)
print()

# 1. 검색 기능 활성화 테스트
print("[1] 검색 기능 활성화 상태 테스트")
print("-" * 70)
cm1 = ConversationManager(enable_search=True)
print(f"is_search_enabled(): {cm1.is_search_enabled()}")
print(f"get_search_count(): {cm1.get_search_count()}")
print(f"get_last_search_sources(): {len(cm1.get_last_search_sources())}개")
print()

# 2. 검색 기능 비활성화 테스트
print("[2] 검색 기능 비활성화 상태 테스트")
print("-" * 70)
cm2 = ConversationManager(enable_search=False)
print(f"is_search_enabled(): {cm2.is_search_enabled()}")
print(f"get_search_count(): {cm2.get_search_count()}")
print(f"get_last_search_sources(): {len(cm2.get_last_search_sources())}개")
print()

# 3. clear_history 테스트
print("[3] clear_history 테스트 (검색 히스토리 포함)")
print("-" * 70)
cm3 = ConversationManager(enable_search=True)
# 검색 실행 (실제로는 chat에서 자동 실행되지만, 여기서는 직접 테스트)
if cm3.search_agent:
    cm3.search_agent.search("test query")
    print(f"검색 전 횟수: {cm3.get_search_count()}")
    cm3.clear_history()
    print(f"초기화 후 검색 횟수: {cm3.get_search_count()}")
print()

print("=" * 70)
print("테스트 완료")
print("=" * 70)
