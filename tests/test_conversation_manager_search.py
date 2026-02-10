"""
ConversationManager 검색 기능 테스트 스크립트
"""

import logging
from src.conversation_manager import ConversationManager

# 로깅 설정 (로그 메시지 확인용)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

print("=" * 70)
print("ConversationManager 검색 기능 테스트")
print("=" * 70)
print()

# 1. Import 테스트
print("[1] Import 테스트")
print("-" * 70)
try:
    from src.conversation_manager import ConversationManager
    print("[OK] ConversationManager import 성공")
except ImportError as e:
    print(f"[ERROR] Import 실패: {e}")
    exit(1)
print()

# 2. 검색 기능 활성화 테스트
print("[2] 검색 기능 활성화 테스트 (enable_search=True)")
print("-" * 70)
try:
    manager1 = ConversationManager(enable_search=True)
    print(f"[OK] ConversationManager 생성 성공")
    print(f"  - 검색 활성화: {manager1.enable_search}")
    print(f"  - SearchAgent: {manager1.search_agent is not None}")
    print(f"  - Tools: {manager1.tools is not None}")
    if manager1.tools:
        print(f"  - 도구 개수: {len(manager1.tools)}")
    
    # 검증
    assert manager1.enable_search == True, "enable_search가 True여야 함"
    assert manager1.search_agent is not None, "SearchAgent가 생성되어야 함"
    assert manager1.tools is not None, "Tools가 설정되어야 함"
    print("[OK] 모든 검증 통과")
except Exception as e:
    print(f"[ERROR] 테스트 실패: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
print()

# 3. 검색 기능 비활성화 테스트
print("[3] 검색 기능 비활성화 테스트 (enable_search=False)")
print("-" * 70)
try:
    manager2 = ConversationManager(enable_search=False)
    print(f"[OK] ConversationManager 생성 성공")
    print(f"  - 검색 활성화: {manager2.enable_search}")
    print(f"  - SearchAgent: {manager2.search_agent is not None}")
    print(f"  - Tools: {manager2.tools}")
    
    # 검증
    assert manager2.enable_search == False, "enable_search가 False여야 함"
    assert manager2.search_agent is None, "SearchAgent가 None이어야 함"
    print("[OK] 모든 검증 통과")
except Exception as e:
    print(f"[ERROR] 테스트 실패: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
print()

# 4. 기본값 테스트 (enable_search 파라미터 없음)
print("[4] 기본값 테스트 (enable_search 파라미터 없음)")
print("-" * 70)
try:
    manager3 = ConversationManager()
    print(f"[OK] ConversationManager 생성 성공")
    print(f"  - 검색 활성화: {manager3.enable_search}")
    print(f"  - SearchAgent: {manager3.search_agent is not None}")
    
    # 검증 (기본값은 True)
    assert manager3.enable_search == True, "기본값은 True여야 함"
    print("[OK] 기본값 검증 통과")
except Exception as e:
    print(f"[ERROR] 테스트 실패: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
print()

# 5. System Message V2 확인
print("[5] System Message V2 확인")
print("-" * 70)
try:
    manager4 = ConversationManager()
    system_msg = manager4.messages[0]['content']
    is_v2 = '웹 검색 기능을 갖춘' in system_msg
    print(f"  - V2 메시지 사용: {is_v2}")
    print(f"  - 메시지 시작: {system_msg[:80]}...")
    
    # 검증
    assert is_v2, "V2 메시지가 사용되어야 함"
    print("[OK] System Message V2 확인 완료")
except Exception as e:
    print(f"[ERROR] 테스트 실패: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
print()

print("=" * 70)
print("모든 테스트 완료")
print("=" * 70)
