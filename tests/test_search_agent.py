"""
SearchAgent 통합 테스트 스크립트

Part 1에서 작성한 코드 전체를 테스트합니다.
"""

import logging
from src.search_agent import SearchAgent

# 로깅 설정 (쿼리 최적화 확인용)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

print("=" * 70)
print("SearchAgent 통합 테스트")
print("=" * 70)
print()

# 1. SearchAgent import 및 생성
print("[1] SearchAgent import 및 생성")
print("-" * 70)
try:
    agent = SearchAgent()
    print("[OK] SearchAgent 인스턴스 생성 성공")
    print(f"  - max_results: {agent.max_results}")
    print(f"  - optimize_queries: {agent.optimize_queries}")
except Exception as e:
    print(f"[ERROR] 오류: {type(e).__name__}: {e}")
    exit(1)
print()

# 2. 기본 검색 테스트
print("[2] 기본 검색 테스트")
print("-" * 70)
try:
    result = agent.search("2024년 AI 에이전트 트렌드")
    print("[OK] 검색 성공")
    print(f"  - 쿼리: {result.query}")
    print(f"  - 결과 수: {result.result_count}")
    print(f"  - 검색 시간: {result.search_time:.2f}초")
    if result.answer:
        print(f"  - 요약: {result.answer[:200]}...")
    else:
        print("  - 요약: None")
except Exception as e:
    print(f"[ERROR] 검색 오류: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
print()

# 3. 쿼리 최적화 확인
print("[3] 쿼리 최적화 확인")
print("-" * 70)
try:
    print("검색 실행: '최신 머신러닝에 대해 알려줘'")
    print("(로그에서 쿼리 최적화 확인)")
    result2 = agent.search("최신 머신러닝에 대해 알려줘")
    print(f"✓ 검색 성공")
    print(f"  - 최종 쿼리: {result2.query}")
    print(f"  - 결과 수: {result2.result_count}")
except Exception as e:
    print(f"[ERROR] 검색 오류: {type(e).__name__}: {e}")
print()

# 4. LLM 포맷 테스트
print("[4] LLM 포맷 테스트")
print("-" * 70)
try:
    formatted = agent.format_for_llm(result)
    print("[OK] 포맷팅 성공")
    print(f"  - 전체 길이: {len(formatted)}자")
    print(f"  - 마크다운 헤더 확인: {'##' in formatted}")
    print(f"  - 마크다운 섹션 확인: {'###' in formatted}")
    print()
    print("포맷팅된 결과 (처음 500자):")
    print("-" * 70)
    # Windows 터미널 호환성을 위해 이모지 제거하여 출력
    preview = formatted[:500].encode('ascii', 'ignore').decode('ascii')
    print(preview)
    print("-" * 70)
    
    # 전체 결과를 파일로 저장
    with open('test_search_agent_formatted.txt', 'w', encoding='utf-8') as f:
        f.write(formatted)
    print("전체 결과가 test_search_agent_formatted.txt 파일로 저장되었습니다.")
except Exception as e:
    print(f"[ERROR] 포맷팅 오류: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
print()

# 5. 출처 확인
print("[5] 출처 확인")
print("-" * 70)
try:
    sources = agent.get_sources()
    print(f"[OK] 출처 조회 성공")
    print(f"  - 출처 개수: {len(sources)}")
    if sources:
        print("  - 출처 목록:")
        for idx, source in enumerate(sources[:3], 1):  # 최대 3개만 표시
            print(f"    {idx}. {source}")
        if len(sources) > 3:
            print(f"    ... 외 {len(sources) - 3}개")
    else:
        print("  - 출처 없음")
except Exception as e:
    print(f"[ERROR] 출처 조회 오류: {type(e).__name__}: {e}")
print()

# 6. 히스토리 확인
print("[6] 히스토리 확인")
print("-" * 70)
try:
    count = agent.get_search_count()
    print(f"[OK] 히스토리 조회 성공")
    print(f"  - 총 검색 횟수: {count}")
    
    last_result = agent.get_last_result()
    if last_result:
        print(f"  - 마지막 검색 쿼리: {last_result.query}")
        print(f"  - 마지막 검색 결과 수: {last_result.result_count}")
    else:
        print("  - 마지막 검색 결과 없음")
except Exception as e:
    print(f"[ERROR] 히스토리 조회 오류: {type(e).__name__}: {e}")
print()

# 7. 히스토리 초기화
print("[7] 히스토리 초기화")
print("-" * 70)
try:
    before_count = agent.get_search_count()
    agent.clear_history()
    after_count = agent.get_search_count()
    print(f"[OK] 히스토리 초기화 성공")
    print(f"  - 초기화 전 검색 횟수: {before_count}")
    print(f"  - 초기화 후 검색 횟수: {after_count}")
except Exception as e:
    print(f"[ERROR] 히스토리 초기화 오류: {type(e).__name__}: {e}")
print()

print("=" * 70)
print("테스트 완료")
print("=" * 70)
