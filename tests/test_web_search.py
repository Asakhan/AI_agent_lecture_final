"""웹 검색 모듈 테스트 스크립트"""

from src.tools.web_search import (
    tavily_search,
    SearchResult,
    format_search_result_for_llm,
    optimize_search_query,
)

print("=" * 60)
print("1. 모듈 Import 테스트")
print("=" * 60)
print("[OK] 모든 모듈이 성공적으로 import되었습니다.")
print()

print("=" * 60)
print("2. 쿼리 최적화 테스트")
print("=" * 60)
test_query = "최신 AI 트렌드에 대해 알려줘"
optimized = optimize_search_query(test_query)
print(f"원본: {test_query}")
print(f"최적화: {optimized}")
print()

print("=" * 60)
print("3. 검색 테스트")
print("=" * 60)
try:
    result = tavily_search("AI agent trends 2024", max_results=3)
    print(f"결과 수: {result.result_count}")
    print(f"요약 존재: {result.has_answer}")
    if result.answer:
        print(f"요약 미리보기: {result.answer[:100]}...")
    else:
        print("요약: None")
    print(f"검색 소요 시간: {result.search_time:.2f}초")
    print(f"출처 개수: {len(result.sources)}")
except Exception as e:
    print(f"검색 오류: {type(e).__name__}: {e}")
print()

print("=" * 60)
print("4. 포맷팅 테스트")
print("=" * 60)
try:
    result = tavily_search("Python web framework", max_results=2)
    formatted = format_search_result_for_llm(result)
    print("포맷팅된 결과 (처음 500자, 이모지 제외):")
    print("-" * 60)
    # 이모지 제거하여 출력 (Windows 터미널 호환성)
    preview = formatted[:500].encode('ascii', 'ignore').decode('ascii')
    print(preview)
    print("-" * 60)
    print(f"전체 길이: {len(formatted)}자")
    print(f"마크다운 형식 확인: {'##' in formatted and '###' in formatted}")
    # 파일로도 저장
    with open('test_formatted_result.txt', 'w', encoding='utf-8') as f:
        f.write(formatted)
    print("전체 결과가 test_formatted_result.txt 파일로 저장되었습니다.")
except Exception as e:
    print(f"포맷팅 테스트 오류: {type(e).__name__}: {e}")

print()
print("=" * 60)
print("테스트 완료")
print("=" * 60)
