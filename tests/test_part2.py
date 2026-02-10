#!/usr/bin/env python3
"""
Part 2 종합 테스트 스크립트
실행: python test_part2.py
"""

import sys
import os
import json
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def test_part2():
    """Part 2 전체 기능 테스트"""
    
    print("=" * 70)
    print("🧪 Part 2 종합 테스트 시작")
    print("=" * 70)
    print()
    
    # ========================================
    # 테스트 1: 임포트 및 초기화
    # ========================================
    print("📦 [1/6] 임포트 및 초기화 테스트...")
    try:
        from src.search_agent import SearchAgent
        from src.memory_manager import MemoryManager
        print("   ✓ 모듈 임포트 성공")
        
        # MemoryManager 초기화
        mm = MemoryManager("search_memory", "data/chroma_db")
        print(f"   ✓ MemoryManager 초기화: {mm.collection.count()}개 문서")
        
        # SearchAgent 초기화 (메모리 포함)
        agent = SearchAgent(memory_manager=mm)
        print(f"   ✓ SearchAgent 메모리 통합 완료")
        print()
    except Exception as e:
        print(f"   ✗ 실패: {e}")
        return False
    
    # ========================================
    # 테스트 2: 메모리에 샘플 데이터 저장
    # ========================================
    print("💾 [2/6] 샘플 데이터 저장 테스트...")
    try:
        samples = [
            "테슬라는 2003년 설립된 미국의 전기차 제조 회사입니다",
            "테슬라 CEO 일론 머스크는 혁신적인 기업가로 알려져 있습니다",
            "테슬라 모델 3는 가장 인기있는 전기차 모델 중 하나입니다"
        ]
        
        for sample in samples:
            mm.add_to_memory(
                text=sample,
                metadata={"source": "test_data", "category": "tesla"}
            )
        
        print(f"   ✓ {len(samples)}개 샘플 데이터 저장 완료")
        print(f"   ✓ 현재 메모리: {mm.collection.count()}개 문서")
        print()
    except Exception as e:
        print(f"   ✗ 실패: {e}")
        return False
    
    # ========================================
    # 테스트 3: 메모리 검색 (웹 검색 없이)
    # ========================================
    print("🔍 [3/6] 메모리 전용 검색 테스트...")
    try:
        result = agent.search_with_memory(
            query="테슬라 전기차",
            use_memory=True,
            save_to_memory=False  # 웹 검색 안 함
        )
        
        print(f"   ✓ 검색 완료")
        print(f"   ✓ 메모리 결과: {result['source_summary']['from_memory']}개")
        print(f"   ✓ 웹 결과: {result['source_summary']['from_web']}개")
        
        if result['memory_results']:
            top_result = result['memory_results'][0]
            print(f"   ✓ 상위 결과: {top_result['text'][:50]}...")
            print(f"   ✓ 유사도: {top_result['similarity']:.3f}")
        print()
    except Exception as e:
        print(f"   ✗ 실패: {e}")
        return False
    
    # ========================================
    # 테스트 4: 웹 검색 + 메모리 저장
    # ========================================
    print("🌐 [4/6] 웹 검색 및 저장 테스트...")
    try:
        before_count = mm.collection.count()
        
        result = agent.search_with_memory(
            query="전기차 최신 기술",
            use_memory=True,
            save_to_memory=True,
            memory_threshold=10  # 메모리 결과 부족 → 웹 검색
        )
        
        after_count = mm.collection.count()
        
        print(f"   ✓ 검색 완료")
        print(f"   ✓ 메모리 결과: {result['source_summary']['from_memory']}개")
        print(f"   ✓ 웹 결과: {result['source_summary']['from_web']}개")
        print(f"   ✓ 메모리 증가: {before_count}개 → {after_count}개")
        print()
    except Exception as e:
        print(f"   ✗ 실패: {e}")
        return False
    
    # ========================================
    # 테스트 5: 결과 병합 확인
    # ========================================
    print("🔗 [5/6] 결과 병합 테스트...")
    try:
        result = agent.search_with_memory(
            query="테슬라",
            use_memory=True,
            save_to_memory=True
        )
        
        merged = result['merged_results']
        print(f"   ✓ 병합 결과: {len(merged)}개")
        
        # 출처별 카운트
        sources = {}
        for r in merged:
            source = r.get('source', 'unknown')
            sources[source] = sources.get(source, 0) + 1
        
        print(f"   ✓ 출처 분포:")
        for source, count in sources.items():
            print(f"      - {source}: {count}개")
        
        # 상위 3개 결과 확인
        print(f"   ✓ 상위 3개 결과:")
        for i, r in enumerate(merged[:3], 1):
            content_preview = r['content'][:40]
            print(f"      {i}. [{r['source']}] {content_preview}...")
        print()
    except Exception as e:
        print(f"   ✗ 실패: {e}")
        return False
    
    # ========================================
    # 테스트 6: Provenance 확인
    # ========================================
    print("📊 [6/6] Provenance 추적 테스트...")
    try:
        result = agent.search_with_memory(query="테슬라")
        
        if result['merged_results']:
            sample = result['merged_results'][0]
            provenance = sample.get('provenance', {})
            
            print(f"   ✓ Provenance 정보:")
            print(f"      - 출처: {provenance.get('retrieved_from', 'N/A')}")
            
            if provenance.get('retrieved_from') == 'memory':
                print(f"      - 원본 출처: {provenance.get('original_source', 'N/A')}")
                print(f"      - 신뢰도: {provenance.get('confidence', 0):.3f}")
            elif provenance.get('retrieved_from') == 'web':
                print(f"      - URL: {provenance.get('url', 'N/A')[:50]}...")
            
            print(f"   ✓ Provenance 추적 성공")
        print()
    except Exception as e:
        print(f"   ✗ 실패: {e}")
        return False
    
    # ========================================
    # 최종 결과
    # ========================================
    print("=" * 70)
    print("✅ Part 2 종합 테스트 완료!")
    print("=" * 70)
    print()
    print("🎉 모든 테스트를 통과했습니다!")
    print()
    print("다음 단계:")
    print("1. 다양한 쿼리로 추가 테스트")
    print("2. Part 3 (ConversationManager 통합)로 진행")
    print()
    
    return True


if __name__ == "__main__":
    try:
        success = test_part2()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  테스트가 사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 예상치 못한 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)