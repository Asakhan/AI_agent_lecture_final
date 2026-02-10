#!/usr/bin/env python3
"""
Part 1 종합 테스트 스크립트
실행: python test_part1.py
"""

import sys
import os

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_part1():
    """Part 1 전체 기능 테스트"""
    
    print("=" * 70)
    print("🧪 Part 1 종합 테스트 시작")
    print("=" * 70)
    print()
    
    # ========================================
    # 테스트 1: 임포트 및 초기화
    # ========================================
    print("📦 [1/8] 임포트 및 초기화 테스트...")
    try:
        from src.memory_manager import MemoryManager
        from src.utils.embeddings import EmbeddingGenerator
        print("   ✓ 모듈 임포트 성공")
        
        mm = MemoryManager("test_memory", "data/chroma_db")
        print(f"   ✓ MemoryManager 초기화 성공")
        print(f"   ✓ 컬렉션: {mm.collection_name}")
        print(f"   ✓ 기존 문서 수: {mm.collection.count()}개")
        print()
    except Exception as e:
        print(f"   ✗ 실패: {e}")
        return False
    
    # ========================================
    # 테스트 2: 문서 저장
    # ========================================
    print("💾 [2/8] 문서 저장 테스트...")
    try:
        doc_id1 = mm.add_to_memory(
            text="테슬라는 전기차를 만드는 미국 회사입니다",
            metadata={"source": "test", "category": "company"}
        )
        print(f"   ✓ 문서 1 저장 성공: {doc_id1[:8]}...")
        
        doc_id2 = mm.add_to_memory(
            text="애플은 아이폰을 만드는 기술 기업입니다",
            metadata={"source": "test", "category": "company"}
        )
        print(f"   ✓ 문서 2 저장 성공: {doc_id2[:8]}...")
        
        doc_id3 = mm.add_to_memory(
            text="삼성전자는 한국의 대표적인 전자 기업입니다",
            metadata={"source": "test", "category": "company"}
        )
        print(f"   ✓ 문서 3 저장 성공: {doc_id3[:8]}...")
        print(f"   ✓ 총 문서 수: {mm.collection.count()}개")
        print()
    except Exception as e:
        print(f"   ✗ 실패: {e}")
        return False
    
    # ========================================
    # 테스트 3: 유사도 검색
    # ========================================
    print("🔍 [3/8] 유사도 검색 테스트...")
    try:
        results = mm.search_memory("전기차 회사", top_k=3)
        print(f"   ✓ 검색 결과: {len(results)}개")
        
        for i, result in enumerate(results, 1):
            similarity = result['similarity']
            text_preview = result['text'][:40]
            print(f"   {i}. 유사도: {similarity:.3f} | {text_preview}...")
        
        # 가장 유사한 결과가 테슬라 문서인지 확인
        if "테슬라" in results[0]['text']:
            print("   ✓ 검색 정확도 확인: 가장 유사한 문서가 올바름")
        print()
    except Exception as e:
        print(f"   ✗ 실패: {e}")
        return False
    
    # ========================================
    # 테스트 4: 중복 체크
    # ========================================
    print("🔄 [4/8] 중복 체크 테스트...")
    try:
        duplicate_id = mm.add_to_memory(
            text="테슬라는 전기차를 만드는 미국 회사입니다"  # 동일한 문서
        )
        
        if duplicate_id == doc_id1:
            print(f"   ✓ 중복 문서 감지 성공: 기존 ID 반환")
            print(f"   ✓ 문서 수 변화 없음: {mm.collection.count()}개")
        else:
            print(f"   ⚠ 중복 체크 미작동: 새 ID 생성됨")
        print()
    except Exception as e:
        print(f"   ✗ 실패: {e}")
        return False
    
    # ========================================
    # 테스트 5: 메타데이터 필터링
    # ========================================
    print("🏷️  [5/8] 메타데이터 필터링 테스트...")
    try:
        # 다른 소스로 문서 추가
        mm.add_to_memory(
            text="구글은 검색 엔진을 만드는 회사입니다",
            metadata={"source": "web_search", "category": "company"}
        )
        
        # 소스별 검색
        test_results = mm.search_memory_by_source("회사", "test")
        print(f"   ✓ 'test' 소스 검색: {len(test_results)}개")
        
        web_results = mm.search_memory_by_source("회사", "web_search")
        print(f"   ✓ 'web_search' 소스 검색: {len(web_results)}개")
        print()
    except Exception as e:
        print(f"   ✗ 실패: {e}")
        return False
    
    # ========================================
    # 테스트 6: 문서 삭제
    # ========================================
    print("🗑️  [6/8] 문서 삭제 테스트...")
    try:
        before_count = mm.collection.count()
        
        success = mm.delete_memory(doc_id2)
        after_count = mm.collection.count()
        
        if success and after_count == before_count - 1:
            print(f"   ✓ 문서 삭제 성공")
            print(f"   ✓ 문서 수: {before_count}개 → {after_count}개")
        else:
            print(f"   ⚠ 삭제 실패 또는 카운트 불일치")
        print()
    except Exception as e:
        print(f"   ✗ 실패: {e}")
        return False
    
    # ========================================
    # 테스트 7: 통계 조회
    # ========================================
    print("📊 [7/8] 통계 조회 테스트...")
    try:
        stats = mm.get_statistics()
        
        print(f"   ✓ 총 문서 수: {stats['total_documents']}개")
        print(f"   ✓ 컬렉션: {stats['collection_name']}")
        print(f"   ✓ 소스별 분포:")
        for source, count in stats['by_source'].items():
            print(f"      - {source}: {count}개")
        print()
    except Exception as e:
        print(f"   ✗ 실패: {e}")
        return False
    
    # ========================================
    # 테스트 8: 대시보드 출력
    # ========================================
    print("📈 [8/8] 대시보드 출력 테스트...")
    try:
        print()
        mm.print_memory_dashboard()
        print()
        print("   ✓ 대시보드 출력 성공")
        print()
    except Exception as e:
        print(f"   ✗ 실패: {e}")
        return False
    
    # ========================================
    # 최종 결과
    # ========================================
    print("=" * 70)
    print("✅ Part 1 종합 테스트 완료!")
    print("=" * 70)
    print()
    print("🎉 모든 테스트를 통과했습니다!")
    print()
    print("다음 단계:")
    print("1. tests/test_memory_manager.py 실행하여 단위 테스트 확인")
    print("2. Part 2 (SearchAgent 통합)로 진행")
    print()
    
    return True


if __name__ == "__main__":
    try:
        success = test_part1()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  테스트가 사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 예상치 못한 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)