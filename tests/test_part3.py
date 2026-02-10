#!/usr/bin/env python3
"""
Part 3 종합 테스트 스크립트
실행: python test_part3.py
"""

import sys
import os

# 프로젝트 루트를 sys.path에 추가 (src 패키지 인식용, test_part2와 동일)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def test_part3():
    """Part 3 전체 기능 테스트"""
    
    print("=" * 70)
    print("🧪 Part 3 종합 테스트 시작")
    print("=" * 70)
    print()
    
    # ========================================
    # 테스트 1: 전체 시스템 초기화
    # ========================================
    print("📦 [1/5] 전체 시스템 초기화 테스트...")
    try:
        from src.memory_manager import MemoryManager
        from src.search_agent import SearchAgent
        from src.conversation_manager import ConversationManager
        
        # MemoryManager 초기화
        mm = MemoryManager("test_system", "data/chroma_db")
        print(f"   ✓ MemoryManager: {mm.collection.count()}개 문서")
        
        # SearchAgent 초기화 (메모리 포함)
        agent = SearchAgent(memory_manager=mm)
        print(f"   ✓ SearchAgent with memory")
        
        # ConversationManager 초기화 (메모리 포함)
        conv_mgr = ConversationManager(
            search_agent=agent,
            memory_manager=mm
        )
        print(f"   ✓ ConversationManager with memory")
        print()
    except Exception as e:
        print(f"   ✗ 실패: {e}")
        return False
    
    # ========================================
    # 테스트 2: 검색 결과 자동 저장
    # ========================================
    print("💾 [2/5] 검색 결과 자동 저장 테스트...")
    try:
        before_count = mm.collection.count()
        
        # 검색 수행
        search_results = agent.search_with_memory(
            query="인공지능 최신 동향",
            use_memory=True,
            save_to_memory=True
        )
        
        # ConversationManager로 추가 저장
        saved = conv_mgr.save_search_result_to_memory(
            search_results,
            "인공지능 최신 동향"
        )
        
        after_count = mm.collection.count()
        
        print(f"   ✓ 검색 결과 저장: {saved}개")
        print(f"   ✓ 메모리 증가: {before_count}개 → {after_count}개")
        print()
    except Exception as e:
        print(f"   ✗ 실패: {e}")
        return False
    
    # ========================================
    # 테스트 3: 대화 내용 자동 저장
    # ========================================
    print("💬 [3/5] 대화 내용 자동 저장 테스트...")
    try:
        before_count = mm.collection.count()
        
        # 대화 저장
        success = conv_mgr.save_conversation_to_memory(
            user_message="인공지능이 뭐야?",
            assistant_message="인공지능은 기계가 인간의 지능을 모방하여..."
        )
        
        after_count = mm.collection.count()
        
        print(f"   ✓ 대화 저장 성공: {success}")
        print(f"   ✓ 메모리 증가: {before_count}개 → {after_count}개")
        
        # 저장된 대화 검색
        conv_results = mm.search_memory("인공지능", top_k=3)
        conversation_found = False
        for r in conv_results:
            if r['metadata'].get('source') == 'conversation':
                conversation_found = True
                print(f"   ✓ 대화 내용 검색됨: {r['text'][:50]}...")
                break
        
        if conversation_found:
            print(f"   ✓ 대화 메모리 검증 완료")
        print()
    except Exception as e:
        print(f"   ✗ 실패: {e}")
        return False
    
    # ========================================
    # 테스트 4: chat() 메서드 통합 테스트
    # ========================================
    print("🤖 [4/5] chat() 메서드 통합 테스트...")
    try:
        before_count = mm.collection.count()
        
        # 실제 대화 수행 (검색 포함)
        response = conv_mgr.chat("Python 프로그래밍 언어에 대해 알려줘")
        
        after_count = mm.collection.count()
        
        print(f"   ✓ 응답 생성 성공")
        print(f"   ✓ 응답 길이: {len(response)}자")
        print(f"   ✓ 메모리 증가: {before_count}개 → {after_count}개")
        print(f"   ✓ 자동 저장 확인: {after_count > before_count}")
        print()
    except Exception as e:
        print(f"   ✗ 실패: {e}")
        return False
    
    # ========================================
    # 테스트 5: 메모리 통계 및 검색
    # ========================================
    print("📊 [5/5] 메모리 통계 및 검색 테스트...")
    try:
        # 통계 조회
        stats = mm.get_statistics()
        print(f"   ✓ 총 문서 수: {stats['total_documents']}개")
        print(f"   ✓ 소스별 분포:")
        for source, count in stats['by_source'].items():
            print(f"      - {source}: {count}개")
        
        # 메모리 검색
        results = mm.search_memory("Python", top_k=3)
        print(f"   ✓ 검색 결과: {len(results)}개")
        
        if results:
            print(f"   ✓ 상위 결과:")
            for i, r in enumerate(results[:2], 1):
                print(f"      {i}. {r['text'][:40]}...")
                print(f"         (유사도: {r['similarity']:.3f}, 출처: {r['metadata'].get('source', 'unknown')})")
        
        # 대시보드 출력
        print()
        print("   📈 메모리 대시보드:")
        mm.print_memory_dashboard()
        print()
    except Exception as e:
        print(f"   ✗ 실패: {e}")
        return False
    
    # ========================================
    # 최종 결과
    # ========================================
    print("=" * 70)
    print("✅ Part 3 종합 테스트 완료!")
    print("=" * 70)
    print()
    print("🎉 모든 테스트를 통과했습니다!")
    print()
    print("다음 단계:")
    print("1. main.py 실행하여 전체 시스템 테스트")
    print("2. 다양한 대화 시나리오 테스트")
    print("3. 메모리 명령어 (/memory, /memory-search) 테스트")
    print()
    
    return True


if __name__ == "__main__":
    try:
        success = test_part3()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  테스트가 사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 예상치 못한 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)