"""
MemoryManager 단위 테스트

MemoryManager 클래스의 주요 기능에 대한 단위 테스트를 수행합니다.
"""

import unittest
import os
import shutil
import logging
import sys
import json
from pathlib import Path

# #region agent log
try:
    log_path = Path(r"c:\Users\asakh\Documents\GitHub\AI_agent_lecture_03\.cursor\debug.log")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "sessionId": "debug-session",
            "runId": "pre-fix",
            "hypothesisId": "A",
            "location": "test_memory_manager.py:import",
            "message": "Import 시도 전 - sys.path 확인",
            "data": {
                "sys_path": sys.path,
                "cwd": os.getcwd(),
                "script_dir": os.path.dirname(os.path.abspath(__file__)),
                "project_root_exists": os.path.exists(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "src_exists": os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")),
            },
            "timestamp": int(os.path.getmtime(__file__) * 1000) if os.path.exists(__file__) else 0
        }) + "\n")
except Exception:
    pass
# #endregion

# 프로젝트 루트를 sys.path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# #region agent log
try:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "sessionId": "debug-session",
            "runId": "pre-fix",
            "hypothesisId": "B",
            "location": "test_memory_manager.py:sys.path",
            "message": "sys.path에 프로젝트 루트 추가 후",
            "data": {
                "project_root": project_root,
                "project_root_in_sys_path": project_root in sys.path,
                "sys_path_updated": sys.path[:3],
            },
            "timestamp": int(os.path.getmtime(__file__) * 1000) if os.path.exists(__file__) else 0
        }) + "\n")
except Exception:
    pass
# #endregion

from src.memory_manager import MemoryManager
from src.utils.embeddings import EmbeddingGenerator

# #region agent log
try:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "sessionId": "debug-session",
            "runId": "pre-fix",
            "hypothesisId": "C",
            "location": "test_memory_manager.py:import_success",
            "message": "Import 성공 확인",
            "data": {
                "MemoryManager_imported": "MemoryManager" in globals(),
                "EmbeddingGenerator_imported": "EmbeddingGenerator" in globals(),
            },
            "timestamp": int(os.path.getmtime(__file__) * 1000) if os.path.exists(__file__) else 0
        }) + "\n")
except Exception:
    pass
# #endregion


class TestMemoryManager(unittest.TestCase):
    """MemoryManager 단위 테스트"""
    
    @classmethod
    def setUpClass(cls):
        """테스트 시작 전 한 번 실행"""
        cls.test_dir = "data/test_chroma_db"
        # 테스트 디렉토리가 있으면 삭제
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
        
        # MemoryManager 초기화
        cls.mm = MemoryManager("test_collection", cls.test_dir)
    
    @classmethod
    def tearDownClass(cls):
        """테스트 종료 후 정리"""
        # MemoryManager 인스턴스 정리 (참조 제거)
        if hasattr(cls, 'mm'):
            # Chroma DB 클라이언트와 컬렉션 참조 명시적으로 제거
            if hasattr(cls.mm, 'collection'):
                cls.mm.collection = None
            if hasattr(cls.mm, 'client'):
                cls.mm.client = None
            if hasattr(cls.mm, 'embedding_generator'):
                cls.mm.embedding_generator = None
            cls.mm = None
        
        # 가비지 컬렉션 강제 실행 (파일 핸들 해제를 위해)
        import gc
        gc.collect()
        
        # Windows에서 파일이 사용 중일 수 있으므로 재시도 로직 추가
        # 실패해도 테스트는 성공으로 간주 (정리 단계이므로)
        if os.path.exists(cls.test_dir):
            import time
            max_retries = 5  # 재시도 횟수 감소 (너무 오래 기다리지 않음)
            retry_delay = 0.5  # 초기 대기 시간
            
            deleted = False
            for attempt in range(max_retries):
                try:
                    shutil.rmtree(cls.test_dir)
                    deleted = True
                    break  # 성공하면 루프 종료
                except (PermissionError, OSError):
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay = min(retry_delay * 1.5, 2.0)
                        # 중간에 가비지 컬렉션 한 번 더 실행
                        if attempt % 2 == 0:
                            gc.collect()
            
            # 삭제 실패 시 조용히 무시 (다음 테스트 실행 시 덮어쓰기됨)
            # Windows에서 Chroma DB가 파일을 열어두는 것은 정상적인 동작입니다
            if not deleted:
                # DEBUG 레벨로만 로깅 (기본 로깅 레벨에서는 출력되지 않음)
                logging.debug(
                    f"테스트 디렉토리 삭제 건너뜀 (Windows 파일 잠금): {cls.test_dir}. "
                    f"다음 테스트 실행 시 자동으로 덮어쓰기됩니다."
                )
    
    def setUp(self):
        """각 테스트 전 실행"""
        # 기존 데이터 정리
        self.mm.clear_all_memory()
    
    def test_initialization(self):
        """MemoryManager 초기화 테스트"""
        self.assertIsNotNone(self.mm.client)
        self.assertIsNotNone(self.mm.collection)
        self.assertIsNotNone(self.mm.embedding_generator)
        self.assertEqual(self.mm.collection_name, "test_collection")
        self.assertEqual(self.mm.persist_directory, self.test_dir)
    
    def test_add_to_memory(self):
        """문서 저장 기능 테스트"""
        text = "테슬라는 전기차 회사입니다"
        doc_id = self.mm.add_to_memory(text)
        
        self.assertIsNotNone(doc_id)
        self.assertEqual(self.mm.collection.count(), 1)
        
        # 저장된 문서 확인
        all_docs = self.mm.get_all_documents()
        self.assertEqual(len(all_docs), 1)
        self.assertEqual(all_docs[0]['text'], text)
        self.assertIn('timestamp', all_docs[0]['metadata'])
        self.assertIn('source', all_docs[0]['metadata'])
        
        # 빈 텍스트는 예외 발생
        with self.assertRaises(ValueError):
            self.mm.add_to_memory("")
        
        # 메타데이터와 함께 저장
        doc_id2 = self.mm.add_to_memory(
            "애플은 아이폰을 만듭니다",
            metadata={"category": "company", "source": "test"}
        )
        self.assertIsNotNone(doc_id2)
        self.assertEqual(self.mm.collection.count(), 2)
    
    def test_search_memory(self):
        """검색 기능 테스트"""
        # 문서 저장
        self.mm.add_to_memory("테슬라는 전기차 회사입니다")
        self.mm.add_to_memory("애플은 아이폰을 만듭니다")
        self.mm.add_to_memory("전기차는 환경 친화적입니다")
        
        # 검색
        results = self.mm.search_memory("전기차", top_k=2)
        
        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 2)
        
        # 결과 구조 확인
        self.assertIn('text', results[0])
        self.assertIn('similarity', results[0])
        self.assertIn('id', results[0])
        self.assertIn('metadata', results[0])
        
        # 유사도 점수 확인
        self.assertGreater(results[0]['similarity'], 0.0)
        self.assertLessEqual(results[0]['similarity'], 1.0)
        
        # 빈 쿼리는 예외 발생
        with self.assertRaises(ValueError):
            self.mm.search_memory("")
    
    def test_duplicate_check(self):
        """중복 문서 체크 테스트"""
        text = "테슬라는 전기차 회사입니다"
        
        # 첫 번째 저장
        doc_id1 = self.mm.add_to_memory(text, check_duplicate=True)
        self.assertEqual(self.mm.collection.count(), 1)
        
        # 중복 저장 시도 (check_duplicate=True)
        doc_id2 = self.mm.add_to_memory(text, check_duplicate=True)
        
        # 동일한 ID 반환되어야 함
        self.assertEqual(doc_id1, doc_id2)
        self.assertEqual(self.mm.collection.count(), 1)  # 중복 저장 안 됨
        
        # check_duplicate=False로 강제 저장
        doc_id3 = self.mm.add_to_memory(text, check_duplicate=False)
        self.assertNotEqual(doc_id1, doc_id3)
        self.assertEqual(self.mm.collection.count(), 2)  # 강제 저장됨
    
    def test_delete_memory(self):
        """문서 삭제 기능 테스트"""
        doc_id = self.mm.add_to_memory("삭제할 문서")
        
        # 삭제 전
        self.assertEqual(self.mm.collection.count(), 1)
        
        # 삭제
        result = self.mm.delete_memory(doc_id)
        self.assertTrue(result)
        
        # 삭제 후
        self.assertEqual(self.mm.collection.count(), 0)
        
        # 존재하지 않는 문서 삭제 시도
        result2 = self.mm.delete_memory("nonexistent_id")
        self.assertFalse(result2)
    
    def test_clear_all_memory(self):
        """전체 메모리 삭제 테스트"""
        # 여러 문서 추가
        for i in range(5):
            self.mm.add_to_memory(f"테스트 문서 {i}")
        
        self.assertEqual(self.mm.collection.count(), 5)
        
        # 전체 삭제
        deleted_count = self.mm.clear_all_memory()
        self.assertEqual(deleted_count, 5)
        self.assertEqual(self.mm.collection.count(), 0)
    
    def test_get_all_documents(self):
        """전체 문서 조회 테스트"""
        # 문서 추가
        texts = ["문서1", "문서2", "문서3"]
        for text in texts:
            self.mm.add_to_memory(text)
        
        # 전체 조회
        all_docs = self.mm.get_all_documents()
        
        self.assertEqual(len(all_docs), 3)
        for doc in all_docs:
            self.assertIn('id', doc)
            self.assertIn('text', doc)
            self.assertIn('metadata', doc)
    
    def test_metadata_filtering(self):
        """메타데이터 필터링 테스트"""
        self.mm.add_to_memory("문서1", metadata={"source": "web", "category": "tech"})
        self.mm.add_to_memory("문서2", metadata={"source": "user", "category": "tech"})
        self.mm.add_to_memory("문서3", metadata={"source": "web", "category": "news"})
        
        # 소스별 검색
        web_results = self.mm.search_memory_by_source("문서", "web", top_k=10)
        
        self.assertGreaterEqual(len(web_results), 2)
        for result in web_results:
            self.assertEqual(result['metadata']['source'], "web")
        
        # 메타데이터로 문서 조회
        web_docs = self.mm.get_documents_by_metadata("source", "web")
        self.assertEqual(len(web_docs), 2)
        
        tech_docs = self.mm.get_documents_by_metadata("category", "tech")
        self.assertEqual(len(tech_docs), 2)
    
    def test_statistics(self):
        """통계 기능 테스트"""
        # 여러 문서 추가
        for i in range(5):
            self.mm.add_to_memory(
                f"테스트 문서 {i}",
                metadata={"source": "test" if i % 2 == 0 else "user"}
            )
        
        stats = self.mm.get_statistics()
        
        self.assertEqual(stats['total_documents'], 5)
        self.assertIn('by_source', stats)
        self.assertIn('by_date', stats)
        self.assertGreater(stats['avg_text_length'], 0)
        self.assertEqual(stats['collection_name'], "test_collection")
        self.assertIn('cache_info', stats)
        
        # 소스별 분포 확인
        self.assertIn('test', stats['by_source'])
        self.assertIn('user', stats['by_source'])
    
    def test_mark_as_important(self):
        """중요 문서 표시 테스트"""
        doc_id = self.mm.add_to_memory("중요한 문서")
        
        # 중요 표시
        result = self.mm.mark_as_important(doc_id)
        self.assertTrue(result)
        
        # 메타데이터 확인
        all_docs = self.mm.get_all_documents()
        important_doc = next((doc for doc in all_docs if doc['id'] == doc_id), None)
        self.assertIsNotNone(important_doc)
        self.assertTrue(important_doc['metadata'].get('important', False))
        
        # 존재하지 않는 문서는 실패
        result2 = self.mm.mark_as_important("nonexistent_id")
        self.assertFalse(result2)
    
    def test_cleanup_old_memories(self):
        """오래된 메모리 정리 테스트"""
        from datetime import datetime, timedelta
        
        # 최근 문서 추가
        self.mm.add_to_memory("최근 문서1")
        self.mm.add_to_memory("최근 문서2")
        
        # 오래된 문서 시뮬레이션 (과거 날짜로 메타데이터 설정)
        old_date = (datetime.now() - timedelta(days=60)).isoformat()
        old_doc_id = self.mm.add_to_memory(
            "오래된 문서",
            metadata={"timestamp": old_date}
        )
        
        # 중요 문서 추가 (오래되었지만 보존되어야 함)
        very_old_date = (datetime.now() - timedelta(days=90)).isoformat()
        important_doc_id = self.mm.add_to_memory(
            "중요한 오래된 문서",
            metadata={"timestamp": very_old_date, "important": True}
        )
        
        self.assertEqual(self.mm.collection.count(), 4)
        
        # 30일 이상 오래된 문서 정리 (중요 문서 보존)
        result = self.mm.cleanup_old_memories(days_old=30, keep_important=True)
        
        self.assertGreater(result['deleted'], 0)
        self.assertIn('kept', result)
        self.assertIn('cutoff_date', result)
        
        # 중요 문서는 보존되어야 함
        remaining_docs = self.mm.get_all_documents()
        remaining_ids = [doc['id'] for doc in remaining_docs]
        self.assertIn(important_doc_id, remaining_ids)
    
    def test_get_memory_stats(self):
        """상세 메모리 통계 테스트"""
        # 여러 문서 추가
        for i in range(3):
            self.mm.add_to_memory(f"통계 테스트 문서 {i}")
        
        stats = self.mm.get_memory_stats()
        
        # 기본 통계 확인
        self.assertEqual(stats['total_documents'], 3)
        self.assertIn('by_source', stats)
        self.assertIn('by_date', stats)
        self.assertGreater(stats['avg_text_length'], 0)
        self.assertEqual(stats['embedding_dimension'], 1536)
        
        # 날짜별 분포 확인
        self.assertIn('last_24h', stats['by_date'])
        self.assertIn('last_7days', stats['by_date'])
        self.assertIn('last_30days', stats['by_date'])
        self.assertIn('older', stats['by_date'])
        
        # 캐시 정보 확인
        self.assertIn('cache_info', stats)
        cache_info = stats['cache_info']
        self.assertIn('size', cache_info)
        self.assertIn('hit_rate', cache_info)
    
    def test_check_duplicate(self):
        """중복 체크 메서드 테스트"""
        text = "중복 체크 테스트 문서"
        
        # 첫 번째 저장
        doc_id1 = self.mm.add_to_memory(text)
        
        # 중복 체크
        duplicate = self.mm.check_duplicate(text, threshold=0.95)
        
        self.assertIsNotNone(duplicate)
        self.assertEqual(duplicate['id'], doc_id1)
        self.assertGreaterEqual(duplicate['similarity'], 0.95)
        
        # 다른 텍스트는 중복 아님
        different_text = "완전히 다른 내용"
        duplicate2 = self.mm.check_duplicate(different_text, threshold=0.95)
        # 결과가 없거나 유사도가 낮아야 함
        if duplicate2:
            self.assertLess(duplicate2['similarity'], 0.95)


if __name__ == '__main__':
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 테스트 실행
    unittest.main(verbosity=2)
