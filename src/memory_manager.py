"""
메모리 관리 모듈

Chroma DB를 사용한 장기 메모리 관리 시스템입니다.
벡터 임베딩을 활용하여 대화 내용과 검색 결과를 저장하고 검색할 수 있습니다.
"""

import chromadb
from chromadb.config import Settings
from src.utils.embeddings import EmbeddingGenerator
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging
import uuid
import os

# 로깅 설정
logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Chroma DB 기반 장기 메모리 관리 시스템.
    
    벡터 임베딩을 사용하여 텍스트를 저장하고, 유사도 기반 검색을 제공합니다.
    대화 내용, 검색 결과, 중요한 정보 등을 장기적으로 저장하고 관리합니다.
    
    Attributes:
        collection_name (str): Chroma 컬렉션 이름
        persist_directory (str): 벡터 DB 저장 경로
        client (chromadb.Client): Chroma DB 클라이언트 인스턴스
        collection (chromadb.Collection): Chroma 컬렉션 인스턴스
        embedding_generator (EmbeddingGenerator): 임베딩 생성기 인스턴스
    """
    
    def __init__(
        self, 
        collection_name: str = "research_memory",
        persist_directory: str = "data/chroma_db"
    ) -> None:
        """
        MemoryManager 인스턴스를 초기화합니다.
        
        Chroma DB 클라이언트와 컬렉션을 초기화하고, 임베딩 생성기를 설정합니다.
        지정된 디렉토리에 벡터 DB가 영구 저장됩니다.
        
        Args:
            collection_name: Chroma 컬렉션 이름 (기본값: "research_memory")
            persist_directory: 벡터 DB 저장 경로 (기본값: "data/chroma_db")
            
        Raises:
            PermissionError: 폴더 생성 권한 오류
            Exception: Chroma DB 초기화 실패 시
        """
        try:
            # 1. 로거 초기화 (이미 모듈 레벨에서 설정됨)
            # logger = logging.getLogger(__name__)
            
            # 2. persist_directory 폴더 생성
            if not os.path.exists(persist_directory):
                os.makedirs(persist_directory, exist_ok=True)
                logger.info(f"디렉토리 생성: {persist_directory}")
            
            # 3. Chroma DB 클라이언트 생성
            self.client = chromadb.PersistentClient(path=persist_directory)
            
            # 4. 컬렉션 생성 또는 로드
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # 코사인 유사도
            )
            
            # 5. EmbeddingGenerator 초기화
            self.embedding_generator = EmbeddingGenerator()
            
            # 6. 속성 저장
            self.collection_name = collection_name
            self.persist_directory = persist_directory
            
            # 7. 초기화 로그
            logger.info(f"MemoryManager 초기화 완료")
            logger.info(f"컬렉션: {collection_name}")
            logger.info(f"저장 경로: {persist_directory}")
            logger.info(f"기존 문서 수: {self.collection.count()}")
            
        except PermissionError as e:
            error_msg = (
                f"폴더 생성 권한 오류: {persist_directory}\n"
                f"디렉토리 생성 권한을 확인해주세요."
            )
            logger.error(error_msg, exc_info=True)
            raise PermissionError(error_msg) from e
            
        except Exception as e:
            error_msg = f"MemoryManager 초기화 실패: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise Exception(error_msg) from e
    
    def check_duplicate(
        self, 
        text: str, 
        threshold: float = 0.95
    ) -> Optional[Dict[str, Any]]:
        """
        중복 문서를 확인합니다.
        
        저장된 문서 중에서 주어진 텍스트와 유사도가 임계값 이상인 문서를 찾습니다.
        
        Args:
            text: 확인할 텍스트
            threshold: 중복 판단 임계값 (0-1, 기본값: 0.95)
                유사도가 이 값 이상이면 중복으로 판단합니다.
            
        Returns:
            Optional[Dict[str, Any]]: 중복 문서 정보 (id, text, metadata, similarity) 또는 None
                중복 문서가 없으면 None을 반환합니다.
        """
        try:
            # 입력 검증
            if not text or not text.strip():
                return None
            
            # 유사 문서 검색
            results = self.search_memory(text, top_k=1)
            
            if not results:
                return None
            
            # 가장 유사한 문서 확인
            most_similar = results[0]
            
            if most_similar['similarity'] >= threshold:
                logger.info(
                    f"중복 문서 발견: {most_similar['id']} "
                    f"(유사도: {most_similar['similarity']:.3f})"
                )
                return most_similar
            
            return None
            
        except Exception as e:
            logger.warning(f"중복 체크 중 오류 발생: {str(e)}")
            return None
    
    def add_to_memory(
        self, 
        text: str, 
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
        check_duplicate: bool = True
    ) -> str:
        """
        텍스트를 벡터 DB에 저장합니다.
        
        텍스트를 임베딩으로 변환하여 Chroma DB에 저장합니다.
        메타데이터와 함께 저장할 수 있으며, 문서 ID를 자동 생성하거나 지정할 수 있습니다.
        중복 문서 체크 기능을 통해 동일하거나 매우 유사한 문서의 중복 저장을 방지할 수 있습니다.
        
        Args:
            text: 저장할 텍스트 내용
            metadata: 문서와 함께 저장할 메타데이터 딕셔너리 (예: timestamp, source 등)
            document_id: 문서 ID (지정하지 않으면 자동 생성)
            check_duplicate: 중복 문서 체크 여부 (기본값: True)
                True인 경우 유사도 0.95 이상인 문서가 있으면 저장하지 않고 기존 문서 ID를 반환합니다.
            
        Returns:
            str: 저장된 문서의 ID (중복 문서가 발견된 경우 기존 문서의 ID)
            
        Raises:
            ValueError: 텍스트가 비어있는 경우
            Exception: API 오류 등 기타 오류
        """
        try:
            # 1. 입력 검증
            if not text or not text.strip():
                raise ValueError("텍스트가 비어있습니다")
            text = text.strip()
            
            # 2. 중복 체크 (check_duplicate=True인 경우)
            if check_duplicate:
                duplicate = self.check_duplicate(text)
                if duplicate:
                    logger.warning(f"중복 문서로 저장 스킵: {duplicate['id']}")
                    return duplicate['id']  # 기존 문서 ID 반환
            
            # 3. document_id 생성 (없는 경우)
            if document_id is None:
                document_id = str(uuid.uuid4())
            
            # 4. 임베딩 생성
            embedding = self.embedding_generator.create_embedding(text)
            
            # 5. 메타데이터 자동 추가
            if metadata is None:
                metadata = {}
            
            # 기존 source가 있으면 유지, 없으면 기본값 설정
            source = metadata.get("source", "user_input")
            
            # timestamp가 이미 있으면 유지, 없으면 현재 시간 설정
            if "timestamp" not in metadata:
                metadata["timestamp"] = datetime.now().isoformat()
            
            # text_length와 source는 항상 업데이트
            metadata.update({
                "text_length": len(text),
                "source": source
            })
            
            # 6. Chroma DB에 저장
            self.collection.add(
                ids=[document_id],
                documents=[text],
                embeddings=[embedding],
                metadatas=[metadata]
            )
            
            # 7. 로깅 및 반환
            logger.info(f"문서 저장 완료: {document_id}")
            return document_id
            
        except ValueError as e:
            logger.error(f"입력 오류: {str(e)}")
            raise
            
        except Exception as e:
            error_msg = f"문서 저장 실패: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise Exception(error_msg) from e
    
    def search_memory(
        self, 
        query: str, 
        top_k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        쿼리와 유사한 문서를 검색합니다.
        
        쿼리 텍스트를 임베딩으로 변환하고, 저장된 문서들과의 유사도를 계산하여
        가장 유사한 문서들을 반환합니다.
        
        Args:
            query: 검색 쿼리 텍스트
            top_k: 반환할 최대 문서 개수 (기본값: 5)
            filter_dict: 메타데이터 필터 딕셔너리 (선택적)
            
        Returns:
            List[Dict[str, Any]]: 검색 결과 리스트
                각 딕셔너리는 다음 키를 포함:
                - id: 문서 ID
                - text: 문서 텍스트
                - metadata: 문서 메타데이터
                - similarity: 유사도 점수 (0-1, 높을수록 유사)
                
        Raises:
            ValueError: 쿼리가 비어있는 경우
            Exception: 검색 실패 시
        """
        try:
            # 1. 입력 검증
            if not query or not query.strip():
                raise ValueError("검색 쿼리가 비어있습니다")
            query = query.strip()
            
            # 2. 쿼리 임베딩 생성
            query_embedding = self.embedding_generator.create_embedding(query)
            
            # 3. Chroma DB 검색
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter_dict  # 메타데이터 필터
            )
            
            # 4. 결과 포맷팅
            formatted_results: List[Dict[str, Any]] = []
            
            # 결과가 있는 경우에만 처리
            if results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    # 거리를 유사도로 변환 (코사인 거리: 0에 가까울수록 유사)
                    # similarity = 1 - distance (코사인 거리의 경우)
                    distance = results['distances'][0][i]
                    similarity = 1.0 - distance
                    
                    formatted_results.append({
                        'id': results['ids'][0][i],
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {},
                        'similarity': similarity
                    })
            
            # 5. 로깅 및 반환
            logger.info(f"검색 완료: {len(formatted_results)}개 결과 (쿼리: {query[:50]}...)")
            return formatted_results
            
        except ValueError as e:
            logger.error(f"입력 오류: {str(e)}")
            raise
            
        except Exception as e:
            error_msg = f"검색 실패: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise Exception(error_msg) from e
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """
        저장된 모든 문서를 조회합니다.
        
        Returns:
            List[Dict[str, Any]]: 모든 문서 리스트
                각 딕셔너리는 다음 키를 포함:
                - id: 문서 ID
                - text: 문서 텍스트
                - metadata: 문서 메타데이터
        """
        try:
            results = self.collection.get()
            
            documents: List[Dict[str, Any]] = []
            for i in range(len(results['ids'])):
                documents.append({
                    'id': results['ids'][i],
                    'text': results['documents'][i],
                    'metadata': results['metadatas'][i] if results['metadatas'] and i < len(results['metadatas']) else {}
                })
            
            logger.info(f"전체 문서 조회: {len(documents)}개")
            return documents
            
        except Exception as e:
            error_msg = f"문서 조회 실패: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise Exception(error_msg) from e
    
    def delete_memory(self, document_id: str) -> bool:
        """
        지정된 문서를 삭제합니다.
        
        Args:
            document_id: 삭제할 문서의 ID
            
        Returns:
            bool: 삭제 성공 여부 (True: 성공, False: 문서를 찾을 수 없음)
        """
        try:
            # 문서 존재 여부 확인
            doc = self.collection.get(ids=[document_id])
            if not doc['ids'] or len(doc['ids']) == 0:
                logger.warning(f"문서를 찾을 수 없음: {document_id}")
                return False
            
            # 문서 삭제
            self.collection.delete(ids=[document_id])
            logger.info(f"문서 삭제 완료: {document_id}")
            return True
        except Exception as e:
            logger.error(f"문서 삭제 실패: {str(e)}")
            return False
    
    def clear_all_memory(self) -> int:
        """
        저장된 모든 문서를 삭제합니다.
        
        컬렉션을 삭제하고 재생성하여 모든 문서를 제거합니다.
        
        Returns:
            int: 삭제된 문서 개수
        """
        try:
            count = self.collection.count()
            
            # 컬렉션 삭제 후 재생성
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"전체 메모리 삭제: {count}개 문서")
            return count
            
        except Exception as e:
            error_msg = f"전체 메모리 삭제 실패: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise Exception(error_msg) from e
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        메모리 통계 정보를 반환합니다.
        
        Returns:
            Dict[str, Any]: 통계 정보 딕셔너리
                - total_documents: 총 문서 개수
                - collection_name: 컬렉션 이름
                - persist_directory: 저장 경로
                - by_source: 소스별 문서 개수
                - by_date: 기간별 문서 개수
                - avg_text_length: 평균 텍스트 길이
                - cache_info: 임베딩 캐시 정보
        """
        try:
            from datetime import timedelta
            
            all_docs = self.get_all_documents()
            
            # 소스별 카운트
            sources: Dict[str, int] = {}
            for doc in all_docs:
                source = doc['metadata'].get('source', 'unknown')
                sources[source] = sources.get(source, 0) + 1
            
            # 날짜별 카운트
            now = datetime.now()
            
            date_counts = {
                'last_24h': 0,
                'last_7days': 0,
                'last_30days': 0,
                'older': 0
            }
            
            for doc in all_docs:
                timestamp_str = doc['metadata'].get('timestamp', '')
                if timestamp_str:
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        age = (now - timestamp.replace(tzinfo=None)).days
                        
                        if age < 1:
                            date_counts['last_24h'] += 1
                        if age < 7:
                            date_counts['last_7days'] += 1
                        if age < 30:
                            date_counts['last_30days'] += 1
                        else:
                            date_counts['older'] += 1
                    except (ValueError, AttributeError):
                        # 날짜 파싱 실패 시 older로 분류
                        date_counts['older'] += 1
                else:
                    date_counts['older'] += 1
            
            # 텍스트 길이 통계
            text_lengths = [len(doc['text']) for doc in all_docs]
            avg_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0
            
            stats = {
                'total_documents': len(all_docs),
                'collection_name': self.collection_name,
                'by_source': sources,
                'by_date': date_counts,
                'avg_text_length': int(avg_length),
                'persist_directory': self.persist_directory,
                'embedding_dimension': self.embedding_generator.get_dimension(),
                'cache_info': self.embedding_generator.get_cache_info()
            }
            
            logger.info(f"통계 조회: {stats['total_documents']}개 문서")
            return stats
            
        except Exception as e:
            error_msg = f"통계 조회 실패: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise Exception(error_msg) from e
    
    def search_memory_by_source(
        self, 
        query: str, 
        source: str, 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        특정 소스에서만 검색합니다.
        
        지정된 소스(source)의 메타데이터를 가진 문서들 중에서만
        쿼리와 유사한 문서를 검색합니다.
        
        Args:
            query: 검색 쿼리 텍스트
            source: 검색할 소스 이름 (예: "web_search", "user_input" 등)
            top_k: 반환할 최대 문서 개수 (기본값: 5)
            
        Returns:
            List[Dict[str, Any]]: 검색 결과 리스트
                각 딕셔너리는 다음 키를 포함:
                - id: 문서 ID
                - text: 문서 텍스트
                - metadata: 문서 메타데이터
                - similarity: 유사도 점수 (0-1, 높을수록 유사)
        """
        try:
            filter_dict = {"source": source}
            logger.info(f"소스별 검색: {source} (쿼리: {query[:50]}...)")
            return self.search_memory(query, top_k, filter_dict)
            
        except Exception as e:
            error_msg = f"소스별 검색 실패: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise Exception(error_msg) from e
    
    def search_memory_by_date_range(
        self, 
        query: str,
        start_date: str,
        end_date: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        날짜 범위 내에서 검색합니다.
        
        지정된 날짜 범위 내에 저장된 문서들 중에서만
        쿼리와 유사한 문서를 검색합니다.
        
        Args:
            query: 검색 쿼리 텍스트
            start_date: 시작 날짜 (ISO 형식 문자열, 예: "2024-01-01T00:00:00")
            end_date: 종료 날짜 (ISO 형식 문자열, 예: "2024-12-31T23:59:59")
            top_k: 반환할 최대 문서 개수 (기본값: 5)
            
        Returns:
            List[Dict[str, Any]]: 검색 결과 리스트
                각 딕셔너리는 다음 키를 포함:
                - id: 문서 ID
                - text: 문서 텍스트
                - metadata: 문서 메타데이터
                - similarity: 유사도 점수 (0-1, 높을수록 유사)
        """
        try:
            # Chroma의 where 필터 사용
            filter_dict = {
                "$and": [
                    {"timestamp": {"$gte": start_date}},
                    {"timestamp": {"$lte": end_date}}
                ]
            }
            
            logger.info(
                f"날짜 범위 검색: {start_date} ~ {end_date} "
                f"(쿼리: {query[:50]}...)"
            )
            return self.search_memory(query, top_k, filter_dict)
            
        except Exception as e:
            error_msg = f"날짜 범위 검색 실패: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise Exception(error_msg) from e
    
    def get_documents_by_metadata(
        self, 
        key: str, 
        value: Any
    ) -> List[Dict[str, Any]]:
        """
        특정 메타데이터 값으로 문서를 조회합니다.
        
        지정된 메타데이터 키-값 쌍을 가진 모든 문서를 반환합니다.
        검색이 아닌 필터링 조회이므로 유사도 점수는 포함되지 않습니다.
        
        Args:
            key: 메타데이터 키 (예: "source", "category" 등)
            value: 메타데이터 값
            
        Returns:
            List[Dict[str, Any]]: 문서 리스트
                각 딕셔너리는 다음 키를 포함:
                - id: 문서 ID
                - text: 문서 텍스트
                - metadata: 문서 메타데이터
        """
        try:
            results = self.collection.get(
                where={key: value}
            )
            
            documents: List[Dict[str, Any]] = []
            for i in range(len(results['ids'])):
                documents.append({
                    'id': results['ids'][i],
                    'text': results['documents'][i],
                    'metadata': results['metadatas'][i] if results['metadatas'] and i < len(results['metadatas']) else {}
                })
            
            logger.info(f"메타데이터 조회: {key}={value} → {len(documents)}개 문서")
            return documents
            
        except Exception as e:
            error_msg = f"메타데이터 조회 실패: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise Exception(error_msg) from e
    
    def cleanup_old_memories(
        self, 
        days_old: int = 30,
        keep_important: bool = True
    ) -> Dict[str, Any]:
        """
        오래된 메모리를 정리합니다.
        
        지정된 일수보다 오래된 문서를 삭제합니다.
        중요 문서는 선택적으로 보존할 수 있습니다.
        
        Args:
            days_old: 이 일수보다 오래된 문서 삭제 (기본값: 30)
            keep_important: 중요 문서 보존 여부 (기본값: True)
                True인 경우 메타데이터에 'important': True가 있는 문서는 삭제하지 않습니다.
            
        Returns:
            Dict[str, Any]: 정리 결과 딕셔너리
                - deleted: 삭제된 문서 개수
                - kept: 보존된 문서 개수
                - cutoff_date: 기준 날짜 (ISO 형식)
        """
        try:
            from datetime import timedelta
            
            # 기준 날짜 계산
            cutoff_date = (datetime.now() - timedelta(days=days_old)).isoformat()
            
            # 전체 문서 조회
            all_docs = self.get_all_documents()
            
            deleted = 0
            kept = 0
            
            for doc in all_docs:
                timestamp = doc['metadata'].get('timestamp', '')
                
                # 오래된 문서인지 확인
                if timestamp and timestamp < cutoff_date:
                    # 중요 문서 체크
                    if keep_important and doc['metadata'].get('important', False):
                        kept += 1
                        logger.debug(f"중요 문서 보존: {doc['id']}")
                        continue
                    
                    # 삭제
                    if self.delete_memory(doc['id']):
                        deleted += 1
                else:
                    kept += 1
            
            logger.info(f"메모리 정리 완료: {deleted}개 삭제, {kept}개 보존")
            
            return {
                "deleted": deleted,
                "kept": kept,
                "cutoff_date": cutoff_date
            }
            
        except Exception as e:
            error_msg = f"메모리 정리 실패: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise Exception(error_msg) from e
    
    def mark_as_important(self, document_id: str) -> bool:
        """
        문서를 중요로 표시합니다.
        
        문서의 메타데이터에 'important': True를 추가하여
        오래된 메모리 정리 시 보존되도록 합니다.
        
        Args:
            document_id: 중요 표시할 문서의 ID
            
        Returns:
            bool: 성공 여부 (True: 성공, False: 문서를 찾을 수 없음)
        """
        try:
            # 문서 조회
            doc = self.collection.get(ids=[document_id])
            
            if not doc['ids']:
                logger.warning(f"문서를 찾을 수 없음: {document_id}")
                return False
            
            # 메타데이터 업데이트
            metadata = doc['metadatas'][0].copy() if doc['metadatas'] and doc['metadatas'][0] else {}
            metadata['important'] = True
            
            # 문서 업데이트 (삭제 후 재생성)
            # Chroma DB는 직접 업데이트가 없으므로 삭제 후 재생성
            text = doc['documents'][0]
            embedding = self.embedding_generator.create_embedding(text)
            
            self.collection.delete(ids=[document_id])
            self.collection.add(
                ids=[document_id],
                documents=[text],
                embeddings=[embedding],
                metadatas=[metadata]
            )
            
            logger.info(f"중요 문서 표시: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"중요 표시 실패: {str(e)}")
            return False
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        상세한 메모리 통계를 반환합니다.
        
        Returns:
            Dict[str, Any]: 상세 통계 정보 딕셔너리
                - total_documents: 총 문서 개수
                - collection_name: 컬렉션 이름
                - by_source: 소스별 문서 개수
                - by_date: 기간별 문서 개수
                - avg_text_length: 평균 텍스트 길이
                - embedding_dimension: 임베딩 차원 수
                - cache_info: 임베딩 캐시 정보
        """
        try:
            from datetime import timedelta
            
            all_docs = self.get_all_documents()
            
            # 소스별 카운트
            sources: Dict[str, int] = {}
            for doc in all_docs:
                source = doc['metadata'].get('source', 'unknown')
                sources[source] = sources.get(source, 0) + 1
            
            # 날짜별 카운트
            now = datetime.now()
            
            date_counts = {
                'last_24h': 0,
                'last_7days': 0,
                'last_30days': 0,
                'older': 0
            }
            
            for doc in all_docs:
                timestamp_str = doc['metadata'].get('timestamp', '')
                if timestamp_str:
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        age = (now - timestamp.replace(tzinfo=None)).days
                        
                        if age < 1:
                            date_counts['last_24h'] += 1
                        if age < 7:
                            date_counts['last_7days'] += 1
                        if age < 30:
                            date_counts['last_30days'] += 1
                        else:
                            date_counts['older'] += 1
                    except (ValueError, AttributeError):
                        # 날짜 파싱 실패 시 older로 분류
                        date_counts['older'] += 1
                else:
                    date_counts['older'] += 1
            
            # 텍스트 길이 통계
            text_lengths = [len(doc['text']) for doc in all_docs]
            avg_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0
            
            stats = {
                'total_documents': len(all_docs),
                'collection_name': self.collection_name,
                'by_source': sources,
                'by_date': date_counts,
                'avg_text_length': int(avg_length),
                'embedding_dimension': self.embedding_generator.get_dimension(),
                'cache_info': self.embedding_generator.get_cache_info()
            }
            
            logger.info(f"상세 통계 조회: {stats['total_documents']}개 문서")
            return stats
            
        except Exception as e:
            error_msg = f"상세 통계 조회 실패: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise Exception(error_msg) from e
    
    def print_memory_dashboard(self) -> None:
        """
        메모리 통계를 대시보드 형태로 출력합니다.
        
        콘솔에 포맷팅된 메모리 통계 정보를 출력합니다.
        """
        try:
            stats = self.get_memory_stats()
            
            print("=" * 60)
            print("📊 메모리 시스템 대시보드")
            print("=" * 60)
            print(f"컬렉션: {stats['collection_name']}")
            print(f"총 문서 수: {stats['total_documents']:,}개")
            print(f"평균 텍스트 길이: {stats['avg_text_length']:,}자")
            print(f"임베딩 차원: {stats['embedding_dimension']}")
            print()
            print("📁 소스별 분포:")
            for source, count in stats['by_source'].items():
                print(f"  • {source}: {count}개")
            print()
            print("📅 기간별 분포:")
            print(f"  • 최근 24시간: {stats['by_date']['last_24h']}개")
            print(f"  • 최근 7일: {stats['by_date']['last_7days']}개")
            print(f"  • 최근 30일: {stats['by_date']['last_30days']}개")
            print(f"  • 그 이전: {stats['by_date']['older']}개")
            print()
            print("💾 캐시 정보:")
            cache_info = stats['cache_info']
            print(f"  • 캐시 크기: {cache_info.get('size', 0)}개")
            hit_rate = cache_info.get('hit_rate', 0)
            print(f"  • 적중률: {hit_rate:.1%}")
            print("=" * 60)
            
            logger.info("대시보드 출력 완료")
            
        except Exception as e:
            error_msg = f"대시보드 출력 실패: {str(e)}"
            logger.error(error_msg, exc_info=True)
            print(f"❌ 대시보드 출력 실패: {error_msg}")


# 테스트 코드 (주석으로 포함)
# from src.memory_manager import MemoryManager
# mm = MemoryManager("test_memory", "data/chroma_db")
# print(f"✓ 초기화 완료: {mm.collection_name}")
# print(f"✓ 문서 수: {mm.collection.count()}")
# 
# doc_id = mm.add_to_memory(
#     text="테슬라는 전기차 회사입니다",
#     metadata={"source": "test", "category": "company"}
# )
# print(f"✓ 문서 저장: {doc_id}")
# 
# results = mm.search_memory("전기차", top_k=3)
# for r in results:
#     print(f"유사도: {r['similarity']:.2f} | {r['text'][:50]}...")
# 
# print(f"✓ 전체 문서: {len(mm.get_all_documents())}")
# print(f"✓ 통계: {mm.get_statistics()}")
# mm.delete_memory(doc_id)
# print(f"✓ 삭제 후: {mm.collection.count()}개")
# 
# # 중복 체크 테스트
# # 첫 번째 저장
# id1 = mm.add_to_memory("테슬라는 전기차 회사입니다")
# print(f"✓ 문서 저장: {id1}")
# 
# # 중복 저장 시도
# id2 = mm.add_to_memory("테슬라는 전기차 회사입니다")
# print(f"✓ 중복 체크: {id1 == id2}")  # True여야 함
# 
# # 소스별 검색
# web_results = mm.search_memory_by_source("테슬라", "web_search")
# print(f"✓ 웹 검색 결과: {len(web_results)}개")
# 
# # 메타데이터 조회
# docs = mm.get_documents_by_metadata("category", "company")
# print(f"✓ 회사 카테고리: {len(docs)}개")
# 
# # 중요 문서 표시
# mm.mark_as_important(doc_id)
# 
# # 30일 이상 오래된 문서 정리
# result = mm.cleanup_old_memories(days_old=30)
# print(f"✓ 삭제: {result['deleted']}개, 보존: {result['kept']}개")
# 
# # 대시보드 출력
# mm.print_memory_dashboard()