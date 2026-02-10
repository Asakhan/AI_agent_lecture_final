"""
임베딩 생성 모듈

OpenAI API를 사용하여 텍스트를 벡터 임베딩으로 변환하는 기능을 제공합니다.
캐싱 및 재시도 로직을 포함하여 효율적이고 안정적인 임베딩 생성을 지원합니다.
"""

from openai import OpenAI
from dotenv import load_dotenv
import os
from typing import List, Dict, Optional, Any
import logging
import time

# 환경변수 로드
load_dotenv()

# 로깅 설정
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    OpenAI API를 사용하여 텍스트를 벡터 임베딩으로 변환하는 클래스.
    
    캐싱 기능을 통해 동일한 텍스트에 대한 중복 API 호출을 방지하고,
    재시도 로직을 통해 안정적인 임베딩 생성을 보장합니다.
    
    Attributes:
        client (OpenAI): OpenAI API 클라이언트 인스턴스
        model (str): 사용할 임베딩 모델명
        cache (Dict[str, List[float]]): 텍스트-임베딩 캐시 딕셔너리
        embedding_count (int): 생성된 임베딩의 총 개수
        api_call_count (int): API 호출 횟수
        cache_hit_count (int): 캐시 히트 횟수
    """
    
    def __init__(self) -> None:
        """
        EmbeddingGenerator 인스턴스를 초기화합니다.
        
        OpenAI 클라이언트를 초기화하고, 캐시 딕셔너리와 통계 카운터를 설정합니다.
        환경변수에서 OPENAI_API_KEY를 읽어옵니다.
        
        Raises:
            ValueError: OPENAI_API_KEY가 설정되지 않은 경우
        """
        # OpenAI 클라이언트 초기화
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY 환경변수가 설정되지 않았습니다. "
                ".env 파일에 OPENAI_API_KEY를 추가해주세요."
            )
        
        self.client = OpenAI(api_key=api_key)
        self.model = "text-embedding-3-small"
        
        # 캐시 딕셔너리 초기화
        self.cache: Dict[str, List[float]] = {}
        
        # 임베딩 생성 카운터 초기화
        self.embedding_count = 0
        self.api_call_count = 0
        self.cache_hit_count = 0
        self.cache_hits = 0
        self.total_requests = 0
        
        logger.info(f"EmbeddingGenerator 초기화 완료 (모델: {self.model})")
    
    def create_embedding(self, text: str) -> List[float]:
        """
        단일 텍스트를 1536차원 벡터로 변환합니다.
        
        캐시에 해당 텍스트의 임베딩이 있으면 캐시에서 반환하고,
        없으면 OpenAI API를 호출하여 임베딩을 생성한 후 캐시에 저장합니다.
        
        Args:
            text (str): 임베딩으로 변환할 텍스트
            
        Returns:
            List[float]: 1536차원 임베딩 벡터
            
        Raises:
            ValueError: 텍스트가 비어있는 경우
            Exception: API 호출 실패 시 (재시도 후에도 실패한 경우)
        """
        if not text or not text.strip():
            raise ValueError("임베딩을 생성할 텍스트가 비어있습니다.")
        
        # 요청 카운터 증가
        self.total_requests += 1
        
        # 캐시 확인
        if text in self.cache:
            self.cache_hit_count += 1
            self.cache_hits += 1
            logger.debug(f"캐시에서 임베딩 반환: {text[:50]}...")
            return self.cache[text]
        
        # API 호출 (재시도 로직 포함)
        embedding = self._call_api_with_retry(text)
        
        # 결과 캐싱
        self.cache[text] = embedding
        self.embedding_count += 1
        
        logger.info(f"임베딩 생성 완료: {text[:50]}...")
        return embedding
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        여러 텍스트를 배치로 처리하여 임베딩을 생성합니다.
        
        여러 텍스트를 효율적으로 처리하기 위해 배치 API 호출을 사용합니다.
        각 텍스트에 대해 캐시를 확인하고, 캐시되지 않은 텍스트만 API로 전송합니다.
        
        Args:
            texts (List[str]): 임베딩으로 변환할 텍스트 리스트
            
        Returns:
            List[List[float]]: 각 텍스트에 대한 1536차원 임베딩 벡터 리스트
            
        Raises:
            ValueError: 텍스트 리스트가 비어있는 경우
        """
        if not texts:
            raise ValueError("임베딩을 생성할 텍스트 리스트가 비어있습니다.")
        
        logger.info(f"배치 임베딩 생성 시작: {len(texts)}개 텍스트")
        
        # 캐시되지 않은 텍스트와 인덱스 추적
        uncached_texts: List[str] = []
        uncached_indices: List[int] = []
        results: List[Optional[List[float]]] = [None] * len(texts)
        
        # 캐시 확인 및 결과 채우기
        for idx, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(f"인덱스 {idx}의 텍스트가 비어있습니다.")
            
            if text in self.cache:
                results[idx] = self.cache[text]
                self.cache_hit_count += 1
            else:
                uncached_texts.append(text)
                uncached_indices.append(idx)
        
        # 캐시되지 않은 텍스트에 대해 API 호출
        if uncached_texts:
            logger.info(f"API 호출 필요: {len(uncached_texts)}개 텍스트")
            embeddings = self._call_batch_api_with_retry(uncached_texts)
            
            # 결과를 results에 채우고 캐시에 저장
            for idx, embedding in zip(uncached_indices, embeddings):
                text = texts[idx]
                results[idx] = embedding
                self.cache[text] = embedding
                self.embedding_count += 1
        
        # None이 없는지 확인 (타입 체크를 위해)
        final_results: List[List[float]] = []
        for result in results:
            if result is None:
                raise RuntimeError("임베딩 생성 중 오류가 발생했습니다.")
            final_results.append(result)
        
        logger.info(f"배치 임베딩 생성 완료: {len(final_results)}개")
        return final_results
    
    def get_dimension(self) -> int:
        """
        임베딩 벡터의 차원 수를 반환합니다.
        
        Returns:
            int: 임베딩 차원 수 (1536)
        """
        return 1536
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        캐시 통계 정보를 반환합니다.
        
        Returns:
            Dict[str, Any]: 캐시 통계 딕셔너리
                - size: 캐시에 저장된 임베딩 개수
                - embedding_count: 생성된 총 임베딩 개수
                - api_call_count: API 호출 횟수
                - cache_hit_count: 캐시 히트 횟수
                - hit_rate: 캐시 적중률 (0-1)
        """
        hit_rate = (
            self.cache_hits / max(self.total_requests, 1)
            if hasattr(self, 'cache_hits') and hasattr(self, 'total_requests')
            else 0.0
        )
        
        return {
            "size": len(self.cache),
            "embedding_count": self.embedding_count,
            "api_call_count": self.api_call_count,
            "cache_hit_count": self.cache_hit_count,
            "hit_rate": hit_rate,
        }
    
    def _call_api_with_retry(self, text: str, max_retries: int = 3) -> List[float]:
        """
        재시도 로직을 포함하여 OpenAI API를 호출합니다.
        
        지수 백오프를 사용하여 재시도 간격을 점진적으로 증가시킵니다.
        
        Args:
            text (str): 임베딩으로 변환할 텍스트
            max_retries (int): 최대 재시도 횟수 (기본값: 3)
            
        Returns:
            List[float]: 1536차원 임베딩 벡터
            
        Raises:
            Exception: 모든 재시도 실패 후 발생
        """
        base_delay = 1.0  # 초기 지연 시간 (초)
        
        for attempt in range(max_retries):
            try:
                self.api_call_count += 1
                logger.debug(f"API 호출 시도 {attempt + 1}/{max_retries}: {text[:50]}...")
                
                response = self.client.embeddings.create(
                    model=self.model,
                    input=text
                )
                
                embedding = response.data[0].embedding
                logger.debug(f"API 호출 성공: {text[:50]}...")
                return embedding
                
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # 지수 백오프
                    logger.warning(
                        f"API 호출 실패 (시도 {attempt + 1}/{max_retries}): {str(e)}. "
                        f"{delay:.1f}초 후 재시도합니다."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"API 호출 최종 실패: {str(e)}")
                    raise Exception(
                        f"임베딩 생성 실패 (최대 재시도 횟수 초과): {str(e)}"
                    ) from e
        
        # 이 코드는 실행되지 않아야 하지만 타입 체커를 위해 추가
        raise RuntimeError("예상치 못한 오류가 발생했습니다.")
    
    def _call_batch_api_with_retry(
        self, 
        texts: List[str], 
        max_retries: int = 3
    ) -> List[List[float]]:
        """
        재시도 로직을 포함하여 여러 텍스트에 대한 OpenAI API를 배치로 호출합니다.
        
        지수 백오프를 사용하여 재시도 간격을 점진적으로 증가시킵니다.
        
        Args:
            texts (List[str]): 임베딩으로 변환할 텍스트 리스트
            max_retries (int): 최대 재시도 횟수 (기본값: 3)
            
        Returns:
            List[List[float]]: 각 텍스트에 대한 1536차원 임베딩 벡터 리스트
            
        Raises:
            Exception: 모든 재시도 실패 후 발생
        """
        base_delay = 1.0  # 초기 지연 시간 (초)
        
        for attempt in range(max_retries):
            try:
                self.api_call_count += 1
                logger.debug(
                    f"배치 API 호출 시도 {attempt + 1}/{max_retries}: "
                    f"{len(texts)}개 텍스트"
                )
                
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts
                )
                
                embeddings = [item.embedding for item in response.data]
                logger.debug(f"배치 API 호출 성공: {len(embeddings)}개 임베딩")
                return embeddings
                
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # 지수 백오프
                    logger.warning(
                        f"배치 API 호출 실패 (시도 {attempt + 1}/{max_retries}): {str(e)}. "
                        f"{delay:.1f}초 후 재시도합니다."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"배치 API 호출 최종 실패: {str(e)}")
                    raise Exception(
                        f"배치 임베딩 생성 실패 (최대 재시도 횟수 초과): {str(e)}"
                    ) from e
        
        # 이 코드는 실행되지 않아야 하지만 타입 체커를 위해 추가
        raise RuntimeError("예상치 못한 오류가 발생했습니다.")
