"""
웹 검색 도구 모듈

Tavily API를 사용하여 웹 검색 기능을 제공합니다.
검색 결과를 구조화된 형태로 반환하며, 검색 쿼리 최적화 기능을 포함합니다.
"""

import os
import time
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv

# 설정 import
from config.settings import (
    TAVILY_ADVANCED_SEARCH_DEPTH,
    QUERY_REMOVE_PHRASES,
    TIME_INDICATOR_PHRASES,
)

# 로깅 설정
logger = logging.getLogger(__name__)

# 환경변수 로드
load_dotenv()

# TavilyClient 캐시 (lazy import용)
_tavily_client: Optional[Any] = None


def _get_tavily_client():
    """
    TavilyClient 인스턴스를 lazy import하여 반환합니다.
    
    Returns:
        TavilyClient: Tavily API 클라이언트 인스턴스
        
    Raises:
        ImportError: tavily 패키지가 설치되지 않은 경우
        ValueError: TAVILY_API_KEY 환경변수가 설정되지 않은 경우
    """
    global _tavily_client
    
    if _tavily_client is not None:
        return _tavily_client
    
    try:
        from tavily import TavilyClient
    except ImportError:
        raise ImportError(
            "tavily 패키지가 설치되지 않았습니다. "
            "다음 명령어로 설치해주세요: pip install tavily-python"
        )
    
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY 환경변수가 설정되지 않았습니다. "
            ".env 파일에 TAVILY_API_KEY를 추가해주세요."
        )
    
    _tavily_client = TavilyClient(api_key=api_key)
    return _tavily_client


@dataclass
class SearchResult:
    """
    검색 결과를 담는 데이터 클래스
    
    Attributes:
        query: 검색 쿼리 문자열
        answer: AI가 생성한 요약 답변 (선택적)
        results: 검색 결과 리스트 (각 결과는 딕셔너리 형태)
        sources: 출처 URL 리스트
        search_time: 검색에 소요된 시간 (초)
        raw_response: Tavily API의 원본 응답 (디버깅용)
    """
    query: str
    answer: Optional[str] = None
    results: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    search_time: float = 0.0
    raw_response: Optional[Dict[str, Any]] = None
    
    @property
    def result_count(self) -> int:
        """
        검색 결과 개수를 반환합니다.
        
        Returns:
            int: 검색 결과 개수
        """
        return len(self.results)
    
    @property
    def has_answer(self) -> bool:
        """
        AI 요약 답변이 존재하는지 여부를 반환합니다.
        
        Returns:
            bool: answer가 존재하면 True, None이거나 빈 문자열이면 False
        """
        return self.answer is not None and self.answer.strip() != ""
    
    def get_top_results(self, n: int = 3) -> List[Dict[str, Any]]:
        """
        상위 n개의 검색 결과를 반환합니다.
        
        Args:
            n: 반환할 결과 개수 (기본값: 3)
            
        Returns:
            List[Dict[str, Any]]: 상위 n개의 검색 결과 리스트
        """
        return self.results[:n]
    
    def get_sources_as_string(self, separator: str = "\n") -> str:
        """
        출처 URL 리스트를 문자열로 반환합니다.
        
        Args:
            separator: URL 사이에 사용할 구분자 (기본값: 줄바꿈)
            
        Returns:
            str: 출처 URL들을 구분자로 연결한 문자열
        """
        return separator.join(self.sources)


def tavily_search(
    query: str,
    api_key: Optional[str] = None,
    search_depth: str = "basic",
    include_answer: bool = True,
    include_raw_content: bool = False,
    max_results: int = 5,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
) -> SearchResult:
    """
    Tavily API를 사용하여 웹 검색을 수행합니다.
    
    Args:
        query: 검색할 쿼리 문자열
        api_key: Tavily API 키 (없으면 환경변수 TAVILY_API_KEY 사용)
        search_depth: 검색 깊이 ("basic" 또는 "advanced")
        include_answer: AI 요약 답변 포함 여부
        include_raw_content: 원본 콘텐츠 포함 여부
        max_results: 최대 검색 결과 개수 (1-10 범위)
        include_domains: 검색에 포함할 도메인 리스트 (선택적)
        exclude_domains: 검색에서 제외할 도메인 리스트 (선택적)
    
    Returns:
        SearchResult: 검색 결과를 담은 SearchResult 객체
        
    Raises:
        ValueError: query가 비어있거나, API 키가 없거나, max_results가 범위를 벗어난 경우
        ImportError: tavily 패키지가 설치되지 않은 경우
        Exception: Tavily API 호출 중 오류가 발생한 경우
        
    Example:
        >>> result = tavily_search("Python 웹 프레임워크")
        >>> print(f"검색 결과: {result.result_count}개")
        >>> print(result.answer)
        >>> for source in result.sources:
        ...     print(source)
    """
    try:
        # 1. 입력 검증
        if not query or not query.strip():
            raise ValueError("검색 쿼리는 비어있을 수 없습니다.")
        
        query = query.strip()
        
        # 2. API 키 설정 (파라미터 → 환경변수 순서)
        if api_key is None:
            api_key = os.getenv("TAVILY_API_KEY")
        
        if not api_key:
            raise ValueError(
                "Tavily API 키가 설정되지 않았습니다. "
                "api_key 파라미터를 제공하거나 TAVILY_API_KEY 환경변수를 설정해주세요."
            )
        
        # 3. max_results 범위 검증
        if not (1 <= max_results <= 10):
            raise ValueError("max_results는 1과 10 사이의 값이어야 합니다.")
        
        # 4. 로깅: 검색 시작
        logger.info(
            f"Tavily 검색 시작 - 쿼리: '{query}', "
            f"depth: {search_depth}, max_results: {max_results}"
        )
        
        # 5. Tavily 클라이언트 생성
        try:
            from tavily import TavilyClient
        except ImportError:
            raise ImportError(
                "tavily 패키지가 설치되지 않았습니다. "
                "다음 명령어로 설치해주세요: pip install tavily-python"
            )
        
        client = TavilyClient(api_key=api_key)
        
        # 6. 검색 파라미터 구성
        search_params: Dict[str, Any] = {
            "query": query,
            "search_depth": search_depth,
            "include_answer": include_answer,
            "include_raw_content": include_raw_content,
            "max_results": max_results,
        }
        
        if include_domains:
            search_params["include_domains"] = include_domains
        
        if exclude_domains:
            search_params["exclude_domains"] = exclude_domains
        
        # 7. 검색 실행 및 시간 측정
        start_time = time.time()
        raw_response = client.search(**search_params)
        search_time = time.time() - start_time
        
        # 8. 결과 파싱
        results = raw_response.get("results", [])
        answer = raw_response.get("answer") if include_answer else None
        
        # results에서 url 추출하여 sources 리스트 생성
        sources = []
        for result in results:
            url = result.get("url")
            if url:
                sources.append(url)
        
        # 9. 로깅: 검색 완료
        logger.info(
            f"Tavily 검색 완료 - 결과: {len(results)}개, "
            f"소요 시간: {search_time:.2f}초"
        )
        
        # 10. SearchResult 객체 생성 및 반환
        return SearchResult(
            query=query,
            answer=answer,
            results=results,
            sources=sources,
            search_time=search_time,
            raw_response=raw_response,
        )
    
    except (ValueError, ImportError):
        # 입력 검증 오류나 ImportError는 그대로 re-raise
        raise
    except Exception as e:
        # 기타 예외는 로깅 후 re-raise
        logger.error(f"Tavily 검색 중 오류 발생: {str(e)}", exc_info=True)
        raise


def tavily_search_with_context(
    query: str,
    context: Optional[str] = None,
    api_key: Optional[str] = None,
    max_results: int = 5,
) -> SearchResult:
    """
    컨텍스트를 포함하여 Tavily API로 웹 검색을 수행합니다.
    
    컨텍스트가 제공되면 검색 쿼리 앞에 추가하여 더 정확한 검색 결과를 얻습니다.
    컨텍스트 검색은 심층 검색(advanced)을 사용합니다.
    
    Args:
        query: 검색할 쿼리 문자열
        context: 검색에 포함할 컨텍스트 정보 (선택적)
        api_key: Tavily API 키 (없으면 환경변수 TAVILY_API_KEY 사용)
        max_results: 최대 검색 결과 개수 (1-10 범위, 기본값: 5)
    
    Returns:
        SearchResult: 검색 결과를 담은 SearchResult 객체
        
    Raises:
        ValueError: query가 비어있거나, API 키가 없거나, max_results가 범위를 벗어난 경우
        ImportError: tavily 패키지가 설치되지 않은 경우
        Exception: Tavily API 호출 중 오류가 발생한 경우
        
    Example:
        >>> result = tavily_search_with_context(
        ...     query="성능 최적화",
        ...     context="Python 웹 프레임워크"
        ... )
        >>> print(f"검색 결과: {result.result_count}개")
        >>> print(result.query)  # "Python 웹 프레임워크 - 성능 최적화"
    """
    # 컨텍스트가 있으면 쿼리 앞에 추가
    if context and context.strip():
        optimized_query = f"{context.strip()} - {query.strip()}"
    else:
        optimized_query = query.strip()
    
    # advanced 검색 깊이로 tavily_search 호출
    return tavily_search(
        query=optimized_query,
        api_key=api_key,
        search_depth=TAVILY_ADVANCED_SEARCH_DEPTH,
        include_answer=True,
        max_results=max_results,
    )


def format_search_result_for_llm(search_result: SearchResult) -> str:
    """
    검색 결과를 LLM 컨텍스트용 마크다운 문자열로 변환합니다.
    
    Args:
        search_result: 포맷팅할 SearchResult 객체
    
    Returns:
        str: 마크다운 형식으로 포맷팅된 검색 결과 문자열
        
    Example:
        >>> result = tavily_search("Python 웹 프레임워크")
        >>> formatted = format_search_result_for_llm(result)
        >>> print(formatted)
    """
    lines = []
    
    # 헤더
    lines.append(f"## 웹 검색 결과: '{search_result.query}'")
    lines.append(f"검색 시간: {search_result.search_time:.2f}초")
    lines.append("")
    
    # 요약 섹션 (has_answer일 때만)
    if search_result.has_answer:
        lines.append("### 📋 요약")
        lines.append(search_result.answer)
        lines.append("")
    
    # 상세 검색 결과
    if search_result.results:
        lines.append("### 📰 상세 검색 결과")
        lines.append("")
        
        for idx, result in enumerate(search_result.results, 1):
            title = result.get("title", "제목 없음")
            url = result.get("url", "")
            score = result.get("score", 0.0)
            content = result.get("content", "")
            
            # content가 300자 초과시 자르기
            if len(content) > 300:
                content = content[:300] + "..."
            
            lines.append(f"**[{idx}] {title}**")
            lines.append(f"- 출처: {url}")
            if score:
                lines.append(f"- 관련도: {score:.2f}")
            if content:
                lines.append(f"- 내용: {content}")
            lines.append("")
    
    # 참고 출처
    if search_result.sources:
        lines.append("### 📚 참고 출처")
        for idx, source in enumerate(search_result.sources, 1):
            lines.append(f"{idx}. {source}")
        lines.append("")
    
    return "\n".join(lines)


def optimize_search_query(user_input: str) -> str:
    """
    사용자 입력을 검색에 최적화된 쿼리로 변환합니다.
    
    불필요한 한국어 표현을 제거하고, 시간 표현을 처리하여
    검색 품질을 향상시킵니다.
    
    Args:
        user_input: 최적화할 사용자 입력 문자열
    
    Returns:
        str: 최적화된 검색 쿼리
        
    Example:
        >>> optimize_search_query("Python 웹 프레임워크에 대해 알려줘")
        'Python 웹 프레임워크'
        
        >>> optimize_search_query("최신 AI 기술")
        'AI 기술 2025'
    """
    if not user_input:
        return ""
    
    # 1. strip() 처리
    query = user_input.strip()
    
    # 2. 불필요한 한국어 표현 제거
    # config/settings.py의 QUERY_REMOVE_PHRASES 사용
    remove_phrases = QUERY_REMOVE_PHRASES + [
        "검색해줘",
        "조사해줘",
        "뭐야",
        "무엇인가요",
        "무엇인지",
        "무엇인가",
        "무엇",
    ]
    
    for phrase in remove_phrases:
        query = query.replace(phrase, "")
    
    # 3. 시간 표현 처리
    current_year = datetime.now().year
    time_phrases = TIME_INDICATOR_PHRASES
    
    has_time_indicator = False
    for phrase in time_phrases:
        if phrase in query:
            query = query.replace(phrase, "")
            has_time_indicator = True
    
    # 시간 표현이 있었으면 현재 연도 추가
    if has_time_indicator:
        query = f"{query} {current_year}".strip()
    
    # 4. 연속 공백 제거
    query = " ".join(query.split())
    
    return query
