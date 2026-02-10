"""
검색 에이전트 모듈

SearchAgent는 웹 검색 도구들을 오케스트레이션하는 상위 레벨 클래스입니다.
src/tools/web_search.py의 함수들을 조합하여 사용합니다.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from src.memory_manager import MemoryManager
from src.tools.web_search import (
    SearchResult,
    tavily_search,
    tavily_search_with_context,
    format_search_result_for_llm,
    optimize_search_query,
)

# 로깅 설정
logger = logging.getLogger(__name__)


class SearchAgent:
    """
    웹 검색 도구들을 오케스트레이션하는 상위 레벨 클래스.
    
    Tavily API를 사용한 웹 검색 기능을 제공하며, 쿼리 최적화,
    검색 히스토리 관리 등의 기능을 포함합니다.
    
    Attributes:
        max_results (int): 기본 검색 결과 개수 (1-10 범위)
        optimize_queries (bool): 검색 쿼리 최적화 여부
        search_history (List[SearchResult]): 검색 히스토리 리스트
        
    Example:
        >>> agent = SearchAgent(max_results=5, optimize_queries=True)
        >>> result = agent.search("AI trends 2024")
        >>> print(f"검색 결과: {result.result_count}개")
    """
    
    def __init__(
        self,
        max_results: int = 5,
        optimize_queries: bool = True,
        memory_manager: Optional[MemoryManager] = None
    ):
        """
        SearchAgent 인스턴스를 초기화합니다.
        
        Args:
            max_results: 기본 검색 결과 개수 (1-10 범위, 기본값: 5)
            optimize_queries: 검색 쿼리 최적화 여부 (기본값: True)
            memory_manager: 메모리 관리자 (선택적, 기본값: None)
            
        Raises:
            ValueError: max_results가 1-10 범위를 벗어난 경우
        """
        # max_results 범위 검증
        if not (1 <= max_results <= 10):
            raise ValueError("max_results는 1과 10 사이의 값이어야 합니다.")
        
        # 속성 초기화
        self.max_results = max_results
        self.optimize_queries = optimize_queries
        self.search_history: List[SearchResult] = []
        
        # 메모리 관리자 추가
        self.memory_manager = memory_manager
        
        # 로깅
        logger.info(
            f"SearchAgent 초기화 - max_results: {max_results}, "
            f"optimize_queries: {optimize_queries}"
        )
        
        if self.memory_manager:
            logger.info("SearchAgent initialized with memory")
        else:
            logger.info("SearchAgent initialized without memory")
    
    def search(
        self,
        query: str,
        search_depth: str = "basic",
        include_answer: bool = True,
        max_results: Optional[int] = None,
        optimize_query: Optional[bool] = None
    ) -> SearchResult:
        """
        웹 검색을 수행합니다.
        
        Args:
            query: 검색할 쿼리 문자열
            search_depth: 검색 깊이 ("basic" 또는 "advanced", 기본값: "basic")
            include_answer: AI 요약 답변 포함 여부 (기본값: True)
            max_results: 최대 검색 결과 개수 (None이면 self.max_results 사용)
            optimize_query: 쿼리 최적화 여부 (None이면 self.optimize_queries 사용)
        
        Returns:
            SearchResult: 검색 결과 객체
            
        Raises:
            ValueError: query가 비어있는 경우
            
        Example:
            >>> agent = SearchAgent()
            >>> result = agent.search("Python web framework")
            >>> print(f"검색 결과: {result.result_count}개")
        """
        # 입력 검증
        if not query or not query.strip():
            raise ValueError("검색 쿼리는 비어있을 수 없습니다.")
        
        original_query = query.strip()
        
        # optimize_query 처리
        should_optimize = optimize_query if optimize_query is not None else self.optimize_queries
        
        if should_optimize:
            optimized_query = optimize_search_query(original_query)
            if optimized_query != original_query:
                logger.info(
                    f"쿼리 최적화: '{original_query}' -> '{optimized_query}'"
                )
            query = optimized_query
        else:
            query = original_query
        
        # max_results 처리
        results_count = max_results if max_results is not None else self.max_results
        
        # tavily_search() 호출
        result = tavily_search(
            query=query,
            search_depth=search_depth,
            include_answer=include_answer,
            max_results=results_count,
        )
        
        # 원본 쿼리를 result.query에 저장 (최적화된 쿼리 대신)
        result.query = original_query
        
        # search_history에 추가
        self.search_history.append(result)
        
        # 결과 반환
        return result
    
    def search_with_memory(
        self,
        query: str,
        use_memory: bool = True,
        save_to_memory: bool = True,
        memory_threshold: int = 3,
        max_results: int = 5
    ) -> Dict[str, Any]:
        """
        메모리를 활용한 지능형 검색을 수행합니다.
        
        메모리에서 관련 정보를 먼저 검색하고, 필요시 웹 검색을 수행합니다.
        웹 검색 결과는 선택적으로 메모리에 저장할 수 있습니다.
        
        Args:
            query: 검색 쿼리 문자열
            use_memory: 메모리 검색 사용 여부 (기본값: True)
            save_to_memory: 웹 검색 결과를 메모리에 저장할지 여부 (기본값: True)
            memory_threshold: 웹 검색을 수행할 최소 메모리 결과 수 (기본값: 3)
                메모리 결과가 이 값보다 적으면 웹 검색을 수행합니다.
            max_results: 최대 반환 결과 수 (기본값: 5)
        
        Returns:
            Dict[str, Any]: 검색 결과 딕셔너리
                - query: 원본 검색 쿼리
                - memory_results: 메모리에서 검색된 결과 리스트
                - web_results: 웹에서 검색된 결과 리스트
                - merged_results: 메모리와 웹 결과를 병합한 리스트
                - source_summary: 결과 출처 요약
                    - from_memory: 메모리에서 온 결과 수
                    - from_web: 웹에서 온 결과 수
                    - total: 전체 결과 수
        
        Raises:
            ValueError: query가 비어있는 경우
        
        Example:
            >>> agent = SearchAgent(memory_manager=mm)
            >>> result = agent.search_with_memory("테슬라")
            >>> print(f"메모리 결과: {len(result['memory_results'])}개")
            >>> print(f"웹 결과: {len(result['web_results'])}개")
        """
        # 입력 검증
        if not query or not query.strip():
            raise ValueError("검색 쿼리는 비어있을 수 없습니다.")
        
        query = query.strip()
        logger.info(f"Search with memory: {query}")
        
        # 결과 저장 구조
        results: Dict[str, Any] = {
            "query": query,
            "memory_results": [],
            "web_results": [],
            "merged_results": [],
            "source_summary": {
                "from_memory": 0,
                "from_web": 0,
                "total": 0
            }
        }
        
        # [1단계] 메모리 검색
        if use_memory and self.memory_manager:
            try:
                memory_results = self.memory_manager.search_memory(
                    query=query,
                    top_k=max_results
                )
                results["memory_results"] = memory_results
                results["source_summary"]["from_memory"] = len(memory_results)
                logger.info(f"Found {len(memory_results)} results in memory")
            except Exception as e:
                logger.error(f"Memory search failed: {e}")
                memory_results = []
        else:
            memory_results = []
            if not use_memory:
                logger.debug("Memory search disabled")
            elif not self.memory_manager:
                logger.debug("Memory manager not available")
        
        # [2단계] 웹 검색 필요성 판단
        need_web_search = len(memory_results) < memory_threshold
        
        if need_web_search:
            logger.info("Memory results insufficient, performing web search")
            
            try:
                # 웹 검색 수행 (기존 tavily_search 함수 사용)
                web_response = tavily_search(
                    query=query,
                    max_results=max_results
                )
                web_results = web_response.results  # SearchResult 객체의 results 속성
                results["web_results"] = web_results
                results["source_summary"]["from_web"] = len(web_results)
                
                logger.info(f"Found {len(web_results)} results from web")
                
                # [3단계] 웹 결과를 메모리에 저장
                if save_to_memory and self.memory_manager:
                    self._save_to_memory(web_results, query)
                    
            except Exception as e:
                logger.error(f"Web search failed: {e}")
                web_results = []
        else:
            logger.info("Sufficient results in memory, skipping web search")
            web_results = []
        
        # [4단계] 결과 병합
        results["merged_results"] = self._merge_results(
            memory_results,
            web_results
        )
        
        # [5단계] 통계 업데이트
        results["source_summary"] = {
            "from_memory": len(memory_results),
            "from_web": len(web_results),
            "total": len(results["merged_results"])
        }
        
        logger.info(f"Search complete: {results['source_summary']}")
        
        return results
    
    def _save_to_memory(
        self,
        web_results: List[Dict[str, Any]],
        query: str
    ) -> int:
        """
        웹 검색 결과를 메모리에 저장합니다.
        
        각 웹 검색 결과를 메모리 관리자에 저장하며,
        중복 체크를 통해 이미 저장된 내용은 건너뜁니다.
        
        Args:
            web_results: Tavily 검색 결과 리스트 (각 결과는 딕셔너리)
            query: 원본 검색 쿼리
        
        Returns:
            int: 저장된 문서 수
        """
        if not self.memory_manager:
            return 0
        
        saved_count = 0
        
        for result in web_results:
            try:
                # 텍스트 추출
                text = result.get("content", "")
                if not text:
                    continue
                
                # 메타데이터 구성
                metadata = {
                    "source": "web_search",
                    "query": query,
                    "url": result.get("url", ""),
                    "title": result.get("title", ""),
                    "score": result.get("score", 0.0),
                    "saved_at": datetime.now().isoformat()
                }
                
                # 메모리에 저장 (중복 체크 활성화)
                doc_id = self.memory_manager.add_to_memory(
                    text=text,
                    metadata=metadata,
                    check_duplicate=True  # 중복 저장 방지
                )
                
                saved_count += 1
                logger.debug(f"Saved to memory: {doc_id}")
                
            except Exception as e:
                logger.error(f"Failed to save result: {e}")
                continue
        
        logger.info(f"Saved {saved_count}/{len(web_results)} results to memory")
        return saved_count
    
    def _merge_results(
        self,
        memory_results: List[Dict[str, Any]],
        web_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        메모리와 웹 결과를 하나로 병합합니다.
        
        메모리 결과와 웹 결과를 통합하고, 중복을 제거한 후
        유사도/점수 기준으로 정렬합니다.
        
        Args:
            memory_results: 메모리 검색 결과 리스트
            web_results: 웹 검색 결과 리스트
        
        Returns:
            List[Dict[str, Any]]: 병합된 결과 리스트 (출처 정보 포함)
        """
        merged: List[Dict[str, Any]] = []
        
        # [1] 메모리 결과 추가
        for r in memory_results:
            merged.append({
                "content": r.get("text", ""),
                "source": "memory",
                "similarity": r.get("similarity", 0.0),
                "metadata": r.get("metadata", {}),
                "provenance": {
                    "retrieved_from": "memory",
                    "original_source": r.get("metadata", {}).get("source", "unknown"),
                    "original_query": r.get("metadata", {}).get("query", ""),
                    "stored_at": r.get("metadata", {}).get("timestamp", ""),
                    "confidence": r.get("similarity", 0.0)
                }
            })
        
        # [2] 웹 결과 추가
        for r in web_results:
            merged.append({
                "content": r.get("content", ""),
                "source": "web",
                "url": r.get("url", ""),
                "title": r.get("title", ""),
                "score": r.get("score", 0.0),
                "provenance": {
                    "retrieved_from": "web",
                    "url": r.get("url", ""),
                    "fetched_at": datetime.now().isoformat(),
                    "tavily_score": r.get("score", 0.0)
                }
            })
        
        # [3] 중복 제거 (옵션)
        merged = self._remove_duplicates(merged)
        
        # [4] 정렬 (유사도/점수 기준)
        def sort_key(item: Dict[str, Any]) -> float:
            if item.get("source") == "memory":
                return item.get("similarity", 0.0)
            else:
                return item.get("score", 0.0)
        
        merged = sorted(merged, key=sort_key, reverse=True)
        
        return merged
    
    def _remove_duplicates(
        self,
        results: List[Dict[str, Any]],
        similarity_threshold: float = 0.95
    ) -> List[Dict[str, Any]]:
        """
        중복 결과를 제거합니다.
        
        매우 유사한 내용이 이미 있는지 확인하여 중복을 제거합니다.
        
        Args:
            results: 병합된 결과 리스트
            similarity_threshold: 중복 판단 임계값 (현재는 사용하지 않음, 향후 확장용)
        
        Returns:
            List[Dict[str, Any]]: 중복이 제거된 결과 리스트
        """
        if not results:
            return []
        
        unique_results: List[Dict[str, Any]] = []
        seen_contents: List[str] = []
        
        for result in results:
            content = result.get("content", "")
            
            if not content:
                continue
            
            # 매우 유사한 내용이 이미 있는지 확인
            is_duplicate = False
            for seen in seen_contents:
                # 간단한 중복 체크 (문자열 포함 여부)
                # 더 정교한 유사도 체크는 향후 EmbeddingGenerator를 사용할 수 있음
                if content in seen or seen in content:
                    # 완전히 동일하거나 한 내용이 다른 내용에 포함된 경우
                    if len(content) > 50 and len(seen) > 50:
                        # 긴 텍스트의 경우 더 엄격한 체크
                        if content == seen:
                            is_duplicate = True
                            break
                    else:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_results.append(result)
                seen_contents.append(content)
        
        removed_count = len(results) - len(unique_results)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} duplicates")
        
        return unique_results
    
    def search_with_context(
        self,
        query: str,
        context: str,
        max_results: Optional[int] = None
    ) -> SearchResult:
        """
        컨텍스트를 포함하여 웹 검색을 수행합니다.
        
        Args:
            query: 검색할 쿼리 문자열
            context: 검색에 포함할 컨텍스트 정보
            max_results: 최대 검색 결과 개수 (None이면 self.max_results 사용)
        
        Returns:
            SearchResult: 검색 결과 객체
            
        Raises:
            ValueError: query나 context가 비어있는 경우
            
        Example:
            >>> agent = SearchAgent()
            >>> result = agent.search_with_context(
            ...     query="성능 최적화",
            ...     context="Python 웹 프레임워크"
            ... )
        """
        if not query or not query.strip():
            raise ValueError("검색 쿼리는 비어있을 수 없습니다.")
        
        if not context or not context.strip():
            raise ValueError("컨텍스트는 비어있을 수 없습니다.")
        
        # max_results 처리
        results_count = max_results if max_results is not None else self.max_results
        
        # tavily_search_with_context() 호출
        result = tavily_search_with_context(
            query=query.strip(),
            context=context.strip(),
            max_results=results_count,
        )
        
        # search_history에 추가
        self.search_history.append(result)
        
        # 결과 반환
        return result
    
    def format_for_llm(self, search_result: SearchResult) -> str:
        """
        검색 결과를 LLM 컨텍스트용 마크다운 문자열로 변환합니다.
        
        Args:
            search_result: 포맷팅할 SearchResult 객체
        
        Returns:
            str: 마크다운 형식으로 포맷팅된 검색 결과 문자열
            
        Example:
            >>> agent = SearchAgent()
            >>> result = agent.search("AI trends")
            >>> formatted = agent.format_for_llm(result)
            >>> print(formatted[:200])
        """
        return format_search_result_for_llm(search_result)
    
    def get_sources(self) -> List[str]:
        """
        마지막 검색 결과의 출처 URL 리스트를 반환합니다.
        
        Returns:
            List[str]: 출처 URL 리스트 (검색 히스토리가 비어있으면 빈 리스트)
            
        Example:
            >>> agent = SearchAgent()
            >>> agent.search("Python tutorial")
            >>> sources = agent.get_sources()
            >>> print(f"출처 개수: {len(sources)}")
        """
        if not self.search_history:
            return []
        
        return self.search_history[-1].sources
    
    def get_last_result(self) -> Optional[SearchResult]:
        """
        마지막 검색 결과를 반환합니다.
        
        Returns:
            Optional[SearchResult]: 마지막 검색 결과 (없으면 None)
            
        Example:
            >>> agent = SearchAgent()
            >>> agent.search("AI trends")
            >>> last_result = agent.get_last_result()
            >>> if last_result:
            ...     print(f"결과 수: {last_result.result_count}")
        """
        if not self.search_history:
            return None
        
        return self.search_history[-1]
    
    def get_search_count(self) -> int:
        """
        검색 히스토리의 검색 횟수를 반환합니다.
        
        Returns:
            int: 검색 히스토리 길이
            
        Example:
            >>> agent = SearchAgent()
            >>> agent.search("query1")
            >>> agent.search("query2")
            >>> print(f"검색 횟수: {agent.get_search_count()}")
            2
        """
        return len(self.search_history)
    
    def clear_history(self) -> None:
        """
        검색 히스토리를 초기화합니다.
        
        Example:
            >>> agent = SearchAgent()
            >>> agent.search("query1")
            >>> agent.clear_history()
            >>> print(agent.get_search_count())
            0
        """
        count = len(self.search_history)
        self.search_history.clear()
        logger.info(f"검색 히스토리 초기화 - 삭제된 검색 기록: {count}개")


# 테스트 코드 (주석)
# from src.search_agent import SearchAgent
# from src.memory_manager import MemoryManager
# 
# # 메모리 없이 초기화
# agent1 = SearchAgent()
# 
# # 메모리와 함께 초기화
# mm = MemoryManager("search_memory", "data/chroma_db")
# agent2 = SearchAgent(memory_manager=mm)
# 
# print(f"Agent1 메모리: {agent1.memory_manager}")  # None
# print(f"Agent2 메모리: {agent2.memory_manager}")  # <MemoryManager object>
# 
# # 완전한 검색 테스트
# result = agent2.search_with_memory(
#     query="테슬라 최신 뉴스",
#     use_memory=True,
#     save_to_memory=True,
#     memory_threshold=3
# )
# 
# print(f"쿼리: {result['query']}")
# print(f"메모리: {result['source_summary']['from_memory']}개")
# print(f"웹: {result['source_summary']['from_web']}개")
# print(f"총: {result['source_summary']['total']}개")
# 
# # 병합된 결과 확인
# for i, r in enumerate(result['merged_results'][:3], 1):
#     source = r.get('source', 'unknown')
#     text_preview = r.get('text', r.get('content', ''))[:50]
#     print(f"{i}. {text_preview}... (출처: {source})")
# 
# # 웹 검색 후 저장 테스트
# from src.tools.web_search import tavily_search
# 
# web_result = tavily_search("테슬라", max_results=5)
# saved = agent2._save_to_memory(web_result.results, "테슬라")
# 
# print(f"저장됨: {saved}개")
# 
# # 메모리에서 다시 검색
# memory_results = mm.search_memory("테슬라", top_k=5)
# print(f"메모리에서 검색: {len(memory_results)}개")
# 
# # 병합 테스트
# memory_res = mm.search_memory("테슬라", top_k=3)
# web_res = tavily_search("테슬라", max_results=3).results
# 
# merged = agent2._merge_results(memory_res, web_res)
# 
# print(f"총 {len(merged)}개 결과:")
# for i, r in enumerate(merged, 1):
#     print(f"{i}. 출처: {r['source']} | {r['content'][:40]}...")
#     print(f"   Provenance: {r['provenance']['retrieved_from']}")