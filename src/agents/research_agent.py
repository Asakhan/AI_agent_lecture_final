"""
ResearchAgent 모듈

BaseAgent를 상속하여 SearchAgent·MemoryManager를 활용해 주제별 정보를 수집하는 에이전트입니다.
"""
from typing import Any, Dict, List

from openai import OpenAI

from config.prompts import RESEARCH_AGENT_PROMPT
from src.agents.base_agent import BaseAgent
from src.memory_manager import MemoryManager
from src.search_agent import SearchAgent


class ResearchAgent(BaseAgent):
    """
    정보 수집 전문가 에이전트.

    주제에 대해 검색 쿼리를 생성하고, 메모리 검색과 웹 검색을 수행하여
    수집 데이터를 반환합니다.
    """

    def __init__(
        self,
        client: OpenAI,
        search_agent: SearchAgent,
        memory_manager: MemoryManager,
    ) -> None:
        """
        Args:
            client: OpenAI API 클라이언트 인스턴스
            search_agent: 웹 검색 에이전트
            memory_manager: 메모리 관리자 (기존 지식 검색용)
        """
        super().__init__(
            client=client,
            name="Researcher",
            role="정보 수집 전문가",
            system_prompt=RESEARCH_AGENT_PROMPT,
        )
        self.search_agent = search_agent
        self.memory_manager = memory_manager

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        주제를 받아 메모리 검색 및 웹 검색을 수행하고 수집 결과를 반환합니다.

        Args:
            input_data: "topic" 키로 주제 문자열을 받음

        Returns:
            topic, memory_data, search_data, source_count, queries_used 를 포함한 딕셔너리
        """
        topic = input_data.get("topic", "").strip() or "일반"
        queries = self._generate_search_queries(topic)
        memory_data = self._search_memory(topic)
        search_data, source_urls = self._execute_searches(queries)
        source_count = len(memory_data) + len(search_data)
        return {
            "topic": topic,
            "memory_data": memory_data,
            "search_data": search_data,
            "source_count": source_count,
            "queries_used": queries,
            "source_urls": source_urls,
        }

    def _generate_search_queries(self, topic: str) -> List[str]:
        """
        LLM을 사용하여 주제에 대한 3-5개의 검색 쿼리를 생성합니다.

        Args:
            topic: 리서치 주제

        Returns:
            검색 쿼리 문자열 리스트. 실패 시 [topic] 반환.
        """
        user_message = f"주제: {topic}\n\n위 주제에 대해 3-5개의 검색 쿼리를 JSON의 queries 배열로 생성하세요."
        data = self._call_llm_json(user_message, temperature=0.3)
        queries = data.get("queries") if isinstance(data.get("queries"), list) else None
        if queries and len(queries) > 0:
            return [str(q).strip() for q in queries if q]
        self.logger.warning("검색 쿼리 생성 실패 또는 빈 결과, topic을 쿼리로 사용")
        return [topic]

    def _search_memory(self, topic: str) -> List[Dict[str, Any]]:
        """
        메모리에서 주제와 관련된 기존 지식을 검색합니다.

        Args:
            topic: 검색할 주제

        Returns:
            검색 결과 딕셔너리 리스트. 에러 시 빈 리스트.
        """
        try:
            return self.memory_manager.search_memory(topic, top_k=3)
        except Exception as e:
            self.logger.error("메모리 검색 실패: %s", e, exc_info=True)
            return []

    def _execute_searches(self, queries: List[str]) -> tuple[List[Dict[str, Any]], List[str]]:
        """
        각 쿼리로 웹 검색을 수행하고 포맷된 결과와 출처 URL 목록을 수집합니다.

        Args:
            queries: 검색 쿼리 문자열 리스트

        Returns:
            (search_data 리스트, 출처 URL 리스트). search_data는 [{"query": str, "result": str}, ...].
        """
        results: List[Dict[str, Any]] = []
        source_urls: List[str] = []
        seen: set[str] = set()
        for query in queries:
            if not query or not str(query).strip():
                continue
            try:
                search_result = self.search_agent.search(str(query).strip())
                formatted = self.search_agent.format_for_llm(search_result)
                results.append({"query": query, "result": formatted})
                for url in getattr(search_result, "sources", []) or []:
                    if url and url not in seen:
                        seen.add(url)
                        source_urls.append(url)
            except Exception as e:
                self.logger.warning("검색 실패 (쿼리: %s): %s", query, e)
        return results, source_urls
