"""
AnalysisAgent 모듈

BaseAgent를 상속하여 수집된 데이터를 LLM으로 분석하고, 클러스터·인사이트·트렌드를 도출합니다.
외부 도구(검색, DB)는 사용하지 않고 순수 LLM 분석만 수행합니다.
"""
from typing import Any, Dict, List

from openai import OpenAI

from config.prompts import ANALYSIS_AGENT_PROMPT
from src.agents.base_agent import BaseAgent

# 각 항목 최대 길이 (토큰/컨텍스트 절약)
_MAX_ITEM_CHARS = 500


class AnalysisAgent(BaseAgent):
    """
    데이터 분석 전문가 에이전트.

    수집된 검색·메모리 데이터를 하나의 텍스트로 통합한 뒤,
    LLM으로 클러스터링·인사이트·트렌드를 분석하여 반환합니다.
    """

    def __init__(self, client: OpenAI) -> None:
        """
        Args:
            client: OpenAI API 클라이언트 인스턴스
        """
        super().__init__(
            client=client,
            name="Analyzer",
            role="데이터 분석 전문가",
            system_prompt=ANALYSIS_AGENT_PROMPT,
        )

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        수집 데이터를 통합한 뒤 LLM으로 분석하여 클러스터·인사이트·트렌드를 반환합니다.

        Args:
            input_data: "topic", "search_data", "memory_data"(optional) 키를 받음

        Returns:
            topic, clusters, insights, trends, raw_data(원본 search_data) 를 포함한 딕셔너리
        """
        topic = input_data.get("topic", "").strip() or "일반"
        search_data = input_data.get("search_data") or []
        memory_data = input_data.get("memory_data") or []
        if not isinstance(search_data, list):
            search_data = []
        if not isinstance(memory_data, list):
            memory_data = []

        combined = self._prepare_data(search_data, memory_data)
        analysis = self._analyze(topic, combined)
        return {
            "topic": topic,
            "clusters": analysis.get("clusters") or [],
            "insights": analysis.get("insights") or [],
            "trends": analysis.get("trends") or [],
            "raw_data": search_data,
        }

    def _prepare_data(
        self,
        search_data: List[Dict[str, Any]],
        memory_data: List[Dict[str, Any]],
    ) -> str:
        """
        검색 데이터와 메모리 데이터를 하나의 분석용 텍스트로 통합합니다.
        각 항목은 500자로 잘라 토큰 과다 사용을 방지합니다.

        Args:
            search_data: [{"query": str, "result": str}, ...] 형태의 웹 검색 결과
            memory_data: 메모리 검색 결과 (예: [{"text": str, ...}, ...])

        Returns:
            "=== 웹 검색 결과 ===" / "=== 기존 지식 ===" 섹션으로 구분된 문자열
        """
        parts: List[str] = []

        if search_data:
            parts.append("=== 웹 검색 결과 ===")
            for i, item in enumerate(search_data, 1):
                query = item.get("query", "")
                result = (item.get("result") or "").strip()
                if len(result) > _MAX_ITEM_CHARS:
                    result = result[:_MAX_ITEM_CHARS] + "..."
                parts.append(f"{i}. [쿼리: {query}]\n{result}")
            parts.append("")

        if memory_data:
            parts.append("=== 기존 지식 ===")
            for i, item in enumerate(memory_data, 1):
                text = (item.get("text") or str(item)).strip()
                if len(text) > _MAX_ITEM_CHARS:
                    text = text[:_MAX_ITEM_CHARS] + "..."
                parts.append(f"{i}. {text}")
            parts.append("")

        if not parts:
            return "(수집된 데이터가 없습니다.)"
        return "\n".join(parts).strip()

    def _analyze(self, topic: str, data: str) -> Dict[str, Any]:
        """
        LLM을 호출하여 통합 데이터를 분석하고 clusters, insights, trends를 JSON으로 반환합니다.

        Args:
            topic: 분석 주제
            data: _prepare_data로 만든 통합 텍스트

        Returns:
            clusters, insights, trends를 포함한 딕셔너리. 실패 시 기본값 반환.
        """
        user_message = f"## 주제\n{topic}\n\n" + ANALYSIS_AGENT_PROMPT.format(
            collected_data=data
        )
        result = self._call_llm_json(user_message, temperature=0.3)
        if result and (
            isinstance(result.get("clusters"), list)
            or isinstance(result.get("insights"), list)
            or isinstance(result.get("trends"), list)
        ):
            return {
                "clusters": result.get("clusters") if isinstance(result.get("clusters"), list) else [],
                "insights": result.get("insights") if isinstance(result.get("insights"), list) else [],
                "trends": result.get("trends") if isinstance(result.get("trends"), list) else [],
            }
        self.logger.warning("분석 결과가 비어 있거나 형식 오류, 기본값 반환")
        return {
            "clusters": [],
            "insights": ["분석 실패"],
            "trends": [],
        }
