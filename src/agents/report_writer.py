"""
ReportWriter 모듈

BaseAgent를 상속하여 분석 결과를 바탕으로 구조화된 Markdown 리포트를 작성합니다.
피드백이 있으면 해당 부분을 반영해 수정본을 작성합니다.
"""
from typing import Any, Dict, List, Optional

from openai import OpenAI

from config.prompts import REPORT_WRITER_PROMPT
from src.agents.base_agent import BaseAgent


class ReportWriter(BaseAgent):
    """
    리포트 작성 전문가 에이전트.

    AnalysisAgent 출력(클러스터·인사이트·트렌드)을 바탕으로
    제목, Executive Summary, 서론, 본론, 트렌드 및 전망, 결론, 참고자료 구조의
    Markdown 리포트를 작성합니다. 피드백 시 개선하여 재작성합니다.
    """

    def __init__(self, client: OpenAI) -> None:
        """
        Args:
            client: OpenAI API 클라이언트 인스턴스
        """
        super().__init__(
            client=client,
            name="Writer",
            role="리포트 작성 전문가",
            system_prompt=REPORT_WRITER_PROMPT,
        )

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        분석 결과로 초안을 작성하거나, 피드백이 있으면 수정본을 작성합니다.

        Args:
            input_data: "topic", "analysis"(AnalysisAgent 출력), "feedback"(optional),
                        수정 시 "report"(이전 리포트 본문) 포함

        Returns:
            topic, report(Markdown 전체 텍스트), word_count, has_revision
        """
        topic = input_data.get("topic", "").strip() or "일반"
        analysis = input_data.get("analysis") or {}
        if not isinstance(analysis, dict):
            analysis = {}
        sources = input_data.get("sources") or []
        if not isinstance(sources, list):
            sources = []
        feedback = input_data.get("feedback")
        previous_report = input_data.get("report", "").strip() if feedback else ""

        if feedback and previous_report:
            report = self._revise_report(topic, analysis, previous_report, str(feedback))
            has_revision = True
        else:
            report = self._write_initial_report(topic, analysis, sources)
            has_revision = False

        word_count = len(report.split()) if report else 0
        return {
            "topic": topic,
            "report": report,
            "word_count": word_count,
            "has_revision": has_revision,
        }

    def _write_initial_report(
        self,
        topic: str,
        analysis: Dict[str, Any],
        sources: Optional[List[str]] = None,
    ) -> str:
        """
        분석 결과를 바탕으로 초안 Markdown 리포트를 작성합니다.

        Args:
            topic: 리포트 주제
            analysis: AnalysisAgent 출력 (clusters, insights, trends)
            sources: 검색에서 수집한 출처 URL 목록 (참고자료 섹션용)

        Returns:
            Markdown 형식의 리포트 문자열
        """
        clusters = analysis.get("clusters") or []
        insights = analysis.get("insights") or []
        trends = analysis.get("trends") or []

        analysis_lines = [
            "### 클러스터(주제별 요약)",
            _format_clusters(clusters),
            "### 핵심 인사이트",
            _format_list(insights),
            "### 트렌드 및 패턴",
            _format_list(trends),
        ]
        analysis_result = "\n".join(analysis_lines).strip()

        if sources:
            sources_block = "\n".join(f"- {url}" for url in sources[:50])
        else:
            sources_block = "(수집된 출처가 없습니다. 참고자료는 생략하거나 '수집된 출처 없음'으로 표기하세요.)"

        user_message = REPORT_WRITER_PROMPT.format(
            analysis_result=analysis_result,
            sources=sources_block,
            feedback="없음",
        )
        return self._call_llm(user_message, temperature=0.7)

    def _revise_report(
        self,
        topic: str,
        analysis: Dict[str, Any],
        previous_report: str,
        feedback: str,
    ) -> str:
        """
        기존 리포트와 피드백을 바탕으로 개선된 버전을 작성합니다.

        Args:
            topic: 리포트 주제
            analysis: 분석 결과 (참고용)
            previous_report: 이전 리포트 전체 텍스트
            feedback: 개선 피드백

        Returns:
            수정된 Markdown 리포트 문자열
        """
        user_message = (
            "다음 기존 리포트를 아래 개선 피드백에 따라 수정한 전체 리포트를 "
            "Markdown 형식으로 작성해주세요. 피드백을 반영한 수정본만 출력하세요.\n\n"
            "## 기존 리포트\n"
            f"{previous_report}\n\n"
            "## 개선 피드백\n"
            f"{feedback}"
        )
        return self._call_llm(user_message, temperature=0.7)


def _format_clusters(clusters: list) -> str:
    """clusters 리스트를 읽기 쉬운 문자열로 변환합니다."""
    if not clusters:
        return "(없음)"
    lines = []
    for i, c in enumerate(clusters, 1):
        if isinstance(c, dict):
            theme = c.get("theme", "")
            summary = c.get("summary", "")
            points = c.get("key_points") or []
            part = f"- **{theme}**: {summary}"
            if points:
                part += "\n  - " + "\n  - ".join(str(p) for p in points)
            lines.append(part)
        else:
            lines.append(f"- {c}")
    return "\n".join(lines) if lines else "(없음)"


def _format_list(items: list) -> str:
    """문자열 리스트를 번호/불릿 문자열로 변환합니다."""
    if not items:
        return "(없음)"
    return "\n".join(f"- {item}" for item in items)
