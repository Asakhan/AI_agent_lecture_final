"""
ResearchCoordinator 모듈

ResearchAgent, AnalysisAgent, ReportWriter, QualityCritic을 순차 조율하여
주제별 최종 리포트를 생성합니다. 기존 AutonomousOrchestrator와 별개의 클래스입니다.
"""
import logging
from typing import Any, Dict, List, Optional

from openai import OpenAI

from src.agents import (
    ResearchAgent,
    AnalysisAgent,
    ReportWriter,
    QualityCritic,
)
from src.memory_manager import MemoryManager
from src.search_agent import SearchAgent


logger = logging.getLogger(__name__)


class ResearchCoordinator:
    """
    리서치 파이프라인 코디네이터.

    4개 전문 에이전트(Researcher → Analyzer → Writer ↔ Critic)를 순차 실행하여
    주제에 대한 최종 리포트를 생성하고, 품질 미달 시 수정 루프를 수행합니다.
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
            memory_manager: 메모리 관리자
        """
        self.client = client
        self.researcher = ResearchAgent(client, search_agent, memory_manager)
        self.analyzer = AnalysisAgent(client)
        self.writer = ReportWriter(client)
        self.critic = QualityCritic(client)
        self.max_revision_rounds = 2
        self.logger = logging.getLogger(__name__)
        self.logger.info("ResearchCoordinator initialized")

    def run(self, topic: str, verbose: bool = True) -> Dict[str, Any]:
        """
        주제에 대해 연구 → 분석 → 작성·검증 루프를 수행하고 최종 결과를 반환합니다.

        Args:
            topic: 리서치 주제
            verbose: True면 단계별 진행 메시지 출력

        Returns:
            topic, report, score, scores, revision_count, research_summary 포함 딕셔너리
        """
        topic = (topic or "").strip() or "일반"
        research_data: Dict[str, Any] = {}
        analysis: Dict[str, Any] = {}
        draft: Dict[str, Any] = {"report": ""}
        review: Dict[str, Any] = {
            "overall_score": 0.0,
            "scores": {},
            "pass": False,
            "feedback": "",
        }
        revision_count = 0

        # Phase 1: 연구
        if verbose:
            print("🔍 [Phase 1/4] Researcher: 정보 수집 중...")
        try:
            research_data = self.researcher.execute({"topic": topic})
            if verbose:
                q = research_data.get("queries_used") or []
                sc = research_data.get("source_count", 0)
                print(f"   쿼리 {len(q)}개, 검색·메모리 결과 {sc}건")
        except Exception as e:
            self.logger.error("Phase 1 (Research) 실패: %s", e, exc_info=True)
            if verbose:
                print(f"   ⚠ 연구 단계 오류: {e}")

        # Phase 2: 분석
        if verbose:
            print("📊 [Phase 2/4] Analyzer: 데이터 분석 중...")
        try:
            analysis = self.analyzer.execute(research_data)
            if verbose:
                clusters = analysis.get("clusters") or []
                insights = analysis.get("insights") or []
                print(f"   클러스터 {len(clusters)}개, 인사이트 {len(insights)}개")
        except Exception as e:
            self.logger.error("Phase 2 (Analysis) 실패: %s", e, exc_info=True)
            if verbose:
                print(f"   ⚠ 분석 단계 오류: {e}")

        # Phase 3: 작성 + 검증 루프
        feedback: Optional[str] = None
        previous_report = ""

        for round_num in range(self.max_revision_rounds + 1):
            if verbose:
                print("✍️ [Phase 3/4] Writer: 리포트 작성 중...")
            try:
                writer_input: Dict[str, Any] = {
                    "topic": topic,
                    "analysis": analysis,
                    "sources": research_data.get("source_urls") or [],
                }
                if feedback:
                    writer_input["feedback"] = feedback
                    writer_input["report"] = previous_report
                draft = self.writer.execute(writer_input)
            except Exception as e:
                self.logger.error("Phase 3 (Write) 실패: %s", e, exc_info=True)
                if verbose:
                    print(f"   ⚠ 작성 단계 오류: {e}")
                break

            if verbose:
                print("🔎 [Phase 3/4] Critic: 품질 검증 중...")
            try:
                review = self.critic.execute({
                    "topic": topic,
                    "report": draft.get("report") or "",
                })
            except Exception as e:
                self.logger.error("Phase 3 (Critic) 실패: %s", e, exc_info=True)
                if verbose:
                    print(f"   ⚠ 검증 단계 오류: {e}")
                break

            if verbose:
                scores = review.get("scores") or {}
                overall = review.get("overall_score", 0)
                for k, v in scores.items():
                    print(f"   - {k}: {v}")
                print(f"   종합: {overall} (합격 기준: {QualityCritic.PASS_THRESHOLD})")

            if review.get("pass"):
                if verbose:
                    print("   ✓ 품질 기준 충족, 완료")
                break

            revision_count += 1
            feedback = review.get("feedback") or ""
            previous_report = draft.get("report") or ""

            if round_num < self.max_revision_rounds and verbose:
                print(f"   수정 반영 후 재검증 (수정 {revision_count}회)")

        # 반환 구조
        research_summary = {
            "queries_used": research_data.get("queries_used") or [],
            "source_count": research_data.get("source_count", 0),
            "insights_count": len(analysis.get("insights") or []),
        }

        return {
            "topic": topic,
            "report": draft.get("report") or "",
            "score": review.get("overall_score", 0.0),
            "scores": review.get("scores") or {},
            "revision_count": revision_count,
            "research_summary": research_summary,
        }

    def get_agents_info(self) -> List[Dict[str, str]]:
        """
        각 에이전트의 name, role 정보를 리스트로 반환합니다. UI 표시용.

        Returns:
            [{"name": str, "role": str}, ...]
        """
        return [
            {"name": self.researcher.name, "role": self.researcher.role},
            {"name": self.analyzer.name, "role": self.analyzer.role},
            {"name": self.writer.name, "role": self.writer.role},
            {"name": self.critic.name, "role": self.critic.role},
        ]
