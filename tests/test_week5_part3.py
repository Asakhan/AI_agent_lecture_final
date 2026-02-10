"""
Week 5 Part 3 통합 테스트

전체 파이프라인 연결 및 5주차·4주차 기능 통합을 검증합니다.
실제 API 호출 없이 Mock을 사용합니다.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock, patch

import pytest

from openai import OpenAI


# ---------------------------------------------------------------------------
# 1. ResearchCoordinator 파이프라인 구조
# ---------------------------------------------------------------------------

def test_full_pipeline_structure():
    """ResearchCoordinator를 Mock 의존성으로 생성하고, 4개 에이전트가 올바른 클래스 인스턴스인지 확인"""
    from src.research_coordinator import ResearchCoordinator
    from src.agents import ResearchAgent, AnalysisAgent, ReportWriter, QualityCritic

    mock_client = MagicMock(spec=OpenAI)
    mock_search_agent = MagicMock()
    mock_memory_manager = MagicMock()

    coordinator = ResearchCoordinator(
        client=mock_client,
        search_agent=mock_search_agent,
        memory_manager=mock_memory_manager,
    )

    assert hasattr(coordinator, "researcher")
    assert hasattr(coordinator, "analyzer")
    assert hasattr(coordinator, "writer")
    assert hasattr(coordinator, "critic")

    assert isinstance(coordinator.researcher, ResearchAgent)
    assert isinstance(coordinator.analyzer, AnalysisAgent)
    assert isinstance(coordinator.writer, ReportWriter)
    assert isinstance(coordinator.critic, QualityCritic)


# ---------------------------------------------------------------------------
# 2. ReportFormatter 체이닝
# ---------------------------------------------------------------------------

def test_report_formatter_integration():
    """실제 Markdown으로 to_markdown → to_html → save_report 순차 호출 후 파일 생성 확인"""
    from src.report_formatter import ReportFormatter

    report = "# 통합 테스트 리포트\n\n## 섹션 1\n본문 내용입니다.\n\n## 섹션 2\n추가 내용."
    metadata = {
        "title": "5주차 통합 테스트",
        "agent_score": 7.5,
        "source_count": 10,
        "revision_count": 1,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        md_out = ReportFormatter.to_markdown(report, metadata)
        assert "---" in md_out
        assert "5주차 통합 테스트" in md_out
        assert report in md_out or "# 통합 테스트 리포트" in md_out

        html_out = ReportFormatter.to_html(report, metadata)
        assert "<!DOCTYPE html>" in html_out
        assert "</html>" in html_out

        save_result = ReportFormatter.save_report(report, metadata, output_dir=tmpdir)
        assert "files" in save_result
        assert len(save_result["files"]) == 2
        for f in save_result["files"]:
            path = f.get("path")
            assert path and os.path.exists(path)


# ---------------------------------------------------------------------------
# 3. 에이전트 execute() 인터페이스 (Mock LLM)
# ---------------------------------------------------------------------------

def test_agent_execute_interface():
    """각 에이전트의 execute()가 Mock된 _call_llm/_call_llm_json 하에서 올바른 키를 반환하는지 확인"""
    from src.agents.research_agent import ResearchAgent
    from src.agents.analysis_agent import AnalysisAgent
    from src.agents.report_writer import ReportWriter
    from src.agents.quality_critic import QualityCritic

    mock_client = MagicMock(spec=OpenAI)
    mock_search = MagicMock()
    mock_memory = MagicMock()

    # ResearchAgent: _call_llm_json, search_agent.search, format_for_llm, memory_manager.search_memory
    mock_search.search.return_value = MagicMock()
    mock_search.format_for_llm.return_value = "검색 결과 텍스트"
    mock_memory.search_memory.return_value = []

    with patch.object(ResearchAgent, "_call_llm_json", return_value={"queries": ["쿼리1", "쿼리2"]}):
        r_agent = ResearchAgent(mock_client, mock_search, mock_memory)
        out = r_agent.execute({"topic": "테스트 주제"})
    assert "topic" in out
    assert "search_data" in out
    assert "source_count" in out

    # AnalysisAgent: _call_llm_json
    with patch.object(AnalysisAgent, "_call_llm_json", return_value={
        "clusters": [{"theme": "A", "summary": "요약", "key_points": []}],
        "insights": ["인사이트1"],
        "trends": ["트렌드1"],
    }):
        a_agent = AnalysisAgent(mock_client)
        out = a_agent.execute({"topic": "T", "search_data": [], "memory_data": []})
    assert "clusters" in out
    assert "insights" in out
    assert "trends" in out

    # ReportWriter: _call_llm
    with patch.object(ReportWriter, "_call_llm", return_value="# 리포트\n\n본문"):
        w_agent = ReportWriter(mock_client)
        out = w_agent.execute({"topic": "T", "analysis": {"clusters": [], "insights": [], "trends": []}})
    assert "report" in out
    assert "word_count" in out

    # QualityCritic: _call_llm_json
    with patch.object(QualityCritic, "_call_llm_json", return_value={
        "scores": {"completeness": 8, "accuracy": 7, "clarity": 8, "structure": 7, "source_quality": 8},
        "overall": 7.6,
        "feedback": {"completeness": "좋음", "accuracy": "보통", "clarity": "좋음", "structure": "보통", "source_quality": "좋음"},
    }):
        c_agent = QualityCritic(mock_client)
        out = c_agent.execute({"topic": "T", "report": "# 리포트\n본문"})
    assert "scores" in out
    assert "overall_score" in out
    assert "pass" in out


# ---------------------------------------------------------------------------
# 4. Coordinator.run() Mock 에이전트
# ---------------------------------------------------------------------------

def test_coordinator_run_with_mocked_agents():
    """4개 에이전트 execute()를 Mock으로 대체한 뒤 coordinator.run()이 올바른 결과 Dict를 반환하는지 확인"""
    from src.research_coordinator import ResearchCoordinator

    mock_client = MagicMock(spec=OpenAI)
    mock_search_agent = MagicMock()
    mock_memory_manager = MagicMock()

    coordinator = ResearchCoordinator(
        client=mock_client,
        search_agent=mock_search_agent,
        memory_manager=mock_memory_manager,
    )

    coordinator.researcher.execute = MagicMock(return_value={
        "topic": "테스트",
        "memory_data": [],
        "search_data": [{"query": "q1", "result": "r1"}],
        "source_count": 1,
        "queries_used": ["q1"],
    })
    coordinator.analyzer.execute = MagicMock(return_value={
        "topic": "테스트",
        "clusters": [{"theme": "T", "summary": "S", "key_points": []}],
        "insights": ["인사이트"],
        "trends": ["트렌드"],
        "raw_data": [],
    })
    coordinator.writer.execute = MagicMock(return_value={
        "topic": "테스트",
        "report": "# 최종 리포트\n\n본문 내용.",
        "word_count": 5,
        "has_revision": False,
    })
    coordinator.critic.execute = MagicMock(return_value={
        "scores": {"completeness": 8, "accuracy": 8, "clarity": 8, "structure": 8, "source_quality": 8},
        "overall_score": 8.0,
        "feedback": "",
        "pass": True,
    })

    result = coordinator.run("테스트", verbose=False)

    assert "report" in result
    assert "score" in result
    assert "revision_count" in result
    assert "topic" in result
    assert "research_summary" in result
    assert result["report"] == "# 최종 리포트\n\n본문 내용."
    assert result["score"] == 8.0


# ---------------------------------------------------------------------------
# 5. main.py import
# ---------------------------------------------------------------------------

def test_main_imports():
    """main.py에서 사용하는 ResearchCoordinator, ReportFormatter import가 정상 동작하는지 확인"""
    from src.research_coordinator import ResearchCoordinator
    from src.report_formatter import ReportFormatter

    assert ResearchCoordinator is not None
    assert ReportFormatter is not None


# ---------------------------------------------------------------------------
# 6. 4주차 기존 기능 유지
# ---------------------------------------------------------------------------

def test_existing_features_intact():
    """4주차 클래스들이 정상 import되는지 확인하여 5주차 변경으로 깨지지 않았는지 검증"""
    from src.orchestrator import AutonomousOrchestrator
    from src.task_planner import TaskPlanner
    from src.react_engine import ReActEngine
    from src.quality_manager import QualityManager
    from src.loop_prevention import LoopPrevention

    assert AutonomousOrchestrator is not None
    assert TaskPlanner is not None
    assert ReActEngine is not None
    assert QualityManager is not None
    assert LoopPrevention is not None
