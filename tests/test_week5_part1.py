"""
Week 5 Part 1 단위 테스트

에이전트 클래스 구조와 인터페이스를 검증합니다.
실제 API 호출 없이 Mock을 사용합니다.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock

import pytest

from openai import OpenAI


# ---------------------------------------------------------------------------
# 1. BaseAgent 추상 클래스
# ---------------------------------------------------------------------------

def test_base_agent_is_abstract():
    """BaseAgent를 직접 인스턴스화하면 TypeError가 발생하는지 확인"""
    from src.agents.base_agent import BaseAgent

    with pytest.raises(TypeError):
        BaseAgent(
            client=MagicMock(),
            name="Test",
            role="Test",
            system_prompt="Test",
        )


def test_base_agent_subclass_interface():
    """BaseAgent를 상속한 DummyAgent가 execute()를 구현하면 정상 생성되고 속성이 설정되는지 확인"""
    from src.agents.base_agent import BaseAgent

    class DummyAgent(BaseAgent):
        def execute(self, input_data):
            return {"done": True}

    client = MagicMock(spec=OpenAI)
    agent = DummyAgent(
        client=client,
        name="Dummy",
        role="테스트 에이전트",
        system_prompt="테스트 프롬프트",
    )

    assert agent.name == "Dummy"
    assert agent.role == "테스트 에이전트"
    assert agent.system_prompt == "테스트 프롬프트"
    assert agent.client is client
    assert agent.execute({}) == {"done": True}


# ---------------------------------------------------------------------------
# 2. ResearchAgent
# ---------------------------------------------------------------------------

def test_research_agent_creation():
    """ResearchAgent가 Mock으로 올바르게 생성되고 name, search_agent, memory_manager가 설정되는지 확인"""
    from src.agents.research_agent import ResearchAgent

    mock_client = MagicMock(spec=OpenAI)
    mock_search_agent = MagicMock()
    mock_memory_manager = MagicMock()

    agent = ResearchAgent(
        client=mock_client,
        search_agent=mock_search_agent,
        memory_manager=mock_memory_manager,
    )

    assert agent.name == "Researcher"
    assert agent.role == "정보 수집 전문가"
    assert agent.search_agent is mock_search_agent
    assert agent.memory_manager is mock_memory_manager


# ---------------------------------------------------------------------------
# 3. AnalysisAgent
# ---------------------------------------------------------------------------

def test_analysis_agent_creation():
    """AnalysisAgent가 올바르게 생성되고 name이 'Analyzer'인지 확인"""
    from src.agents.analysis_agent import AnalysisAgent

    mock_client = MagicMock(spec=OpenAI)
    agent = AnalysisAgent(client=mock_client)

    assert agent.name == "Analyzer"
    assert agent.role == "데이터 분석 전문가"


# ---------------------------------------------------------------------------
# 4. ReportWriter
# ---------------------------------------------------------------------------

def test_report_writer_creation():
    """ReportWriter가 올바르게 생성되고 name이 'Writer'인지 확인"""
    from src.agents.report_writer import ReportWriter

    mock_client = MagicMock(spec=OpenAI)
    agent = ReportWriter(client=mock_client)

    assert agent.name == "Writer"
    assert agent.role == "리포트 작성 전문가"


# ---------------------------------------------------------------------------
# 5. QualityCritic
# ---------------------------------------------------------------------------

def test_quality_critic_creation():
    """QualityCritic이 올바르게 생성되고 name이 'Critic', PASS_THRESHOLD가 7.0인지 확인"""
    from src.agents.quality_critic import QualityCritic

    mock_client = MagicMock(spec=OpenAI)
    agent = QualityCritic(client=mock_client)

    assert agent.name == "Critic"
    assert agent.role == "품질 검증 전문가"
    assert QualityCritic.PASS_THRESHOLD == 7.0


# ---------------------------------------------------------------------------
# 6. config.prompts
# ---------------------------------------------------------------------------

def test_prompts_exist():
    """config.prompts에 Week 5용 5개 프롬프트 상수가 존재하고 빈 문자열이 아닌지 확인"""
    import config.prompts as prompts

    required = [
        "RESEARCH_AGENT_PROMPT",
        "ANALYSIS_AGENT_PROMPT",
        "REPORT_WRITER_PROMPT",
        "CRITIC_AGENT_PROMPT",
        "COORDINATOR_PROMPT",
    ]
    for name in required:
        assert hasattr(prompts, name), f"{name}이 config.prompts에 없습니다."
        value = getattr(prompts, name)
        assert isinstance(value, str), f"{name}은 문자열이어야 합니다."
        assert len(value) > 0, f"{name}이 비어 있으면 안 됩니다."


# ---------------------------------------------------------------------------
# 7. src.agents 패키지 import
# ---------------------------------------------------------------------------

def test_agents_package_imports():
    """src.agents에서 BaseAgent, ResearchAgent, AnalysisAgent, ReportWriter, QualityCritic을 import할 수 있는지 확인"""
    from src.agents import (
        BaseAgent,
        ResearchAgent,
        AnalysisAgent,
        ReportWriter,
        QualityCritic,
    )

    assert BaseAgent is not None
    assert ResearchAgent is not None
    assert AnalysisAgent is not None
    assert ReportWriter is not None
    assert QualityCritic is not None
