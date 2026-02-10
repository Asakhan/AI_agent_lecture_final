"""
Week 5 Part 2 단위 테스트

ResearchCoordinator, ReportFormatter 클래스 구조와 동작을 검증합니다.
실제 API 호출 없이 Mock·임시 디렉토리를 사용합니다.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock

import pytest

from openai import OpenAI


# ---------------------------------------------------------------------------
# 1. ResearchCoordinator
# ---------------------------------------------------------------------------

def test_research_coordinator_creation():
    """ResearchCoordinator가 Mock으로 올바르게 생성되고 4개 에이전트·max_revision_rounds가 설정되는지 확인"""
    from src.research_coordinator import ResearchCoordinator

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
    assert coordinator.researcher.name == "Researcher"
    assert coordinator.analyzer.name == "Analyzer"
    assert coordinator.writer.name == "Writer"
    assert coordinator.critic.name == "Critic"
    assert coordinator.max_revision_rounds == 2


def test_research_coordinator_agents_info():
    """get_agents_info()가 4개 에이전트 정보를 반환하고 각각 name, role 키가 있는지 확인"""
    from src.research_coordinator import ResearchCoordinator

    mock_client = MagicMock(spec=OpenAI)
    mock_search_agent = MagicMock()
    mock_memory_manager = MagicMock()

    coordinator = ResearchCoordinator(
        client=mock_client,
        search_agent=mock_search_agent,
        memory_manager=mock_memory_manager,
    )
    info = coordinator.get_agents_info()

    assert len(info) == 4
    for item in info:
        assert "name" in item
        assert "role" in item
        assert isinstance(item["name"], str)
        assert isinstance(item["role"], str)


# ---------------------------------------------------------------------------
# 3. ReportFormatter - to_markdown
# ---------------------------------------------------------------------------

def test_report_formatter_to_markdown():
    """ReportFormatter.to_markdown() 결과에 YAML front matter, metadata title, 원본 report가 포함되는지 확인"""
    from src.report_formatter import ReportFormatter

    report = "# 테스트 리포트\n\n본문 내용입니다."
    metadata = {
        "title": "AI 반도체 시장 분석",
        "agent_score": 7.8,
        "source_count": 12,
        "revision_count": 1,
    }

    result = ReportFormatter.to_markdown(report, metadata)

    assert "---" in result
    assert "AI 반도체 시장 분석" in result
    assert "# 테스트 리포트" in result
    assert "본문 내용입니다." in result


# ---------------------------------------------------------------------------
# 4. ReportFormatter - to_html
# ---------------------------------------------------------------------------

def test_report_formatter_to_html():
    """ReportFormatter.to_html() 결과에 DOCTYPE, html 태그, metadata 정보가 포함되는지 확인"""
    from src.report_formatter import ReportFormatter

    report = "## 제목\n\n단락 내용."
    metadata = {
        "title": "테스트 리포트",
        "agent_score": 8.0,
        "source_count": 5,
    }

    result = ReportFormatter.to_html(report, metadata)

    assert "<!DOCTYPE html>" in result
    assert "<html" in result
    assert "</html>" in result
    assert "테스트 리포트" in result
    assert "8" in result or "품질" in result or "점수" in result


# ---------------------------------------------------------------------------
# 5. ReportFormatter - save_report (임시 디렉토리)
# ---------------------------------------------------------------------------

def test_report_formatter_save_report():
    """임시 디렉토리에 save_report() 후 files 키·2개 파일·실제 파일 존재 여부 확인, 테스트 후 정리"""
    from src.report_formatter import ReportFormatter

    report = "# 리포트\n\n본문입니다."
    metadata = {"title": "테스트", "agent_score": 7.0, "source_count": 3, "revision_count": 0}

    with tempfile.TemporaryDirectory() as tmpdir:
        result = ReportFormatter.save_report(report, metadata, output_dir=tmpdir)

        assert "files" in result
        files = result["files"]
        assert len(files) == 2

        formats = {f["format"] for f in files}
        assert "markdown" in formats
        assert "html" in formats

        for f in files:
            path = f.get("path")
            assert path is not None
            assert os.path.exists(path), f"파일이 생성되지 않음: {path}"

        assert "preview" in result
        assert "word_count" in result
        assert result["word_count"] >= 0


# ---------------------------------------------------------------------------
# 6. ReportFormatter - safe filename (특수문자 제거)
# ---------------------------------------------------------------------------

def test_report_formatter_safe_filename():
    """특수문자가 포함된 주제로 save_report() 호출 시 파일명에 특수문자가 없고 파일이 정상 생성되는지 확인"""
    from src.report_formatter import ReportFormatter

    report = "본문"
    metadata = {
        "title": "AI 반도체 시장 분석!! @2025",
        "agent_score": 6,
        "source_count": 0,
        "revision_count": 0,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        result = ReportFormatter.save_report(report, metadata, output_dir=tmpdir)
        files = result.get("files") or []

        for f in files:
            path = f.get("path", "")
            basename = os.path.basename(path)
            assert "!" not in basename
            assert "@" not in basename
            assert os.path.exists(path)


# ---------------------------------------------------------------------------
# 7. ReportFormatter - print_report_summary
# ---------------------------------------------------------------------------

def test_report_formatter_print_summary(capsys):
    """print_report_summary()가 에러 없이 실행되고, 캡처한 출력에 점수가 포함되는지 확인"""
    from src.report_formatter import ReportFormatter

    result = {
        "files": [
            {"format": "markdown", "path": "data/reports/test.md"},
            {"format": "html", "path": "data/reports/test.html"},
        ],
        "preview": "미리보기...",
        "word_count": 100,
    }
    score = 7.5

    ReportFormatter.print_report_summary(result, score)

    captured = capsys.readouterr()
    assert "7.5" in captured.out or "7.5/10" in captured.out
    assert "품질" in captured.out or "점수" in captured.out
    assert "100" in captured.out
