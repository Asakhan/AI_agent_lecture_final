"""
ReportFormatter 모듈

리포트를 Markdown/HTML로 변환하고 파일로 저장하는 유틸리티 클래스입니다.
모든 메서드는 정적 메서드입니다.
"""
import os
import re
from datetime import datetime
from typing import Any, Dict, List

# markdown 패키지 선택적 사용 (없으면 간단 HTML 대체)
try:
    import markdown as md_lib
    _HAS_MARKDOWN = True
except ImportError:
    _HAS_MARKDOWN = False


class ReportFormatter:
    """
    리포트를 Markdown·HTML로 변환하고 저장하는 유틸리티.
    모든 메서드는 @staticmethod입니다.
    """

    @staticmethod
    def to_markdown(report: str, metadata: Dict[str, Any]) -> str:
        """
        리포트 앞에 YAML front matter를 붙인 Markdown 문자열을 반환합니다.

        Args:
            report: 본문 리포트 텍스트
            metadata: title, agent_score, source_count, revision_count 등 (date는 자동)

        Returns:
            YAML front matter + 본문
        """
        title = metadata.get("title", "리포트")
        now = datetime.now().strftime("%Y-%m-%d")
        score = metadata.get("agent_score", 0)
        if isinstance(score, (int, float)):
            score_str = f"{score}/10"
        else:
            score_str = str(score)
        sources = metadata.get("source_count", 0)
        revisions = metadata.get("revision_count", 0)

        front = (
            "---\n"
            f"title: {title}\n"
            f"date: {now}\n"
            f"agent_score: {score_str}\n"
            f"sources: {sources}\n"
            f"revisions: {revisions}\n"
            "---\n\n"
        )
        return front + (report or "")

    @staticmethod
    def to_html(report: str, metadata: Dict[str, Any]) -> str:
        """
        Markdown 본문을 HTML 문서로 변환합니다.
        markdown 패키지가 있으면 사용하고, 없으면 간단한 HTML 태그로 대체합니다.

        Args:
            report: Markdown 본문 (front matter 제외해도 됨)
            metadata: title, agent_score, source_count 등 (상단 표시용)

        Returns:
            <!DOCTYPE html> ~ </html> 완성된 HTML 문자열
        """
        title = metadata.get("title", "리포트")
        now = datetime.now().strftime("%Y-%m-%d")
        score = metadata.get("agent_score", 0)
        sources = metadata.get("source_count", 0)

        if _HAS_MARKDOWN:
            body_html = md_lib.markdown(
                report or "",
                extensions=["extra", "nl2br"],
                extension_configs={},
            )
        else:
            # 간단 대체: 단락 구분만
            escaped = (report or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            paras = [p.strip() for p in escaped.split("\n\n") if p.strip()]
            body_html = "".join(f"<p>{p}</p>" for p in paras)

        meta_block = (
            f'<p class="meta">생성일: {now} | 품질점수: {score}/10 | 참고자료: {sources}건</p>'
        )

        html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap" rel="stylesheet">
  <style>
    body {{ font-family: 'Noto Sans KR', sans-serif; max-width: 800px; margin: 0 auto; padding: 2rem; line-height: 1.7; color: #333; }}
    .meta {{ color: #666; font-size: 0.9rem; margin-bottom: 1.5rem; padding-bottom: 1rem; border-bottom: 1px solid #eee; }}
    h1, h2, h3 {{ margin-top: 1.5em; color: #111; }}
    pre, code {{ background: #f5f5f5; padding: 0.2em 0.4em; border-radius: 4px; font-size: 0.9em; }}
    pre {{ padding: 1rem; overflow-x: auto; }}
    ul, ol {{ padding-left: 1.5rem; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  {meta_block}
  <div class="content">
  {body_html}
  </div>
</body>
</html>"""
        return html

    @staticmethod
    def _safe_topic(topic: str, max_len: int = 30) -> str:
        """파일명용 안전한 주제 문자열 (공백→_, 특수문자 제거, 길이 제한)."""
        if not topic or not isinstance(topic, str):
            return "report"
        s = topic.strip().replace(" ", "_")
        s = re.sub(r"[^\w_]", "", s, flags=re.UNICODE)
        s = s or "report"
        return s[:max_len]

    @staticmethod
    def save_report(
        report: str,
        metadata: Dict[str, Any],
        output_dir: str = "data/reports",
    ) -> Dict[str, Any]:
        """
        리포트를 Markdown·HTML 파일로 저장합니다.

        Args:
            report: 본문 리포트 텍스트
            metadata: title(topic), agent_score, source_count, revision_count 등
            output_dir: 저장 디렉토리 (없으면 생성)

        Returns:
            files (경로 목록), preview (앞 500자), word_count
        """
        os.makedirs(output_dir, exist_ok=True)
        topic = metadata.get("title", metadata.get("topic", "report"))
        safe = ReportFormatter._safe_topic(str(topic))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{safe}_{timestamp}"

        md_path = os.path.join(output_dir, f"{base_name}.md")
        html_path = os.path.join(output_dir, f"{base_name}.html")

        md_content = ReportFormatter.to_markdown(report, metadata)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        html_content = ReportFormatter.to_html(report, metadata)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        preview = (report or "")[:500]
        if len(report or "") > 500:
            preview += "..."

        return {
            "files": [
                {"format": "markdown", "path": md_path},
                {"format": "html", "path": html_path},
            ],
            "preview": preview,
            "word_count": len((report or "").split()),
        }

    @staticmethod
    def print_report_summary(result: Dict[str, Any], score: float) -> None:
        """
        터미널에 리포트 생성 결과를 요약 출력합니다.

        Args:
            result: save_report() 반환값 (files, preview, word_count)
            score: 품질 점수
        """
        files = result.get("files") or []
        word_count = result.get("word_count", 0)

        print("\n" + "=" * 50)
        print("📄 리포트 생성 완료")
        print("=" * 50)
        print(f"  품질 점수: {score}/10")
        print(f"  분량: {word_count}단어")
        print("  저장된 파일:")
        for f in files:
            print(f"    - [{f.get('format', '')}] {f.get('path', '')}")
        print("=" * 50)
