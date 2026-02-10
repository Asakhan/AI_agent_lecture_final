"""
QualityCritic 모듈

BaseAgent를 상속하여 리포트 전용 5항목(completeness, accuracy, clarity, structure, source_quality)
품질 평가를 수행합니다. 기존 QualityManager와 별개인 리포트 전용 에이전트입니다.
"""
from typing import Any, Dict

from openai import OpenAI

from config.prompts import CRITIC_AGENT_PROMPT
from src.agents.base_agent import BaseAgent

# 5개 항목 키 (CRITIC_AGENT_PROMPT와 동일 순서)
_SCORE_KEYS = (
    "completeness",
    "accuracy",
    "clarity",
    "structure",
    "source_quality",
)

_DEFAULT_SCORE = 5.0


class QualityCritic(BaseAgent):
    """
    품질 검증 전문가 에이전트.

    리포트를 5가지 기준(완성도, 정확성, 명확성, 구조, 출처 품질)으로 1~10점 평가하고,
    개선 피드백 문자열을 반환합니다. QualityManager(3항목)와 별도입니다.
    """

    PASS_THRESHOLD: float = 7.0

    def __init__(self, client: OpenAI) -> None:
        """
        Args:
            client: OpenAI API 클라이언트 인스턴스
        """
        super().__init__(
            client=client,
            name="Critic",
            role="품질 검증 전문가",
            system_prompt=CRITIC_AGENT_PROMPT,
        )

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        리포트를 5항목으로 평가하고 점수·종합 점수·피드백·합격 여부를 반환합니다.

        Args:
            input_data: "topic", "report" 키를 받음

        Returns:
            scores, overall_score, feedback(str), pass(bool) 를 포함한 딕셔너리
        """
        topic = input_data.get("topic", "").strip() or "일반"
        report = input_data.get("report") or ""
        if not isinstance(report, str):
            report = str(report)

        result = self._evaluate_report(topic, report)
        scores = result.get("scores") or {}
        overall = result.get("overall_score", _DEFAULT_SCORE)
        feedback_str = result.get("feedback", "")
        passed = overall >= self.PASS_THRESHOLD

        return {
            "scores": scores,
            "overall_score": overall,
            "feedback": feedback_str,
            "pass": passed,
        }

    def _evaluate_report(self, topic: str, report: str) -> Dict[str, Any]:
        """
        LLM으로 리포트를 5항목 1~10점 평가하고, overall 및 피드백을 반환합니다.

        Args:
            topic: 리포트 주제 (참고용)
            report: 평가할 리포트 전체 텍스트

        Returns:
            scores(dict), overall_score(float), feedback(str) 포함.
            실패 시 기본값(전 항목 5.0, pass=False)에 맞는 값 반환.
        """
        user_message = CRITIC_AGENT_PROMPT.format(report=report)
        data = self._call_llm_json(user_message, temperature=0.3)

        if not data:
            self.logger.warning("평가 결과 없음, 기본값 반환")
            return self._default_evaluation_result()

        scores_raw = data.get("scores")
        if not isinstance(scores_raw, dict):
            return self._default_evaluation_result()

        scores: Dict[str, float] = {}
        for key in _SCORE_KEYS:
            val = scores_raw.get(key)
            try:
                scores[key] = float(val) if val is not None else _DEFAULT_SCORE
            except (TypeError, ValueError):
                scores[key] = _DEFAULT_SCORE
            scores[key] = max(1.0, min(10.0, scores[key]))

        overall = data.get("overall")
        if overall is not None:
            try:
                overall_score = float(overall)
            except (TypeError, ValueError):
                overall_score = sum(scores.values()) / len(_SCORE_KEYS)
        else:
            overall_score = sum(scores.values()) / len(_SCORE_KEYS)
        overall_score = round(overall_score, 1)

        feedback_raw = data.get("feedback")
        if isinstance(feedback_raw, dict):
            parts = [
                f"- **{k}**: {v}" for k in _SCORE_KEYS
                for v in [feedback_raw.get(k, "")]
                if v
            ]
            feedback_str = "\n".join(parts) if parts else "구체적 피드백 없음"
        elif isinstance(feedback_raw, str):
            feedback_str = feedback_raw
        else:
            feedback_str = "구체적 피드백 없음"

        return {
            "scores": scores,
            "overall_score": overall_score,
            "feedback": feedback_str,
        }

    def _default_evaluation_result(self) -> Dict[str, Any]:
        """평가 실패 시 반환하는 기본 구조."""
        scores = {k: _DEFAULT_SCORE for k in _SCORE_KEYS}
        return {
            "scores": scores,
            "overall_score": _DEFAULT_SCORE,
            "feedback": "평가를 수행할 수 없습니다.",
        }
