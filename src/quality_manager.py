"""
품질 관리 모듈
ReAct 실행 결과의 품질을 평가하고 재시도 여부를 판단합니다.
"""
from typing import Dict, Any, Optional
from openai import OpenAI
import json
import logging

logger = logging.getLogger(__name__)


class QualityManager:
    """결과 품질 평가 및 재시도 판단"""

    def __init__(
        self,
        client: OpenAI,
        min_quality_score: float = 7.0,
        max_retries: int = 3,
    ):
        self.client = client
        self.min_quality_score = min_quality_score
        self.max_retries = max_retries
        self.evaluation_history: list = []
        logger.info(f"QualityManager initialized (min_score: {min_quality_score})")

    def evaluate(self, task: str, result: str) -> Dict[str, Any]:
        """결과 품질 평가"""
        prompt = f"""당신은 AI 결과물 품질 평가 전문가입니다.

## 평가할 작업
{task}

## 평가할 결과
{result}

## 평가 기준 (각 1-10점)
1. completeness (완전성): 요청한 내용을 모두 포함하는가
2. accuracy (정확성): 정보가 정확한가
3. relevance (관련성): 요청과 관련된 내용인가

## 출력 형식 (JSON)
{{"completeness": 점수, "accuracy": 점수, "relevance": 점수, "overall": 평균점수, "feedback": "개선 방향"}}

JSON만 출력하세요."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "품질 평가 전문가입니다. JSON만 출력합니다."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
            )

            evaluation = json.loads(response.choices[0].message.content)

            if "overall" not in evaluation:
                scores = [
                    evaluation.get("completeness", 5),
                    evaluation.get("accuracy", 5),
                    evaluation.get("relevance", 5),
                ]
                evaluation["overall"] = sum(scores) / len(scores)

            evaluation["pass"] = evaluation["overall"] >= self.min_quality_score
            self.evaluation_history.append({"task": task[:100], "evaluation": evaluation})

            logger.info(
                f"Quality: {evaluation['overall']:.1f}/10 ({'PASS' if evaluation['pass'] else 'FAIL'})"
            )
            return evaluation

        except Exception as e:
            logger.error(f"Quality evaluation failed: {e}")
            return {
                "completeness": 5,
                "accuracy": 5,
                "relevance": 5,
                "overall": 5.0,
                "feedback": "평가 실패",
                "pass": False,
            }

    def should_retry(self, evaluation: Dict[str, Any]) -> bool:
        """재시도 필요 여부 판단"""
        return not evaluation.get("pass", False)

    def get_improvement_prompt(
        self,
        original_task: str,
        previous_result: str,
        evaluation: Dict[str, Any],
    ) -> str:
        """개선을 위한 프롬프트 생성"""
        return f"""이전 결과가 품질 기준을 충족하지 못했습니다.

## 원래 작업
{original_task}

## 평가 피드백
- 점수: {evaluation.get('overall', 0):.1f}/10
- 개선점: {evaluation.get('feedback', '품질 개선 필요')}

## 요청
위 피드백을 반영하여 더 완전하고 정확한 결과를 제공해주세요."""

    def get_stats(self) -> Dict[str, Any]:
        """평가 통계 반환"""
        if not self.evaluation_history:
            return {"total_evaluations": 0}

        scores = [e["evaluation"]["overall"] for e in self.evaluation_history]
        passed = sum(1 for e in self.evaluation_history if e["evaluation"]["pass"])

        return {
            "total_evaluations": len(self.evaluation_history),
            "average_score": sum(scores) / len(scores),
            "pass_rate": passed / len(self.evaluation_history),
        }
