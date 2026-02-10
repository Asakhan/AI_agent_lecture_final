"""
ReAct 실행 엔진
Thought-Action-Observation 순환을 통해 작업을 자율적으로 수행합니다.
"""
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from openai import OpenAI
import json
import re
import logging

from config.prompts import REACT_SYSTEM_PROMPT, REACT_USER_PROMPT
from src.loop_prevention import LoopPrevention

logger = logging.getLogger(__name__)


@dataclass
class ReActStep:
    """ReAct 단일 스텝 기록"""

    iteration: int
    thought: str
    action: Dict[str, Any]
    observation: str


class ReActEngine:
    """ReAct 패턴 기반 자율 실행 엔진"""

    def __init__(
        self,
        client: OpenAI,
        tools: Dict[str, Callable],
        max_iterations: int = 10,
    ):
        self.client = client
        self.tools = tools
        self.max_iterations = max_iterations
        self.history: List[ReActStep] = []
        self.loop_prevention = LoopPrevention(max_iterations=max_iterations)
        logger.info(f"ReActEngine initialized with tools: {list(tools.keys())}")

    def run(self, task: str) -> str:
        """작업 실행"""
        self.reset()
        logger.info(f"Starting ReAct for task: {task[:50]}...")

        for i in range(self.max_iterations):
            logger.info(f"=== Iteration {i+1}/{self.max_iterations} ===")

            try:
                thought, action = self._generate_thought_and_action(task)
                logger.info(f"Thought: {thought[:100]}...")
                logger.info(f"Action: {action}")

                loop_check = self.loop_prevention.check_loop(thought, action)
                if loop_check["should_stop"]:
                    return self._force_finish(loop_check["reason"])

                if action.get("action") == "finish":
                    return action.get("answer", "완료")

                observation = self._execute_action(action)
                logger.info(f"Observation: {observation[:200]}...")

                self.history.append(ReActStep(i + 1, thought, action, observation))

            except Exception as e:
                logger.error(f"Error in iteration {i+1}: {e}")
                continue

        return self._force_finish("최대 반복 횟수 도달")

    def _generate_thought_and_action(self, task: str) -> tuple:
        """LLM을 통해 Thought와 Action 생성"""
        history_str = self._format_history()

        system_prompt = REACT_SYSTEM_PROMPT.format(task=task, history=history_str)

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": REACT_USER_PROMPT},
            ],
            temperature=0.7,
        )

        return self._parse_response(response.choices[0].message.content)

    def _parse_response(self, response: str) -> tuple:
        """LLM 응답에서 Thought와 Action 파싱"""
        thought_match = re.search(
            r"Thought:\s*(.+?)(?=Action:|$)", response, re.DOTALL
        )
        thought = thought_match.group(1).strip() if thought_match else "생각 파싱 실패"

        action_match = re.search(r"Action:\s*(\{.+\})", response, re.DOTALL)

        if action_match:
            try:
                action_str = re.sub(r"\s+", " ", action_match.group(1))
                action = json.loads(action_str)
            except json.JSONDecodeError:
                action = {"action": "analyze", "action_input": "파싱 실패"}
        else:
            action = {"action": "analyze", "action_input": response[:200] if response else ""}

        return thought, action

    def _execute_action(self, action: Dict[str, Any]) -> str:
        """Action 실행"""
        action_name = action.get("action", "unknown")
        action_input = action.get("action_input", "")

        if action_name == "finish":
            return action.get("answer", "완료")

        if action_name in self.tools:
            try:
                result = self.tools[action_name](action_input)
                return str(result) if result else "결과 없음"
            except Exception as e:
                return f"도구 실행 오류: {e}"
        else:
            return f"알 수 없는 도구: {action_name}"

    def _format_history(self) -> str:
        """히스토리를 문자열로 포맷팅"""
        if not self.history:
            return "아직 수행한 작업이 없습니다."

        formatted = []
        for step in self.history:
            formatted.append(
                f"[Step {step.iteration}]\n"
                f"Thought: {step.thought}\n"
                f"Action: {json.dumps(step.action, ensure_ascii=False)}\n"
                f"Observation: {step.observation[:500]}..."
            )
        return "\n\n".join(formatted)

    def _force_finish(self, reason: str) -> str:
        """강제 종료"""
        logger.warning(f"Force finishing: {reason}")
        observations = [step.observation for step in self.history]
        if not observations:
            return f"작업 미완료: {reason}"
        return "수집된 정보:\n" + "\n---\n".join(observations[-3:])

    def reset(self) -> None:
        """상태 초기화"""
        self.history = []
        self.loop_prevention.reset()

    def get_stats(self) -> Dict[str, Any]:
        """실행 통계 반환"""
        return {
            "total_steps": len(self.history),
            "loop_stats": self.loop_prevention.get_stats(),
            "actions_used": [step.action.get("action") for step in self.history],
        }
