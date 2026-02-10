"""
무한 루프 방지 모듈
ReAct 실행 중 무한 루프를 감지하고 방지합니다.
"""
from typing import List, Dict, Any, Set
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class LoopPrevention:
    """
    무한 루프 감지 및 방지

    감지 기준:
    1. 동일한 Thought 반복
    2. 동일한 Action 3회 연속
    3. 최대 반복 횟수 초과
    """

    max_iterations: int = 10
    max_same_action: int = 3

    action_history: List[str] = field(default_factory=list)
    thought_hashes: Set[int] = field(default_factory=set)
    iteration_count: int = 0

    def reset(self) -> None:
        """상태 초기화"""
        self.action_history = []
        self.thought_hashes = set()
        self.iteration_count = 0

    def check_loop(self, thought: str, action: Dict[str, Any]) -> Dict[str, Any]:
        """루프 여부 확인"""
        self.iteration_count += 1

        result = {"is_loop": False, "reason": None, "should_stop": False}

        # 1. 최대 반복 횟수 확인
        if self.iteration_count > self.max_iterations:
            result["is_loop"] = True
            result["reason"] = f"최대 반복 횟수({self.max_iterations}) 초과"
            result["should_stop"] = True
            return result

        # 2. 동일 Thought 반복 확인
        thought_hash = hash(thought[:100])
        if thought_hash in self.thought_hashes:
            result["is_loop"] = True
            result["reason"] = "동일한 Thought 반복"
        else:
            self.thought_hashes.add(thought_hash)

        # 3. 동일 Action 연속 확인
        action_str = f"{action.get('action', '')}:{str(action.get('action_input', ''))[:50]}"
        self.action_history.append(action_str)

        if len(self.action_history) >= self.max_same_action:
            recent = self.action_history[-self.max_same_action:]
            if len(set(recent)) == 1:
                result["is_loop"] = True
                result["reason"] = f"동일 Action {self.max_same_action}회 연속"
                result["should_stop"] = True

        return result

    def get_stats(self) -> Dict[str, Any]:
        """현재 통계 반환"""
        return {
            "iteration_count": self.iteration_count,
            "unique_thoughts": len(self.thought_hashes),
            "total_actions": len(self.action_history),
            "max_iterations": self.max_iterations,
        }


# ============================================================================
# 테스트 코드 (주석)
# ============================================================================
# from src.loop_prevention import LoopPrevention
# lp = LoopPrevention(max_iterations=5, max_same_action=3)
# result = lp.check_loop("첫 번째 생각", {"action": "search_web", "action_input": "AI"})
# print(f"Loop: {result['is_loop']}")  # False
