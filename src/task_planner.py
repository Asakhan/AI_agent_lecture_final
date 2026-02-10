"""
태스크 플래너 모듈

TaskStatus Enum과 Subtask dataclass를 정의합니다.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from openai import OpenAI
import json
import logging

from config.prompts import TASK_DECOMPOSE_PROMPT

logger = logging.getLogger(__name__)


# ============================================================================
# TaskStatus Enum
# ============================================================================

class TaskStatus(str, Enum):
    """태스크 상태"""

    PENDING = "pending"           # 대기 중
    IN_PROGRESS = "in_progress"   # 실행 중
    COMPLETED = "completed"       # 완료
    FAILED = "failed"             # 실패
    SKIPPED = "skipped"           # 건너뜀


# ============================================================================
# Subtask Dataclass
# ============================================================================

@dataclass
class Subtask:
    """하위 태스크"""

    id: str                                    # 태스크 고유 ID (예: "task_1")
    description: str                           # 태스크 설명
    priority: int                               # 우선순위 (1이 가장 높음)
    dependencies: List[str] = field(default_factory=list)  # 의존 태스크 ID 목록
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None               # 실행 결과
    attempts: int = 0                          # 시도 횟수
    max_attempts: int = 3                      # 최대 시도 횟수


# ============================================================================
# TaskPlanner 클래스
# ============================================================================

class TaskPlanner:
    """LLM을 활용한 작업 분해 플래너"""

    def __init__(self, client: OpenAI) -> None:
        """
        Args:
            client: OpenAI 클라이언트
        """
        self.client = client
        self.tasks: List[Subtask] = []
        self.original_goal: str = ""
        logger.info("TaskPlanner initialized")

    def decompose(self, goal: str) -> List[Subtask]:
        """목표를 서브태스크로 분해"""
        self.original_goal = goal
        self.tasks = []

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "작업 분해 전문가입니다. JSON만 출력합니다."},
                    {"role": "user", "content": TASK_DECOMPOSE_PROMPT.format(goal=goal)},
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            data = json.loads(content)

            for task_data in data.get("subtasks", []):
                subtask = Subtask(
                    id=task_data["id"],
                    description=task_data["description"],
                    priority=task_data["priority"],
                    dependencies=task_data.get("dependencies", []),
                )
                self.tasks.append(subtask)

            logger.info(f"Decomposed '{goal}' into {len(self.tasks)} subtasks")
            return self.tasks

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            fallback_task = Subtask(id="task_1", description=goal, priority=1)
            self.tasks = [fallback_task]
            return self.tasks

    def get_next_task(self) -> Optional[Subtask]:
        """실행 가능한 다음 태스크 반환"""
        executable = [
            task for task in self.tasks
            if task.status == TaskStatus.PENDING
            and self._dependencies_met(task)
            and task.attempts < task.max_attempts
        ]

        if not executable:
            # 재시도 가능한 실패 태스크 확인
            retryable = [
                task for task in self.tasks
                if task.status == TaskStatus.FAILED
                and task.attempts < task.max_attempts
                and self._dependencies_met(task)
            ]
            if retryable:
                task = sorted(retryable, key=lambda t: t.priority)[0]
                task.status = TaskStatus.IN_PROGRESS
                task.attempts += 1
                return task
            return None

        next_task = sorted(executable, key=lambda t: t.priority)[0]
        next_task.status = TaskStatus.IN_PROGRESS
        next_task.attempts += 1

        logger.info(f"Next task: {next_task.id} (attempt {next_task.attempts})")
        return next_task

    def update_status(
        self,
        task_id: str,
        status: TaskStatus,
        result: Optional[str] = None,
    ) -> bool:
        """태스크 상태 업데이트"""
        task = self._get_task_by_id(task_id)
        if not task:
            logger.error(f"Task not found: {task_id}")
            return False

        old_status = task.status
        task.status = status

        if result is not None:
            task.result = result

        logger.info(f"Task {task_id}: {old_status.value} → {status.value}")
        return True

    def is_complete(self) -> bool:
        """모든 태스크 완료 여부 확인"""
        if not self.tasks:
            return True

        for task in self.tasks:
            if task.status not in [TaskStatus.COMPLETED, TaskStatus.SKIPPED]:
                return False
        return True

    def _get_task_by_id(self, task_id: str) -> Optional[Subtask]:
        """ID로 태스크 찾기"""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    def _dependencies_met(self, task: Subtask) -> bool:
        """태스크의 의존성이 모두 충족되었는지 확인"""
        for dep_id in task.dependencies:
            dep_task = self._get_task_by_id(dep_id)
            if not dep_task:
                logger.warning(f"Dependency {dep_id} not found for {task.id}")
                return False
            if dep_task.status != TaskStatus.COMPLETED:
                return False
        return True

    def visualize(self) -> None:
        """현재 태스크 상태를 시각적으로 출력"""
        status_icons = {
            TaskStatus.PENDING: "⏳",
            TaskStatus.IN_PROGRESS: "🔄",
            TaskStatus.COMPLETED: "✅",
            TaskStatus.FAILED: "❌",
            TaskStatus.SKIPPED: "⏭️",
        }

        print("\n" + "=" * 60)
        print(f"📋 Task Plan: {self.original_goal[:50]}...")
        print("=" * 60)

        for task in self.tasks:
            icon = status_icons.get(task.status, "❓")
            deps_str = f" (depends: {', '.join(task.dependencies)})" if task.dependencies else ""

            print(f"\n{icon} [{task.id}] {task.description}")
            print(f"   Priority: {task.priority} | Status: {task.status.value}{deps_str}")

            if task.result:
                result_preview = task.result[:100] + "..." if len(task.result) > 100 else task.result
                print(f"   Result: {result_preview}")

        completed = sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED)
        total = len(self.tasks)
        progress = (completed / total * 100) if total > 0 else 0

        print("\n" + "-" * 60)
        print(f"📊 Progress: {completed}/{total} ({progress:.1f}%)")
        print("=" * 60 + "\n")

    def get_summary(self) -> Dict[str, Any]:
        """현재 상태 요약 반환"""
        return {
            "goal": self.original_goal,
            "total_tasks": len(self.tasks),
            "completed": sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED),
            "failed": sum(1 for t in self.tasks if t.status == TaskStatus.FAILED),
            "pending": sum(1 for t in self.tasks if t.status == TaskStatus.PENDING),
            "in_progress": sum(1 for t in self.tasks if t.status == TaskStatus.IN_PROGRESS),
            "is_complete": self.is_complete(),
        }


# ============================================================================
# 테스트 코드 (주석)
# ============================================================================
# from src.task_planner import TaskStatus, Subtask
#
# # TaskStatus 테스트
# print(TaskStatus.PENDING.value)  # "pending"
#
# # Subtask 테스트
# task = Subtask(
#     id="task_1",
#     description="시장 규모 조사",
#     priority=1,
#     dependencies=[]
# )
# print(f"Task: {task.id}, Status: {task.status.value}")
