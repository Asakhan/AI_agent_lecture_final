"""
자율 실행 오케스트레이터
TaskPlanner, ReActEngine, QualityManager를 조율하여 복잡한 작업을 수행합니다.
"""
from typing import Dict, Any, List, Optional, Callable
from openai import OpenAI
import logging

from src.task_planner import TaskPlanner, TaskStatus
from src.react_engine import ReActEngine
from src.quality_manager import QualityManager
from src.search_agent import SearchAgent
from src.memory_manager import MemoryManager
from config.prompts import SYNTHESIS_PROMPT

logger = logging.getLogger(__name__)


class AutonomousOrchestrator:
    """자율 실행 오케스트레이터"""

    def __init__(
        self,
        client: OpenAI,
        memory_manager: Optional[MemoryManager] = None,
        search_agent: Optional[SearchAgent] = None,
    ):
        self.client = client

        self.memory_manager = memory_manager or MemoryManager(
            collection_name="orchestrator_memory",
            persist_directory="data/chroma_db",
        )
        self.search_agent = search_agent or SearchAgent(
            memory_manager=self.memory_manager
        )

        self.task_planner = TaskPlanner(client)
        self.quality_manager = QualityManager(client)

        self.tools = self._create_tool_registry()
        self.react_engine = ReActEngine(client, self.tools)

        self.execution_log: List[Dict] = []
        logger.info("AutonomousOrchestrator initialized")

    def _create_tool_registry(self) -> Dict[str, Callable]:
        """도구 레지스트리 생성"""

        def search_web(query: str) -> str:
            result = self.search_agent.search(query)
            return self.search_agent.format_for_llm(result)

        def search_memory(query: str) -> str:
            results = self.memory_manager.search_memory(query, top_k=5)
            if not results:
                return "메모리에서 관련 정보를 찾지 못했습니다."
            return "\n".join([f"- {r['text'][:200]}..." for r in results])

        def store_knowledge(text: str) -> str:
            doc_id = self.memory_manager.add_to_memory(
                text, metadata={"source": "autonomous_agent"}
            )
            return f"지식 저장 완료 (ID: {doc_id})"

        def analyze(data: str) -> str:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "정보 분석 전문가입니다. 핵심을 요약하세요.",
                    },
                    {"role": "user", "content": f"분석해주세요:\n\n{data}"},
                ],
                temperature=0.5,
            )
            return response.choices[0].message.content

        return {
            "search_web": search_web,
            "search_memory": search_memory,
            "store_knowledge": store_knowledge,
            "analyze": analyze,
        }

    def execute(self, goal: str, verbose: bool = True) -> str:
        """목표 실행"""
        if verbose:
            print(f"\n🎯 목표: {goal}")
            print("=" * 60)

        # 1. 작업 분해
        if verbose:
            print("\n📋 Step 1: 작업 분해 중...")
        self.task_planner.decompose(goal)
        if verbose:
            self.task_planner.visualize()

        # 2. 서브태스크 순차 실행
        results = []

        while not self.task_planner.is_complete():
            task = self.task_planner.get_next_task()
            if not task:
                break

            if verbose:
                print(f"\n🔄 실행 중: [{task.id}] {task.description}")

            result = self._execute_with_quality(task.description, verbose)

            if result:
                self.task_planner.update_status(
                    task.id, TaskStatus.COMPLETED, result
                )
                results.append(
                    {
                        "task_id": task.id,
                        "description": task.description,
                        "result": result,
                    }
                )
                if verbose:
                    print(f"✅ [{task.id}] 완료")
            else:
                self.task_planner.update_status(task.id, TaskStatus.FAILED)
                if verbose:
                    print(f"❌ [{task.id}] 실패")

        # 3. 결과 종합
        if verbose:
            print("\n📝 Step 3: 결과 종합 중...")

        final_report = self._synthesize_results(results, goal)

        self.execution_log.append(
            {
                "goal": goal,
                "tasks": self.task_planner.get_summary(),
                "results_count": len(results),
            }
        )

        if verbose:
            print("\n" + "=" * 60)
            print("✨ 작업 완료!")
            self.task_planner.visualize()

        return final_report

    def _execute_with_quality(
        self, task_description: str, verbose: bool = True
    ) -> Optional[str]:
        """품질 평가와 함께 태스크 실행"""
        result = None
        for attempt in range(self.quality_manager.max_retries):
            self.react_engine.reset()
            result = self.react_engine.run(task_description)

            evaluation = self.quality_manager.evaluate(
                task_description, result
            )

            if verbose:
                print(f"   품질 점수: {evaluation['overall']:.1f}/10")

            if evaluation["pass"]:
                return result

            if attempt < self.quality_manager.max_retries - 1:
                if verbose:
                    print(
                        f"   재시도 중... ({attempt + 2}/{self.quality_manager.max_retries})"
                    )
                task_description = self.quality_manager.get_improvement_prompt(
                    task_description, result, evaluation
                )

        return result

    def _synthesize_results(self, results: List[Dict], goal: str) -> str:
        """결과 종합"""
        if not results:
            return "수집된 결과가 없습니다."

        results_text = ""
        for r in results:
            results_text += f"\n### {r['description']}\n{r['result']}\n"

        prompt = SYNTHESIS_PROMPT.format(goal=goal, results=results_text)

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "리서치 결과 종합 전문가입니다."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
        )

        return response.choices[0].message.content

    def get_stats(self) -> Dict[str, Any]:
        """실행 통계 반환"""
        return {
            "total_executions": len(self.execution_log),
            "quality_stats": self.quality_manager.get_stats(),
            "last_execution": (
                self.execution_log[-1] if self.execution_log else None
            ),
        }
