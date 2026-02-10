"""TaskPlanner 단위 테스트"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from src.task_planner import TaskPlanner, TaskStatus, Subtask


def test_subtask_creation():
    """Subtask 생성 테스트"""
    print("\n" + "=" * 50)
    print("TEST 1: Subtask Creation")
    print("=" * 50)

    task = Subtask(id="task_1", description="테스트 태스크", priority=1)

    assert task.id == "task_1"
    assert task.status == TaskStatus.PENDING
    assert task.attempts == 0
    print("✅ 기본 Subtask 생성 성공")

    task2 = Subtask(id="task_2", description="의존성 태스크", priority=2, dependencies=["task_1"])
    assert "task_1" in task2.dependencies
    print("✅ 의존성 Subtask 생성 성공")

    return True


def test_task_decomposition():
    """작업 분해 테스트"""
    print("\n" + "=" * 50)
    print("TEST 2: Task Decomposition")
    print("=" * 50)

    client = OpenAI()
    planner = TaskPlanner(client)

    tasks = planner.decompose("전기차 시장 동향 분석")

    assert len(tasks) >= 3
    assert len(tasks) <= 7
    print(f"✅ {len(tasks)}개 태스크로 분해됨")

    for task in tasks:
        assert task.id.startswith("task_")
        assert len(task.description) > 0
        print(f"  - {task.id}: {task.description[:40]}...")

    return True


def test_dependency_management():
    """의존성 관리 테스트"""
    print("\n" + "=" * 50)
    print("TEST 3: Dependency Management")
    print("=" * 50)

    client = OpenAI()
    planner = TaskPlanner(client)

    planner.tasks = [
        Subtask(id="task_1", description="조사", priority=1),
        Subtask(id="task_2", description="분석", priority=2, dependencies=["task_1"]),
        Subtask(id="task_3", description="정리", priority=3, dependencies=["task_1", "task_2"]),
    ]
    planner.original_goal = "테스트"

    next_task = planner.get_next_task()
    assert next_task.id == "task_1"
    print("✅ 의존성 없는 태스크 우선 선택")

    planner.tasks[0].status = TaskStatus.PENDING
    task2 = planner._get_task_by_id("task_2")
    assert not planner._dependencies_met(task2)
    print("✅ 미충족 의존성 정상 감지")

    planner.tasks[0].status = TaskStatus.COMPLETED
    assert planner._dependencies_met(task2)
    print("✅ 충족 의존성 정상 감지")

    return True


def test_task_execution_flow():
    """태스크 실행 흐름 테스트"""
    print("\n" + "=" * 50)
    print("TEST 4: Task Execution Flow")
    print("=" * 50)

    client = OpenAI()
    planner = TaskPlanner(client)

    planner.decompose("Python 튜토리얼 작성")
    print(f"📋 총 {len(planner.tasks)}개 태스크")
    planner.visualize()

    iteration = 0
    while not planner.is_complete() and iteration < 20:
        task = planner.get_next_task()
        if not task:
            break

        print(f"🔄 실행 중: {task.id}")
        planner.update_status(task.id, TaskStatus.COMPLETED, f"{task.id} 결과")
        iteration += 1

    planner.visualize()
    assert planner.is_complete()
    print("✅ 전체 실행 흐름 테스트 성공")

    return True


def test_retry_mechanism():
    """재시도 메커니즘 테스트"""
    print("\n" + "=" * 50)
    print("TEST 5: Retry Mechanism")
    print("=" * 50)

    client = OpenAI()
    planner = TaskPlanner(client)

    planner.tasks = [
        Subtask(id="task_1", description="실패할 태스크", priority=1, max_attempts=3),
    ]
    planner.original_goal = "재시도 테스트"

    task = planner.get_next_task()
    assert task.attempts == 1
    print("✅ 1차 시도")

    planner.update_status(task.id, TaskStatus.FAILED)

    task = planner.get_next_task()
    assert task.attempts == 2
    print("✅ 2차 시도")

    planner.update_status(task.id, TaskStatus.FAILED)
    task = planner.get_next_task()
    assert task.attempts == 3
    print("✅ 3차 시도")

    planner.update_status(task.id, TaskStatus.FAILED)
    task = planner.get_next_task()
    assert task is None
    print("✅ max_attempts 초과 후 재시도 차단")

    return True


def run_all_tests():
    """모든 테스트 실행"""
    print("\n🧪 TaskPlanner 단위 테스트 시작 " + "=" * 30)

    tests = [
        ("Subtask Creation", test_subtask_creation),
        ("Task Decomposition", test_task_decomposition),
        ("Dependency Management", test_dependency_management),
        ("Execution Flow", test_task_execution_flow),
        ("Retry Mechanism", test_retry_mechanism),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, "PASS" if result else "FAIL"))
        except Exception as e:
            print(f"❌ {name} 실패: {e}")
            results.append((name, "ERROR"))

    print("\n" + "=" * 50)
    print("📊 테스트 결과 요약")
    print("=" * 50)

    for name, status in results:
        icon = "✅" if status == "PASS" else "❌"
        print(f"{icon} {name}: {status}")

    passed = sum(1 for _, s in results if s == "PASS")
    print(f"\n총 결과: {passed}/{len(results)} 통과")

    return passed == len(results)


if __name__ == "__main__":
    import sys

    success = run_all_tests()
    sys.exit(0 if success else 1)
