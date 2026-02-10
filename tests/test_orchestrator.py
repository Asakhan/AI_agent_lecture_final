"""AutonomousOrchestrator 통합 테스트"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from src.loop_prevention import LoopPrevention
from src.quality_manager import QualityManager
from src.react_engine import ReActEngine
from src.orchestrator import AutonomousOrchestrator


def test_loop_prevention():
    """LoopPrevention 테스트"""
    print("\n" + "=" * 50)
    print("TEST 1: Loop Prevention")
    print("=" * 50)

    lp = LoopPrevention(max_iterations=5, max_same_action=3)

    result = lp.check_loop("생각1", {"action": "search_web", "action_input": "AI"})
    assert not result["is_loop"]
    print("✅ 정상 액션 통과")

    lp.check_loop("생각2", {"action": "analyze", "action_input": "data"})

    for i in range(3):
        result = lp.check_loop(
            f"생각{i+3}", {"action": "search_web", "action_input": "same"}
        )

    assert result["is_loop"]
    print("✅ 동일 액션 연속 감지")

    lp.reset()
    for i in range(6):
        result = lp.check_loop(
            f"생각{i}", {"action": f"action_{i}", "action_input": str(i)}
        )

    assert result["should_stop"]
    print("✅ 최대 반복 감지")

    return True


def test_quality_manager():
    """QualityManager 테스트"""
    print("\n" + "=" * 50)
    print("TEST 2: Quality Manager")
    print("=" * 50)

    client = OpenAI()
    qm = QualityManager(client, min_quality_score=7.0)

    good_result = """
    AI 반도체 시장 분석:
    1. 시장 규모: 2024년 약 500억 달러
    2. 성장률: 연평균 25%
    3. 주요 기업: NVIDIA(70%), AMD, Intel
    """

    eval1 = qm.evaluate("AI 반도체 시장 조사", good_result)
    print(f"좋은 결과 점수: {eval1['overall']:.1f}/10")
    assert eval1["overall"] >= 6
    print("✅ 좋은 결과 평가 완료")

    poor_result = "AI 시장이 성장 중입니다."
    eval2 = qm.evaluate("AI 반도체 시장 조사", poor_result)
    print(f"부족한 결과 점수: {eval2['overall']:.1f}/10")
    print("✅ 부족한 결과 평가 완료")

    return True


def test_react_engine():
    """ReActEngine 테스트"""
    print("\n" + "=" * 50)
    print("TEST 3: ReAct Engine")
    print("=" * 50)

    client = OpenAI()

    def mock_search(query):
        return f"'{query}' 검색 결과: AI 시장은 500억 달러 규모입니다."

    def mock_analyze(data):
        return f"분석 완료: {data[:50]}..."

    tools = {"search_web": mock_search, "analyze": mock_analyze}

    engine = ReActEngine(client, tools, max_iterations=5)
    result = engine.run("AI 시장 규모를 조사하세요")

    assert result is not None
    print(f"✅ ReAct 실행 완료 (스텝 수: {len(engine.history)})")

    return True


def test_orchestrator():
    """Orchestrator 테스트"""
    print("\n" + "=" * 50)
    print("TEST 4: Orchestrator")
    print("=" * 50)

    client = OpenAI()
    orchestrator = AutonomousOrchestrator(client)

    print("🎯 간단한 목표로 테스트 중...")
    result = orchestrator.execute(
        "Python의 장단점을 3가지씩 정리해주세요", verbose=False
    )

    assert result is not None
    assert len(result) > 50
    print(f"✅ 결과 생성됨 (길이: {len(result)}자)")

    stats = orchestrator.get_stats()
    print(f"   총 실행 횟수: {stats['total_executions']}")
    print("✅ Orchestrator 테스트 완료")

    return True


def run_all_tests():
    """모든 테스트 실행"""
    print("\n🧪 Part 2 통합 테스트 시작 " + "=" * 30)

    tests = [
        ("Loop Prevention", test_loop_prevention),
        ("Quality Manager", test_quality_manager),
        ("ReAct Engine", test_react_engine),
        ("Orchestrator", test_orchestrator),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, "PASS" if result else "FAIL"))
        except Exception as e:
            print(f"❌ {name} 실패: {e}")
            import traceback

            traceback.print_exc()
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
    success = run_all_tests()
    sys.exit(0 if success else 1)
