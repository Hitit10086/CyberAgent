"""
示例: 使用 CyberAgent 进行自动驾驶仿真场景分析

演示完整的多 Agent 协作流程。
"""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.orchestrator import Orchestrator, OrchestratorConfig
from core.scenario import Scenario


def create_example_scenario() -> Scenario:
    """创建一个十字路口仿真场景示例"""
    raw_data = {
        "lanes": [
            {
                "id": "lane_ego", "type": "driving",
                "points": [
                    {"x": 0, "y": 0}, {"x": 100, "y": 0},
                    {"x": 200, "y": 0}, {"x": 300, "y": 0},
                ],
                "speed_limit": 60.0,
            },
            {
                "id": "lane_cross", "type": "driving",
                "points": [
                    {"x": 150, "y": -50}, {"x": 150, "y": 0},
                    {"x": 150, "y": 50}, {"x": 150, "y": 100},
                ],
                "speed_limit": 40.0,
            },
        ],
        "obstacles": [
            {
                "id": "obs_vehicle_1", "type": "vehicle",
                "position": {"x": 120, "y": 0},
                "velocity": {"x": 8.0, "y": 0.0},
            },
            {
                "id": "obs_pedestrian_1", "type": "pedestrian",
                "position": {"x": 200, "y": 3.5},
                "velocity": {"x": 0.0, "y": -1.2},
            },
        ],
        "traffic_lights": [
            {
                "id": "tl_intersection",
                "state": "green",
                "position": {"x": 150, "y": 0},
                "remaining_time": 25.0,
            },
        ],
        "ego_vehicle": {
            "position": {"x": 0, "y": 0, "z": 0},
            "heading": 0.0,
            "speed": 30.0,
            "destination": {"x": 300, "y": 0},
        },
    }

    return Scenario(
        scenario_id="intersection-001",
        name="城市十字路口场景",
        description="典型城市十字路口，包含直行车道、横向车道、一辆动态车和一个行人",
        raw_data=raw_data,
    )


def main():
    print("=" * 60)
    print("CyberAgent — 多 Agent 自动驾驶仿真场景分析")
    print("=" * 60)
    print()

    # 1. 创建场景
    scenario = create_example_scenario()
    print(f"[LOAD] Loading scenario: {scenario.name}")
    print(f"   ID: {scenario.scenario_id}")
    print()

    # 2. 配置并运行多 Agent 协作分析
    config = OrchestratorConfig(
        max_retries=2,
        safety_threshold=0.70,
        enable_chain_of_thought=True,
    )
    orchestrator = Orchestrator(config=config)

    print("[AGENT] Launching multi-agent collaborative analysis...")
    print("   Perception -> Planner (CoT) -> Verifier -> Reporter")
    print()

    result = orchestrator.analyze(scenario)

    # 3. 输出报告
    print("=" * 60)
    print(result.report)
    print("=" * 60)
    print()

    print(f"[TIME] Elapsed: {result.elapsed_seconds:.3f}s")
    print(f"[TOKEN] Total token usage: {result.token_usage.total_tokens:,}")
    print(f"[SAFETY] Safety score: {result.safety_score:.2%}")
    print(f"[VERIFY] Verification: {'PASSED' if result.verification_passed else 'FAILED'}")
    print()

    # 4. Token 消耗详情
    print("[TOKEN] Token breakdown by agent:")
    for agent, tokens in result.token_usage.agent_breakdown.items():
        print(f"   {agent}: {tokens:,} tokens")

    # 5. 推理链展示
    print()
    print("[CoT] Chain-of-Thought reasoning trace:")
    for step in result.optimal_trajectory.reasoning_chain:
        print(f"   {step}")


if __name__ == "__main__":
    main()
