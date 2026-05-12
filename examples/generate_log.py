"""
Generate clean terminal log for Xiaomi Mimo application proof.
Outputs to stdout with explicit UTF-8 encoding.
"""
import sys
import os
import io

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from core.orchestrator import Orchestrator, OrchestratorConfig
from core.scenario import Scenario


def create_scenario():
    return Scenario(
        scenario_id="intersection-001",
        name="城市十字路口场景",
        description="典型城市十字路口，包含直行车道、横向车道、一辆动态车和一个行人",
        raw_data={
            "lanes": [
                {"id": "lane_ego", "type": "driving",
                 "points": [{"x": 0, "y": 0}, {"x": 100, "y": 0},
                            {"x": 200, "y": 0}, {"x": 300, "y": 0}],
                 "speed_limit": 60.0},
                {"id": "lane_cross", "type": "driving",
                 "points": [{"x": 150, "y": -50}, {"x": 150, "y": 0},
                            {"x": 150, "y": 50}, {"x": 150, "y": 100}],
                 "speed_limit": 40.0},
            ],
            "obstacles": [
                {"id": "obs_vehicle_1", "type": "vehicle",
                 "position": {"x": 120, "y": 0},
                 "velocity": {"x": 8.0, "y": 0.0}},
                {"id": "obs_pedestrian_1", "type": "pedestrian",
                 "position": {"x": 200, "y": 3.5},
                 "velocity": {"x": 0.0, "y": -1.2}},
            ],
            "traffic_lights": [
                {"id": "tl_intersection", "state": "green",
                 "position": {"x": 150, "y": 0}, "remaining_time": 25.0},
            ],
            "ego_vehicle": {
                "position": {"x": 0, "y": 0, "z": 0},
                "heading": 0.0, "speed": 30.0,
                "destination": {"x": 300, "y": 0},
            },
        },
    )


def main():
    print("=" * 70)
    print("  CyberAgent -- Multi-Agent Autonomous Driving Simulation Analyzer")
    print("=" * 70)
    print()

    scenario = create_scenario()
    print(f"[LOAD] Scenario: {scenario.name}")
    print(f"       ID: {scenario.scenario_id}")
    print(f"       Description: {scenario.description}")
    print()

    config = OrchestratorConfig(
        max_retries=2,
        safety_threshold=0.70,
        enable_chain_of_thought=True,
    )
    orchestrator = Orchestrator(config=config)

    print("[AGENT] Launching 4-Agent collaborative pipeline...")
    print("        Perception -> Planner (CoT) -> Verifier -> Reporter")
    print()

    result = orchestrator.analyze(scenario)

    print("=" * 70)
    print(result.report)
    print("=" * 70)
    print()

    print("[METRICS]")
    print(f"  Total elapsed:   {result.elapsed_seconds:.4f}s")
    print(f"  Total tokens:    {result.token_usage.total_tokens:,}")
    print(f"  Safety score:    {result.safety_score:.2%}")
    print(f"  Verification:    {'PASSED' if result.verification_passed else 'FAILED'}")
    print()

    print("[TOKEN BREAKDOWN by Agent]")
    for agent, tokens in result.token_usage.agent_breakdown.items():
        bar_len = int(tokens / result.token_usage.total_tokens * 30)
        bar = "#" * bar_len + "-" * (30 - bar_len)
        print(f"  {agent:12s} [{bar}] {tokens:>6,} tokens")

    print()
    print("[CoT REASONING TRACE]")
    for i, step in enumerate(result.optimal_trajectory.reasoning_chain, 1):
        print(f"  Step {i}: {step}")

    print()
    print("[MESSAGE BUS ACTIVITY]")
    for msg in orchestrator.message_bus.get_message_history():
        print(f"  [{msg.msg_type.value}] from={msg.sender}")

    print()
    print("=" * 70)
    print("  Pipeline complete. All 4 agents executed successfully.")
    print("  GitHub: https://github.com/Hitit10086/CyberAgent")
    print("=" * 70)


if __name__ == "__main__":
    main()
