"""
CyberAgent 测试套件

覆盖 Multi-Agent 协作全流程: Perception → Planner → Verifier → Reporter
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from core.scenario import Scenario, SceneGraph, Obstacle, Lane, TrafficLight
from core.message_bus import MessageBus, Message, MessageType
from core.token_tracker import TokenTracker
from core.orchestrator import Orchestrator, OrchestratorConfig
from agents.perception import PerceptionAgent
from agents.planner import PlannerAgent
from agents.verifier import VerifierAgent
from agents.reporter import ReporterAgent


class TestMessageBus:
    """消息总线测试"""

    def test_register_agent(self):
        bus = MessageBus()
        bus.register("test_agent", object())
        assert bus.get_agent("test_agent") is not None

    def test_publish_subscribe(self):
        bus = MessageBus()
        received = []

        def handler(msg: Message):
            received.append(msg)

        bus.subscribe("test", MessageType.SCENE_GRAPH_READY, handler)
        msg = Message(msg_type=MessageType.SCENE_GRAPH_READY,
                      sender="sender", payload={"data": "test"})
        bus.publish(msg)
        assert len(received) == 1
        assert received[0].payload["data"] == "test"

    def test_broadcast(self):
        bus = MessageBus()
        received = []

        bus.subscribe("test", MessageType.TRAJECTORY_PROPOSED, lambda m: received.append(m))
        bus.broadcast("sender", MessageType.TRAJECTORY_PROPOSED, {"x": 1})
        assert len(received) == 1


class TestTokenTracker:
    """Token 追踪器测试"""

    def test_record_and_summary(self):
        tracker = TokenTracker()
        tracker.record("perception", "perception", 500, 200)
        tracker.record("planner", "planning", 1200, 800)

        summary = tracker.summary()
        assert summary.prompt_tokens == 1700
        assert summary.completion_tokens == 1000
        assert summary.total_tokens == 2700
        assert "perception" in summary.agent_breakdown
        assert summary.agent_breakdown["perception"] == 700

    def test_phase_breakdown(self):
        tracker = TokenTracker()
        tracker.record("agent1", "phase_a", 100, 50)
        tracker.record("agent1", "phase_b", 200, 100)

        breakdown = tracker.phase_breakdown()
        assert breakdown["phase_a"] == 150
        assert breakdown["phase_b"] == 300


class TestScenario:
    """场景数据模型测试"""

    def test_scene_graph_creation(self):
        lane = Lane(lane_id="L1", points=[(0, 0), (100, 0)])
        obs = Obstacle(obstacle_id="O1", obstacle_type="vehicle",
                       position=(50, 0))
        sg = SceneGraph(lanes=[lane], obstacles=[obs],
                        ego_position=(0, 0), destination=(100, 0))
        assert len(sg.lanes) == 1
        assert len(sg.obstacles) == 1

    def test_scenario_from_file(self):
        scenario = Scenario(
            scenario_id="test-001",
            name="测试场景",
            raw_data={"lanes": [], "obstacles": [], "traffic_lights": [],
                      "ego_vehicle": {"position": {"x": 0, "y": 0},
                                      "heading": 0.0,
                                      "destination": {"x": 100, "y": 0}}},
        )
        assert scenario.scenario_id == "test-001"


class TestPerceptionAgent:
    """Perception Agent 测试"""

    def test_process_empty_scenario(self):
        bus = MessageBus()
        tracker = TokenTracker()
        agent = PerceptionAgent(bus, tracker)

        scenario = Scenario(
            scenario_id="empty",
            name="空场景",
            raw_data={
                "lanes": [], "obstacles": [],
                "traffic_lights": [],
                "ego_vehicle": {"position": {"x": 0, "y": 0},
                                "heading": 0.0,
                                "destination": {"x": 100, "y": 0}},
            },
        )

        output = agent.process(scenario)
        assert output.detected_objects == 0
        assert len(output.scene_graph.lanes) == 0
        assert output.confidence > 0

    def test_process_with_obstacles(self):
        bus = MessageBus()
        tracker = TokenTracker()
        agent = PerceptionAgent(bus, tracker)

        scenario = Scenario(
            scenario_id="obs-test",
            name="障碍物场景",
            raw_data={
                "lanes": [
                    {"id": "L1", "type": "driving",
                     "points": [{"x": 0, "y": 0}, {"x": 100, "y": 0}],
                     "speed_limit": 60.0},
                ],
                "obstacles": [
                    {"id": "O1", "type": "vehicle",
                     "position": {"x": 50, "y": 0},
                     "velocity": {"x": 5.0, "y": 0.0}},
                ],
                "traffic_lights": [],
                "ego_vehicle": {"position": {"x": 0, "y": 0},
                                "heading": 0.0,
                                "destination": {"x": 100, "y": 0}},
            },
        )

        output = agent.process(scenario)
        assert output.detected_objects == 1
        assert len(output.scene_graph.lanes) == 1
        assert len(output.scene_graph.obstacles) == 1


class TestPlannerAgent:
    """Planner Agent 测试"""

    def test_plan_with_cot(self):
        bus = MessageBus()
        tracker = TokenTracker()
        agent = PlannerAgent(bus, tracker)

        sg = SceneGraph(
            lanes=[Lane(lane_id="L1", points=[(0, 0), (100, 0)])],
            obstacles=[Obstacle(obstacle_id="O1", obstacle_type="vehicle",
                                position=(50, 0))],
            ego_position=(0, 0),
            destination=(100, 0),
        )

        output = agent.plan(sg, enable_cot=True)
        assert len(output.trajectory.reasoning_chain) == 5  # 5步CoT
        assert output.trajectory.confidence > 0
        assert len(output.trajectory.waypoints) > 0

    def test_plan_without_cot(self):
        bus = MessageBus()
        tracker = TokenTracker()
        agent = PlannerAgent(bus, tracker)

        sg = SceneGraph(
            lanes=[],
            ego_position=(0, 0),
            destination=(50, 0),
        )

        output = agent.plan(sg, enable_cot=False)
        assert len(output.trajectory.waypoints) > 0
        assert "直接规划" in output.reasoning_summary


class TestVerifierAgent:
    """Verifier Agent 测试"""

    def test_verify_empty_scene(self):
        bus = MessageBus()
        tracker = TokenTracker()
        agent = VerifierAgent(bus, tracker)

        from core.scenario import Trajectory, Waypoint

        traj = Trajectory(waypoints=[], speeds=[], confidence=1.0)
        sg = SceneGraph(
            lanes=[],
            obstacles=[],
            ego_position=(0, 0),
            destination=(100, 0),
        )

        result = agent.verify(traj, sg)
        assert result.safety_score >= 0.99
        assert result.passed

    def test_verify_with_collision_risk(self):
        bus = MessageBus()
        tracker = TokenTracker()
        agent = VerifierAgent(bus, tracker)

        from core.scenario import Trajectory, Waypoint

        traj = Trajectory(
            waypoints=[Waypoint(x=50, y=0)],
            speeds=[30.0],
            confidence=0.8,
        )
        sg = SceneGraph(
            lanes=[],
            obstacles=[Obstacle(obstacle_id="O1", obstacle_type="vehicle",
                                position=(50.1, 0.1))],
            ego_position=(0, 0),
            destination=(100, 0),
        )

        result = agent.verify(traj, sg)
        assert result.safety_score < 1.0  # 存在碰撞风险


class TestReporterAgent:
    """Reporter Agent 测试"""

    def test_generate_markdown_report(self):
        bus = MessageBus()
        tracker = TokenTracker()

        # 先模拟前置 Agent 消耗 (这样 summary 中有数据)
        tracker.record("perception", "perception", 500, 200)
        tracker.record("planner", "planning", 800, 400)

        agent = ReporterAgent(bus, tracker)

        from core.scenario import (Scenario, PerceptionOutput, SceneGraph,
                                    PlannerOutput, Trajectory, VerificationResult,
                                    Waypoint)

        scenario = Scenario(
            scenario_id="test-rpt",
            name="报告测试",
            raw_data={},
        )
        perception = PerceptionOutput(
            scene_graph=SceneGraph(
                lanes=[],
                obstacles=[],
                ego_position=(0, 0),
                destination=(100, 0),
            ),
            detected_objects=2,
            confidence=0.92,
            processing_time_ms=15.0,
        )
        planner = PlannerOutput(
            trajectory=Trajectory(waypoints=[], confidence=0.85),
            reasoning_summary="推理完成",
        )
        verification = VerificationResult(
            safety_score=0.92,
            collision_risk=0.05,
            traffic_rule_compliance=0.95,
            feedback="所有检查通过",
            passed=True,
        )

        report = agent.generate(scenario, perception, planner, verification)
        assert "报告测试" in report
        assert "92.0%" in report
        assert "[PASS]" in report


class TestOrchestratorIntegration:
    """多 Agent 协作集成测试"""

    def test_full_pipeline(self):
        """完整的多 Agent 协作流水线测试"""
        config = OrchestratorConfig(max_retries=1, enable_chain_of_thought=True)
        orchestrator = Orchestrator(config=config)

        scenario = Scenario(
            scenario_id="integration-test",
            name="集成测试场景",
            raw_data={
                "lanes": [
                    {"id": "L1", "type": "driving",
                     "points": [{"x": 0, "y": 0}, {"x": 100, "y": 0}],
                     "speed_limit": 60.0},
                ],
                "obstacles": [],
                "traffic_lights": [
                    {"id": "TL1", "state": "green",
                     "position": {"x": 50, "y": 0}, "remaining_time": 15.0},
                ],
                "ego_vehicle": {
                    "position": {"x": 0, "y": 0},
                    "heading": 0.0,
                    "destination": {"x": 100, "y": 0},
                },
            },
        )

        result = orchestrator.analyze(scenario)

        # 验证完整结果
        assert result.elapsed_seconds >= 0
        assert result.token_usage.total_tokens > 0
        assert result.safety_score >= 0.0
        assert result.safety_score <= 1.0
        assert len(result.report) > 0
        assert len(result.token_usage.agent_breakdown) == 4  # 4个Agent

    def test_batch_analysis(self):
        """批量分析测试"""
        orchestrator = Orchestrator()

        scenarios = [
            Scenario(
                scenario_id=f"batch-{i}",
                name=f"批量场景{i}",
                raw_data={
                    "lanes": [],
                    "obstacles": [],
                    "traffic_lights": [],
                    "ego_vehicle": {"position": {"x": 0, "y": 0},
                                    "heading": 0.0,
                                    "destination": {"x": 100, "y": 0}},
                },
            )
            for i in range(3)
        ]

        results = orchestrator.analyze_batch(scenarios)
        assert len(results) == 3
        for r in results:
            assert r.safety_score >= 0.0
