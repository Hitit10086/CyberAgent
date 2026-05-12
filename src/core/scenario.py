"""
Scenario: 仿真场景数据结构

定义自动驾驶仿真场景的标准数据模型，兼容 Apollo/SimOne 格式。
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Waypoint:
    x: float
    y: float
    z: float = 0.0
    speed_limit: float = 60.0


@dataclass
class Obstacle:
    obstacle_id: str
    obstacle_type: str  # vehicle, pedestrian, cyclist, static
    position: tuple[float, float]
    velocity: tuple[float, float] = (0.0, 0.0)
    polygon: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class TrafficLight:
    light_id: str
    state: str  # red, yellow, green
    position: tuple[float, float]
    remaining_time: float = 0.0


@dataclass
class Lane:
    lane_id: str
    points: list[tuple[float, float]]
    lane_type: str = "driving"  # driving, turning, merging
    speed_limit: float = 60.0


@dataclass
class SceneGraph:
    """结构化场景图 — Perception Agent 的输出"""
    lanes: list[Lane] = field(default_factory=list)
    obstacles: list[Obstacle] = field(default_factory=list)
    traffic_lights: list[TrafficLight] = field(default_factory=list)
    ego_position: tuple[float, float] = (0.0, 0.0)
    ego_heading: float = 0.0
    destination: tuple[float, float] = (0.0, 0.0)
    timestamp: float = 0.0


@dataclass
class Trajectory:
    """规划轨迹"""
    waypoints: list[Waypoint] = field(default_factory=list)
    speeds: list[float] = field(default_factory=list)
    reasoning_chain: list[str] = field(default_factory=list)  # CoT 推理步骤
    confidence: float = 0.0


@dataclass
class Scenario:
    """仿真场景定义"""
    scenario_id: str
    name: str
    description: str = ""
    scene_graph: Optional[SceneGraph] = None
    raw_data: dict = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str) -> "Scenario":
        """从 JSON 文件加载场景"""
        import json
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class PerceptionOutput:
    scene_graph: SceneGraph
    detected_objects: int
    confidence: float
    processing_time_ms: float


@dataclass
class PlannerOutput:
    trajectory: Trajectory
    alternatives: list[Trajectory] = field(default_factory=list)
    reasoning_summary: str = ""


@dataclass
class VerificationResult:
    safety_score: float  # 0.0 - 1.0
    collision_risk: float
    traffic_rule_compliance: float
    feedback: str = ""
    passed: bool = False


@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    agent_breakdown: dict[str, int] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    safety_score: float
    optimal_trajectory: Trajectory
    report: str
    token_usage: TokenUsage
    elapsed_seconds: float
    verification_passed: bool
