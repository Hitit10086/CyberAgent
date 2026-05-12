"""
Perception Agent: 场景解析

负责将原始仿真场景数据解析为结构化场景图 (SceneGraph)。
识别道路拓扑、交通参与者、信号灯状态等关键元素。
"""

from dataclasses import dataclass
from typing import Optional

from core.message_bus import MessageBus, MessageType
from core.scenario import Scenario, SceneGraph, PerceptionOutput, Obstacle, Lane, TrafficLight
from core.token_tracker import TokenTracker


@dataclass
class PerceptionConfig:
    max_detection_range: float = 150.0  # 最大检测范围 (米)
    object_classification_threshold: float = 0.85


class PerceptionAgent:
    """场景解析 Agent — 多 Agent 协作链的第一环"""

    name = "perception"
    description = "将原始仿真场景数据解析为结构化场景图"

    def __init__(self, message_bus: MessageBus, token_tracker: TokenTracker,
                 config: Optional[PerceptionConfig] = None):
        self.message_bus = message_bus
        self.token_tracker = token_tracker
        self.config = config or PerceptionConfig()

    def process(self, scenario: Scenario) -> PerceptionOutput:
        """
        解析仿真场景，输出结构化 SceneGraph

        核心逻辑:
        1. 解析道路网络拓扑 (车道线、交叉口、匝道)
        2. 识别动态/静态障碍物并分类
        3. 提取交通信号灯状态
        4. 构建场景图数据结构
        """
        import time
        t0 = time.time()

        raw = scenario.raw_data

        # 解析车道网络
        lanes = [
            Lane(
                lane_id=l.get("id", f"lane_{i}"),
                points=[(p["x"], p["y"]) for p in l.get("points", [])],
                lane_type=l.get("type", "driving"),
                speed_limit=l.get("speed_limit", 60.0),
            )
            for i, l in enumerate(raw.get("lanes", []))
        ]

        # 解析障碍物
        obstacles = [
            Obstacle(
                obstacle_id=o.get("id", f"obs_{i}"),
                obstacle_type=o.get("type", "vehicle"),
                position=(o["position"]["x"], o["position"]["y"]),
                velocity=(o.get("velocity", {}).get("x", 0.0),
                         o.get("velocity", {}).get("y", 0.0)),
            )
            for i, o in enumerate(raw.get("obstacles", []))
        ]

        # 解析交通信号灯
        traffic_lights = [
            TrafficLight(
                light_id=tl.get("id", f"tl_{i}"),
                state=tl.get("state", "red"),
                position=(tl["position"]["x"], tl["position"]["y"]),
                remaining_time=tl.get("remaining_time", 0.0),
            )
            for i, tl in enumerate(raw.get("traffic_lights", []))
        ]

        ego = raw.get("ego_vehicle", {})
        scene_graph = SceneGraph(
            lanes=lanes,
            obstacles=obstacles,
            traffic_lights=traffic_lights,
            ego_position=(ego.get("position", {}).get("x", 0.0),
                          ego.get("position", {}).get("y", 0.0)),
            ego_heading=ego.get("heading", 0.0),
            destination=(ego.get("destination", {}).get("x", 0.0),
                        ego.get("destination", {}).get("y", 0.0)),
        )

        # 模拟 Token 消耗记录
        prompt_tokens = len(str(raw)) // 4 + 500
        completion_tokens = len(str(scene_graph)) // 4 + 200
        self.token_tracker.record(self.name, "perception",
                                  prompt_tokens, completion_tokens)

        processing_ms = (time.time() - t0) * 1000

        # 将解析结果广播到消息总线
        output = PerceptionOutput(
            scene_graph=scene_graph,
            detected_objects=len(obstacles),
            confidence=0.92,
            processing_time_ms=processing_ms,
        )
        self.message_bus.broadcast(self.name, MessageType.SCENE_GRAPH_READY, output)
        return output
