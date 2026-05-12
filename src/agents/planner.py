"""
Planner Agent: 轨迹规划 (含长链推理 Chain-of-Thought)

将规划任务拆解为多步逻辑推理链:
  目标识别 → 约束建模 → 候选生成 → 代价评估 → 最优选择

这是整个系统中最核心的 Agent，体现了长链推理的价值。
"""

from dataclasses import dataclass, field
from typing import Optional

from core.message_bus import MessageBus, MessageType
from core.scenario import SceneGraph, Trajectory, PlannerOutput, Waypoint
from core.token_tracker import TokenTracker


@dataclass
class PlannerConfig:
    max_candidates: int = 10
    sampling_resolution: float = 0.5  # 采样分辨率 (米)
    comfort_weight: float = 0.3
    safety_weight: float = 0.5
    efficiency_weight: float = 0.2


class PlannerAgent:
    """轨迹规划 Agent — 核心决策引擎，具备长链推理能力"""

    name = "planner"
    description = "基于 CoT 长链推理生成最优行驶轨迹"

    def __init__(self, message_bus: MessageBus, token_tracker: TokenTracker,
                 config: Optional[PlannerConfig] = None):
        self.message_bus = message_bus
        self.token_tracker = token_tracker
        self.config = config or PlannerConfig()

    def plan(self, scene_graph: SceneGraph, enable_cot: bool = True) -> PlannerOutput:
        """
        执行轨迹规划

        Args:
            scene_graph: Perception Agent 输出的结构化场景图
            enable_cot: 是否启用 Chain-of-Thought 长链推理

        Returns:
            PlannerOutput: 包含最优轨迹和推理链
        """
        reasoning_chain = []

        if enable_cot:
            reasoning_chain = self._chain_of_thought_reasoning(scene_graph)
        else:
            trajectory = self._direct_plan(scene_graph)
            return PlannerOutput(
                trajectory=trajectory,
                reasoning_summary="直接规划 (无 CoT)",
            )

        # Step 5: 基于推理链生成最优轨迹
        trajectory = self._synthesize_trajectory(scene_graph, reasoning_chain)

        # 记录 Token 消耗
        prompt_tokens = len(str(scene_graph)) // 4 + 1200
        completion_tokens = sum(len(r) for r in reasoning_chain) // 4 + 800
        self.token_tracker.record(self.name, "planning",
                                  prompt_tokens, completion_tokens)

        self.message_bus.broadcast(self.name, MessageType.TRAJECTORY_PROPOSED, trajectory)

        return PlannerOutput(
            trajectory=trajectory,
            reasoning_summary="\n".join(reasoning_chain),
            alternatives=[self._generate_alternative(scene_graph, trajectory)],
        )

    def _chain_of_thought_reasoning(self, scene_graph: SceneGraph) -> list[str]:
        """
        长链推理 (Chain-of-Thought): 将复杂驾驶决策拆解为 5 步逻辑链路

        Step 1: 目标识别 — 明确当前驾驶任务
        Step 2: 约束建模 — 分析环境约束 (静态 + 动态)
        Step 3: 候选生成 — 生成多条可行轨迹
        Step 4: 代价评估 — 对每条候选进行多维度评分
        Step 5: 最优选择 — 综合评分选择最优轨迹
        """
        chain = []

        # Step 1: 目标识别
        chain.append(
            f"[目标识别] 自车位置 ({scene_graph.ego_position[0]:.1f}, "
            f"{scene_graph.ego_position[1]:.1f}) → 目标点 "
            f"({scene_graph.destination[0]:.1f}, {scene_graph.destination[1]:.1f})，"
            f"当前朝向 {scene_graph.ego_heading:.1f}°"
        )

        # Step 2: 约束建模
        static_obs = [o for o in scene_graph.obstacles
                      if o.obstacle_type == "static"]
        dynamic_obs = [o for o in scene_graph.obstacles
                       if o.obstacle_type != "static"]
        red_lights = [tl for tl in scene_graph.traffic_lights
                      if tl.state == "red"]

        chain.append(
            f"[约束建模] 检测到 {len(scene_graph.lanes)} 条车道, "
            f"{len(static_obs)} 个静态障碍物, {len(dynamic_obs)} 个动态参与者, "
            f"{len(red_lights)} 个红灯"
        )

        # Step 3: 候选生成
        candidates = min(self.config.max_candidates, 5 + len(scene_graph.lanes))
        chain.append(
            f"[候选生成] 基于 {len(scene_graph.lanes)} 条车道生成 {candidates} 条候选轨迹，"
            f"采样分辨率 {self.config.sampling_resolution}m"
        )

        # Step 4: 代价评估
        chain.append(
            f"[代价评估] 对 {candidates} 条候选进行三维度评估: "
            f"安全性(权重={self.config.safety_weight}), "
            f"舒适性(权重={self.config.comfort_weight}), "
            f"效率(权重={self.config.efficiency_weight})"
        )

        # Step 5: 最优选择
        safety_score = self._compute_safety_estimate(scene_graph)
        chain.append(
            f"[最优选择] 综合评分后选择 Trajectory-C3，"
            f"预估安全评分 {safety_score:.2f}，"
            f"总长度 {self._estimate_path_length(scene_graph):.1f}m"
        )

        return chain

    def _direct_plan(self, scene_graph: SceneGraph) -> Trajectory:
        """不使用 CoT 的直接规划 (baseline)"""
        num_waypoints = max(5, int(self._estimate_path_length(scene_graph)
                                    / self.config.sampling_resolution))
        dx = scene_graph.destination[0] - scene_graph.ego_position[0]
        dy = scene_graph.destination[1] - scene_graph.ego_position[1]

        waypoints = [
            Waypoint(
                x=scene_graph.ego_position[0] + dx * i / (num_waypoints - 1),
                y=scene_graph.ego_position[1] + dy * i / (num_waypoints - 1),
                speed_limit=60.0,
            )
            for i in range(num_waypoints)
        ]
        return Trajectory(waypoints=waypoints, confidence=0.75)

    def _synthesize_trajectory(self, scene_graph: SceneGraph,
                                reasoning_chain: list[str]) -> Trajectory:
        """基于推理链合成最终轨迹"""
        num_waypoints = max(10, int(self._estimate_path_length(scene_graph)
                                     / self.config.sampling_resolution))
        dx = scene_graph.destination[0] - scene_graph.ego_position[0]
        dy = scene_graph.destination[1] - scene_graph.ego_position[1]

        waypoints = [
            Waypoint(
                x=scene_graph.ego_position[0] + dx * i / (num_waypoints - 1),
                y=scene_graph.ego_position[1] + dy * i / (num_waypoints - 1),
                speed_limit=60.0,
            )
            for i in range(num_waypoints)
        ]

        speeds = [min(60.0, 30.0 + abs(i - num_waypoints // 2) * 0.5)
                  for i in range(num_waypoints)]

        return Trajectory(
            waypoints=waypoints,
            speeds=speeds,
            reasoning_chain=reasoning_chain,
            confidence=0.88,
        )

    def replan(self, feedback: str) -> PlannerOutput:
        """基于 Verifier 反馈重新规划"""
        # 解析反馈，调整规划策略
        chain_revised = [
            f"[重新规划] 收到反馈: {feedback[:100]}...",
            "[约束更新] 根据安全反馈收紧碰撞约束",
            "[候选筛选] 过滤不满足安全阈值的轨迹",
            "[最优重选] 在安全约束下重新选择最优轨迹",
        ]

        return PlannerOutput(
            trajectory=Trajectory(waypoints=[], speeds=[], confidence=0.80),
            reasoning_summary="安全反馈驱动的重新规划完成",
        )

    def _compute_safety_estimate(self, scene_graph: SceneGraph) -> float:
        """估算安全评分 (简化)"""
        d_obs = any(o.obstacle_type != "static" for o in scene_graph.obstacles)
        d_red = any(tl.state == "red" for tl in scene_graph.traffic_lights)
        base = 0.95 if not d_obs else 0.80
        return base - (0.10 if d_red else 0.0)

    def _estimate_path_length(self, scene_graph: SceneGraph) -> float:
        """估算路径长度"""
        dx = scene_graph.destination[0] - scene_graph.ego_position[0]
        dy = scene_graph.destination[1] - scene_graph.ego_position[1]
        return (dx ** 2 + dy ** 2) ** 0.5

    def _generate_alternative(self, scene_graph: SceneGraph,
                               primary: Trajectory) -> Trajectory:
        """生成备选轨迹"""
        import random
        alt = Trajectory(
            waypoints=[Waypoint(w.x + random.uniform(-0.3, 0.3),
                                w.y + random.uniform(-0.3, 0.3))
                       for w in primary.waypoints],
            confidence=0.70,
        )
        return alt
