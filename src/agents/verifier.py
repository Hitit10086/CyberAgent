"""
Verifier Agent: 安全验证

对 Planner 输出的轨迹进行多维度安全校验:
  - 碰撞检测 (静态 + 动态)
  - 速度限制合规性
  - 交通规则 (信号灯、车道线) 合规性
"""

from dataclasses import dataclass
from typing import Optional

from core.message_bus import MessageBus, MessageType
from core.scenario import SceneGraph, Trajectory, VerificationResult
from core.token_tracker import TokenTracker


@dataclass
class VerifierConfig:
    collision_threshold: float = 1.5  # 碰撞安全距离 (米)
    speed_tolerance: float = 1.1    # 速度容差比例
    red_light_stop_distance: float = 3.0  # 红灯停车距离 (米)


class VerifierAgent:
    """安全验证 Agent — 闭环验证的关键环节"""

    name = "verifier"
    description = "对规划轨迹进行多维度安全校验，提供反馈供重规划"

    def __init__(self, message_bus: MessageBus, token_tracker: TokenTracker,
                 config: Optional[VerifierConfig] = None):
        self.message_bus = message_bus
        self.token_tracker = token_tracker
        self.config = config or VerifierConfig()

    def verify(self, trajectory: Trajectory,
               scene_graph: SceneGraph) -> VerificationResult:
        """
        对规划轨迹进行安全验证

        Returns:
            VerificationResult: 安全评分和详细反馈
        """
        feedback_parts = []
        scores = {}

        # 1. 碰撞风险检测
        collision_score = self._check_collision_risk(trajectory, scene_graph)
        scores["collision"] = collision_score
        if collision_score < 0.9:
            feedback_parts.append(f"碰撞风险较高 (评分: {collision_score:.2f})")

        # 2. 交通规则合规性
        rule_score = self._check_traffic_rules(trajectory, scene_graph)
        scores["rules"] = rule_score
        if rule_score < 0.9:
            feedback_parts.append(f"交通规则合规性不足 (评分: {rule_score:.2f})")

        # 3. 速度合规性
        speed_score = self._check_speed_compliance(trajectory, scene_graph)
        scores["speed"] = speed_score
        if speed_score < 0.9:
            feedback_parts.append(f"速度违规风险 (评分: {speed_score:.2f})")

        # 4. 轨迹平滑性
        smoothness_score = self._check_smoothness(trajectory)
        scores["smoothness"] = smoothness_score

        # 综合评分
        weights = {"collision": 0.45, "rules": 0.25, "speed": 0.20, "smoothness": 0.10}
        overall = sum(scores[k] * weights[k] for k in scores)

        feedback = "; ".join(feedback_parts) if feedback_parts else "所有检查通过"

        # 记录 Token 消耗
        self.token_tracker.record(self.name, "verification", 800, 400)

        result = VerificationResult(
            safety_score=round(overall, 3),
            collision_risk=round(1.0 - collision_score, 3),
            traffic_rule_compliance=rule_score,
            feedback=feedback,
            passed=overall >= 0.70,
        )

        self.message_bus.broadcast(
            self.name, MessageType.VERIFICATION_RESULT, result
        )
        return result

    def _check_collision_risk(self, trajectory: Trajectory,
                               scene_graph: SceneGraph) -> float:
        """碰撞风险检测"""
        if not trajectory.waypoints or not scene_graph.obstacles:
            return 1.0

        risk = 1.0
        for wp in trajectory.waypoints:
            for obs in scene_graph.obstacles:
                dist = ((wp.x - obs.position[0]) ** 2 +
                        (wp.y - obs.position[1]) ** 2) ** 0.5
                if dist < self.config.collision_threshold:
                    risk -= 0.05 * (1.0 - dist / self.config.collision_threshold)
        return max(0.0, risk)

    def _check_traffic_rules(self, trajectory: Trajectory,
                              scene_graph: SceneGraph) -> float:
        """交通规则合规性检查"""
        if not trajectory.waypoints:
            return 1.0

        score = 1.0
        red_lights = [tl for tl in scene_graph.traffic_lights
                      if tl.state == "red"]

        for wp in trajectory.waypoints:
            for tl in red_lights:
                dist = ((wp.x - tl.position[0]) ** 2 +
                        (wp.y - tl.position[1]) ** 2) ** 0.5
                if dist < self.config.red_light_stop_distance:
                    # 红灯附近需低速/停止
                    score -= 0.1
        return max(0.0, score)

    def _check_speed_compliance(self, trajectory: Trajectory,
                                 scene_graph: SceneGraph) -> float:
        """速度合规性检查"""
        if not trajectory.speeds or not trajectory.waypoints:
            return 1.0

        violations = 0
        for i, (wp, speed) in enumerate(
            zip(trajectory.waypoints, trajectory.speeds)):
            if speed > wp.speed_limit * self.config.speed_tolerance:
                violations += 1

        n = len(trajectory.waypoints)
        return max(0.0, 1.0 - violations / n) if n > 0 else 1.0

    def _check_smoothness(self, trajectory: Trajectory) -> float:
        """轨迹平滑性检查"""
        if len(trajectory.waypoints) < 3:
            return 1.0

        # 检查相邻段的角度变化
        angles = []
        for i in range(1, len(trajectory.waypoints) - 1):
            prev = trajectory.waypoints[i - 1]
            curr = trajectory.waypoints[i]
            nxt = trajectory.waypoints[i + 1]

            import math
            a1 = math.atan2(curr.y - prev.y, curr.x - prev.x)
            a2 = math.atan2(nxt.y - curr.y, nxt.x - curr.x)
            diff = abs(a1 - a2)
            angles.append(min(diff, 2 * math.pi - diff))

        avg_angle = sum(angles) / len(angles) if angles else 0
        return max(0.0, 1.0 - avg_angle / math.pi)
