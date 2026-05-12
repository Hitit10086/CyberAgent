"""
Reporter Agent: 报告生成

汇总多 Agent 协作全过程的输出，生成结构化的 Markdown/JSON 分析报告。
"""

from dataclasses import dataclass
from typing import Optional

from core.message_bus import MessageBus, MessageType
from core.scenario import (Scenario, PerceptionOutput, PlannerOutput,
                             VerificationResult)
from core.token_tracker import TokenTracker


@dataclass
class ReporterConfig:
    output_format: str = "markdown"  # markdown, json, html
    include_token_report: bool = True
    include_reasoning_chain: bool = True


class ReporterAgent:
    """报告生成 Agent — 多 Agent 协作链的终点"""

    name = "reporter"
    description = "汇总协作全过程的输出并生成可读性强的分析报告"

    def __init__(self, message_bus: MessageBus, token_tracker: TokenTracker,
                 config: Optional[ReporterConfig] = None):
        self.message_bus = message_bus
        self.token_tracker = token_tracker
        self.config = config or ReporterConfig()

    def generate(self, scenario: Scenario, perception: PerceptionOutput,
                 planner: PlannerOutput, verification: VerificationResult) -> str:
        """
        生成完整分析报告

        Args:
            scenario: 原始场景
            perception: Perception Agent 输出
            planner: Planner Agent 输出
            verification: Verifier Agent 输出

        Returns:
            str: 格式化的分析报告
        """
        if self.config.output_format == "markdown":
            report = self._generate_markdown(scenario, perception, planner, verification)
        elif self.config.output_format == "json":
            import json
            report = json.dumps(self._build_report_dict(
                scenario, perception, planner, verification
            ), ensure_ascii=False, indent=2)
        else:
            report = self._generate_markdown(scenario, perception, planner, verification)

        # 记录 Token 消耗
        self.token_tracker.record(self.name, "reporting", 500, len(report) // 4)

        self.message_bus.broadcast(self.name, MessageType.REPORT_GENERATED, report)
        return report

    def _generate_markdown(self, scenario: Scenario,
                            perception: PerceptionOutput,
                            planner: PlannerOutput,
                            verification: VerificationResult) -> str:
        """生成 Markdown 格式报告"""
        lines = [
            f"# 场景分析报告: {scenario.name}",
            f"",
            f"**场景 ID**: `{scenario.scenario_id}`",
            f"**分析时间**: {__import__('datetime').datetime.now().isoformat()}",
            f"**整体安全评分**: `{verification.safety_score:.2%}`",
            f"**验证状态**: {'[PASS]' if verification.passed else '[FAIL]'}",
            f"",
            f"---",
            f"",
            f"## 1. 场景解析 (Perception Agent)",
            f"",
            f"- 检测到障碍物: {perception.detected_objects} 个",
            f"- 解析置信度: {perception.confidence:.1%}",
            f"- 处理耗时: {perception.processing_time_ms:.1f}ms",
            f"",
            f"### 场景元素",
            f"",
            f"| 类型 | 数量 |",
            f"|------|------|",
            f"| 车道 | {len(perception.scene_graph.lanes)} |",
            f"| 障碍物 | {len(perception.scene_graph.obstacles)} |",
            f"| 交通信号灯 | {len(perception.scene_graph.traffic_lights)} |",
            f"",
        ]

        # 推理链
        if self.config.include_reasoning_chain and planner.trajectory.reasoning_chain:
            lines += [
                f"## 2. 轨迹规划 (Planner Agent) — 长链推理过程",
                f"",
            ]
            for i, step in enumerate(planner.trajectory.reasoning_chain, 1):
                lines.append(f"**Step {i}**: {step}")
                lines.append("")
            lines += [
                f"**规划置信度**: {planner.trajectory.confidence:.1%}",
                f"",
            ]

        # 安全验证
        lines += [
            f"## 3. 安全验证 (Verifier Agent)",
            f"",
            f"| 检查项 | 评分 |",
            f"|--------|------|",
            f"| 碰撞风险 | {1.0 - verification.collision_risk:.1%} |",
            f"| 交通规则合规性 | {verification.traffic_rule_compliance:.1%} |",
            f"| 综合安全评分 | {verification.safety_score:.1%} |",
            f"",
            f"**反馈**: {verification.feedback}",
            f"",
        ]

        # Token 消耗
        if self.config.include_token_report:
            usage = self.token_tracker.summary()
            lines += [
                f"## 4. Token 消耗统计",
                f"",
                f"| 指标 | 数值 |",
                f"|------|------|",
                f"| Prompt Tokens | {usage.prompt_tokens:,} |",
                f"| Completion Tokens | {usage.completion_tokens:,} |",
                f"| Total Tokens | {usage.total_tokens:,} |",
                f"",
            ]

        return "\n".join(lines)

    def _build_report_dict(self, scenario, perception, planner, verification) -> dict:
        """构建 JSON 格式报告"""
        usage = self.token_tracker.summary()
        return {
            "scenario_id": scenario.scenario_id,
            "scenario_name": scenario.name,
            "safety_score": verification.safety_score,
            "verification_passed": verification.passed,
            "detected_objects": perception.detected_objects,
            "trajectory_confidence": planner.trajectory.confidence,
            "reasoning_chain": planner.trajectory.reasoning_chain,
            "token_usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "agent_breakdown": usage.agent_breakdown,
            },
        }
