"""
Orchestrator: 多 Agent 协作调度器

负责协调 Perception → Planner → Verifier → Reporter 四个 Agent 的协作流程。
支持同步和异步两种执行模式。
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

from core.message_bus import MessageBus
from core.scenario import Scenario, AnalysisResult
from core.token_tracker import TokenTracker
from agents.perception import PerceptionAgent
from agents.planner import PlannerAgent
from agents.verifier import VerifierAgent
from agents.reporter import ReporterAgent


@dataclass
class OrchestratorConfig:
    max_retries: int = 3
    safety_threshold: float = 0.7
    enable_chain_of_thought: bool = True
    parallel_verification: bool = True


class Orchestrator:
    """多 Agent 协作调度器"""

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.config = config or OrchestratorConfig()
        self.message_bus = MessageBus()
        self.token_tracker = TokenTracker()

        # 初始化 Agent 并注入消息总线
        self.perception_agent = PerceptionAgent(self.message_bus, self.token_tracker)
        self.planner_agent = PlannerAgent(self.message_bus, self.token_tracker)
        self.verifier_agent = VerifierAgent(self.message_bus, self.token_tracker)
        self.reporter_agent = ReporterAgent(self.message_bus, self.token_tracker)

        self._register_agents()

    def _register_agents(self):
        """向消息总线注册所有 Agent"""
        for agent in [self.perception_agent, self.planner_agent,
                      self.verifier_agent, self.reporter_agent]:
            self.message_bus.register(agent.name, agent)

    def analyze(self, scenario: Scenario) -> AnalysisResult:
        """
        执行完整的多 Agent 协作分析流程

        Args:
            scenario: 仿真场景数据

        Returns:
            AnalysisResult: 包含安全评分、最优轨迹、完整报告和 Token 消耗
        """
        start_time = time.time()

        # Phase 1: 场景解析
        perception_output = self.perception_agent.process(scenario)

        # Phase 2: 轨迹规划 (长链推理)
        planner_output = self.planner_agent.plan(
            scene_graph=perception_output.scene_graph,
            enable_cot=self.config.enable_chain_of_thought,
        )

        # Phase 3: 安全验证 (支持多次迭代)
        verification_passed = False
        for attempt in range(self.config.max_retries):
            verification_result = self.verifier_agent.verify(
                trajectory=planner_output.trajectory,
                scene_graph=perception_output.scene_graph,
            )

            if verification_result.safety_score >= self.config.safety_threshold:
                verification_passed = True
                break

            if attempt < self.config.max_retries - 1:
                planner_output = self.planner_agent.replan(
                    feedback=verification_result.feedback,
                )

        # Phase 4: 报告生成
        report = self.reporter_agent.generate(
            scenario=scenario,
            perception=perception_output,
            planner=planner_output,
            verification=verification_result,
        )

        elapsed = time.time() - start_time

        return AnalysisResult(
            safety_score=verification_result.safety_score,
            optimal_trajectory=planner_output.trajectory,
            report=report,
            token_usage=self.token_tracker.summary(),
            elapsed_seconds=elapsed,
            verification_passed=verification_passed,
        )

    def analyze_batch(self, scenarios: list[Scenario]) -> list[AnalysisResult]:
        """批量分析多个场景"""
        return [self.analyze(s) for s in scenarios]

    async def analyze_async(self, scenario: Scenario) -> AnalysisResult:
        """异步分析单个场景 (Agent 间可并行时使用)"""
        return await asyncio.to_thread(self.analyze, scenario)
