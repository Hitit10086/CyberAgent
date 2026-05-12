"""
Token Tracker: 全链路 Token 消耗追踪

记录每个 Agent 的 Token 消耗，支持按 Agent/阶段/场景维度聚合统计。
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field

from .scenario import TokenUsage


@dataclass
class TokenRecord:
    agent_name: str
    phase: str
    prompt_tokens: int
    completion_tokens: int
    timestamp: float = field(default_factory=time.time)


class TokenTracker:
    """Token 消耗追踪器"""

    def __init__(self):
        self._records: list[TokenRecord] = []

    def record(self, agent_name: str, phase: str,
               prompt_tokens: int, completion_tokens: int):
        """记录一次 Token 消耗"""
        self._records.append(TokenRecord(
            agent_name=agent_name,
            phase=phase,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ))

    def summary(self) -> TokenUsage:
        """生成 Token 消耗汇总"""
        total_prompt = sum(r.prompt_tokens for r in self._records)
        total_completion = sum(r.completion_tokens for r in self._records)

        breakdown: dict[str, int] = defaultdict(int)
        for r in self._records:
            breakdown[r.agent_name] += r.prompt_tokens + r.completion_tokens

        return TokenUsage(
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            total_tokens=total_prompt + total_completion,
            agent_breakdown=dict(breakdown),
        )

    def phase_breakdown(self) -> dict[str, int]:
        """按阶段统计 Token 消耗"""
        breakdown: dict[str, int] = defaultdict(int)
        for r in self._records:
            breakdown[r.phase] += r.prompt_tokens + r.completion_tokens
        return dict(breakdown)
