# CyberAgent — Multi-Agent Autonomous Driving Simulation & Analysis Platform

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/AI_Framework-Anthropic_Claude_SDK-orange)](https://docs.anthropic.com/en/docs/claude-code)

**CyberAgent** 是一个基于多 Agent 协作的自动驾驶仿真场景智能分析平台。它将复杂的自动驾驶仿真任务拆解为多个专业化 AI Agent 协同完成：场景解析 Agent 识别关键交通元素，轨迹规划 Agent 执行长链推理生成最优路径，安全验证 Agent 对规划结果进行多维度安全校验，报告生成 Agent 自动汇总分析结果并输出可视化报告。整个系统实现了 **"场景输入 → 多 Agent 协作推理 → 自动化验证 → 报告输出"** 的完整闭环。

---

## 核心亮点

- **多 Agent 协作架构**：4 个专业化 Agent 通过消息总线协作，每个 Agent 聚焦独立子任务
- **长链推理引擎**：轨迹规划 Agent 内置 CoT (Chain-of-Thought) 推理，将复杂驾驶决策拆解为多步逻辑推理
- **自动化闭环验证**：安全验证 Agent 自动运行仿真测试，对规划结果进行碰撞检测、交通规则合规性校验
- **可观测性**：全链路 Token 消耗追踪与 Agent 决策过程可视化

---

## 架构概览

```
┌──────────────────────────────────────────────────────────────┐
│                     Orchestrator (调度器)                      │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐ │
│  │ 场景解析  │  │ 轨迹规划  │  │ 安全验证  │  │  报告生成     │ │
│  │ Agent    │→│ Agent     │→│ Agent     │→│  Agent       │ │
│  │ (Percep) │  │ (Plan)   │  │ (Verify)  │  │  (Report)   │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘ │
│       │             │              │               │         │
│       └─────────────┴──────────────┴───────────────┘         │
│                         │                                    │
│               ┌─────────▼─────────┐                          │
│               │   Message Bus    │                          │
│               │   (消息总线)      │                          │
│               └──────────────────┘                          │
└──────────────────────────────────────────────────────────────┘
```

## 安装

```bash
git clone https://github.com/linhanyuan/CyberAgent.git
cd CyberAgent
pip install -r requirements.txt
```

## 快速开始

```python
from src.core.orchestrator import Orchestrator
from src.core.scenario import Scenario

# 加载仿真场景
scenario = Scenario.from_file("examples/scenarios/intersection.json")

# 启动多 Agent 协作分析
orchestrator = Orchestrator()
result = orchestrator.analyze(scenario)

print(f"安全评分: {result.safety_score}")
print(f"最优轨迹: {result.optimal_trajectory}")
print(f"Token 消耗: {result.token_usage}")
```

## 项目结构

```
CyberAgent/
├── src/
│   ├── agents/           # 专业化 Agent 实现
│   │   ├── perception.py # 场景解析 Agent
│   │   ├── planner.py    # 轨迹规划 Agent (含 CoT 推理)
│   │   ├── verifier.py   # 安全验证 Agent
│   │   └── reporter.py   # 报告生成 Agent
│   ├── core/             # 核心框架
│   │   ├── orchestrator.py  # 多 Agent 调度器
│   │   ├── message_bus.py   # 消息总线
│   │   ├── scenario.py      # 场景数据结构
│   │   └── token_tracker.py # Token 消耗追踪
│   └── utils/            # 工具函数
├── examples/             # 示例场景
├── tests/                # 自动化测试
├── config/               # 配置文件
└── docs/                 # 文档
```

## 核心逻辑流

1. **场景输入** → 加载 Apollo/SimOne 仿真场景 JSON
2. **Perception Agent** → 解析道路拓扑、交通参与者、信号灯状态，输出结构化场景图
3. **Planner Agent (长链推理)** → 基于 CoT 将规划任务拆解为：目标识别 → 约束建模 → 候选生成 → 代价评估 → 最优选择
4. **Verifier Agent** → 对规划轨迹进行碰撞检测、速度限制校验、交通规则合规性检查
5. **Reporter Agent** → 汇总各 Agent 输出，生成 Markdown/JSON 分析报告
6. **闭环反馈** → 不安全场景自动标记并触发重新规划

## 应用场景

- 中国机器人及人工智能大赛 (CRAIC) 自动驾驶仿真赛项
- Apollo 自动驾驶仿真场景自动化分析
- 智能驾驶算法迭代中的回归测试自动化

## License

MIT License
