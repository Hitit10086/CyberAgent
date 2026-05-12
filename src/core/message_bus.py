"""
Message Bus: Agent 间消息总线

提供 Agent 间的异步消息传递、事件广播和状态同步能力。
支持 Pub/Sub 模式，确保 Agent 间松耦合。
"""

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class MessageType(Enum):
    SCENE_GRAPH_READY = "scene_graph_ready"
    TRAJECTORY_PROPOSED = "trajectory_proposed"
    VERIFICATION_RESULT = "verification_result"
    REPLAN_REQUEST = "replan_request"
    REPORT_GENERATED = "report_generated"
    ERROR = "error"


@dataclass
class Message:
    msg_type: MessageType
    sender: str
    payload: Any
    timestamp: float = field(default_factory=__import__("time").time)


class MessageBus:
    """Agent 间消息传递中枢"""

    def __init__(self):
        self._subscribers: dict[MessageType, list[Callable]] = defaultdict(list)
        self._agents: dict[str, Any] = {}
        self._message_log: list[Message] = []
        self._lock = threading.Lock()

    def register(self, agent_name: str, agent_instance: Any):
        """注册 Agent 到消息总线"""
        with self._lock:
            self._agents[agent_name] = agent_instance

    def subscribe(self, agent_name: str, msg_type: MessageType, callback: Callable):
        """订阅特定类型的消息"""
        with self._lock:
            self._subscribers[msg_type].append(callback)

    def publish(self, message: Message):
        """发布消息给所有订阅者"""
        self._message_log.append(message)
        with self._lock:
            for callback in self._subscribers.get(message.msg_type, []):
                try:
                    callback(message)
                except Exception as e:
                    self.publish(Message(
                        msg_type=MessageType.ERROR,
                        sender="MessageBus",
                        payload={"original": message, "error": str(e)},
                    ))

    def broadcast(self, sender: str, msg_type: MessageType, payload: Any):
        """广播消息到所有订阅者"""
        self.publish(Message(msg_type=msg_type, sender=sender, payload=payload))

    def get_agent(self, name: str):
        """按名称获取已注册的 Agent"""
        return self._agents.get(name)

    def get_message_history(self) -> list[Message]:
        """获取消息历史 (用于调试和可观测性)"""
        return list(self._message_log)
