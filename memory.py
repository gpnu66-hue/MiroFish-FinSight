"""记忆机制(满足大作业 Memory 必做 + 长期记忆加分)。

- 短期记忆 trim_messages():保留最近 N 轮对话,防止上下文无限增长。
- 长期记忆 LongTermMemory:本地 JSON 持久化用户偏好/事实,跨会话保留(加分项)。
"""
import json
import os

from langchain_core.messages import SystemMessage

import config


def trim_messages(messages, last_n=None):
    """保留最近 last_n 条消息(默认 config.MEMORY_LAST_N)。

    始终保留首条 SystemMessage(系统人设)以及初始主题/公司上下文,
    再追加最近 last_n 条,从而既"记得最近对话"又"不忘角色与背景"。
    """
    if last_n is None:
        last_n = config.MEMORY_LAST_N
    messages = list(messages)
    if len(messages) <= last_n:
        return messages

    head = []
    rest = messages
    # 保留首条系统消息
    if rest and isinstance(rest[0], SystemMessage):
        head.append(rest[0])
        rest = rest[1:]
    # 保留最初的用户输入(主题/公司上下文),避免多轮后丢失背景
    body = rest
    if body:
        head.append(body[0])
        body = body[1:]
    return head + body[-(last_n - 1):] if last_n > 1 else head


class LongTermMemory:
    """极简长期记忆:JSON 键值存储,跨会话持久化(加分项)。"""

    def __init__(self, path=None):
        self.path = path or os.path.join(config.PROJECT_ROOT, "long_term_memory.json")
        self._store = self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._store, f, ensure_ascii=False, indent=2)

    def remember(self, key, value):
        self._store[key] = value
        self._save()

    def recall(self, key, default=None):
        return self._store.get(key, default)

    def all_facts(self):
        return dict(self._store)
