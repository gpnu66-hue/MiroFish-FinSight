"""ragas 0.4.3 <-> langchain-community 0.4.2 兼容垫片。

ragas 0.4.3 在模块顶层执行:
    from langchain_community.chat_models.vertexai import ChatVertexAI
但 langchain-community 0.4.2 已移除该模块(sunset 迁移),导致 `from ragas import ...`
直接 ImportError,整个 agents.py 无法加载。

本项目不使用 VertexAI(全程用国产 Qwen + SiliconFlow),而 ragas 仅在 isinstance() 检查里
引用 ChatVertexAI、从不实例化它,因此注入一个空 stub 类即可满足导入。

必须在 `from ragas import ...` 之前 import 本模块(agents.py 第一方库导入处)。
"""
import sys
import types

_MOD_NAME = "langchain_community.chat_models.vertexai"

if _MOD_NAME not in sys.modules:
    try:
        __import__(_MOD_NAME)
    except Exception:
        # 真实模块不存在,注入最小 stub
        stub = types.ModuleType(_MOD_NAME)

        class ChatVertexAI:  # ragas 仅做 isinstance(llm, ChatVertexAI) 检查
            pass

        class VertexAI:
            pass

        stub.ChatVertexAI = ChatVertexAI
        stub.VertexAI = VertexAI
        sys.modules[_MOD_NAME] = stub
