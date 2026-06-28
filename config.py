"""集中配置:SiliconFlow(国产 Qwen 模型)/ Firecrawl / 路径 / 阈值。

所有项从环境变量读取并带默认值,项目开箱即用,也便于切换模型或服务。
对应大作业"国产模型选型与可配置性"要求。
"""
import os
import re
from dotenv import load_dotenv

load_dotenv()

# ---- SiliconFlow(国产 Qwen,作为 Agent 大脑)----
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
SILICONFLOW_LLM_MODEL = os.getenv("SILICONFLOW_LLM_MODEL", "Qwen/Qwen2.5-72B-Instruct")
# 全栈统一 72B-Instruct:小模型(7B)在 SiliconFlow 上对结构化 JSON 输出不稳
# (RAGAS / 营收抽取都依赖强 JSON),72B 输出合规率高,质量稳定。
SILICONFLOW_AGENT_MODEL = os.getenv("SILICONFLOW_AGENT_MODEL", "Qwen/Qwen2.5-72B-Instruct")
SILICONFLOW_EMBED_MODEL = os.getenv("SILICONFLOW_EMBED_MODEL", "BAAI/bge-m3")  # 多语言,rag.py 验证可在 SiliconFlow 调用
# RAG 知识库用多语言模型(中英文均可检索);与正/反分析节点的英文模型分离
SILICONFLOW_RAG_EMBED_MODEL = os.getenv("SILICONFLOW_RAG_EMBED_MODEL", "BAAI/bge-m3")

# ---- Firecrawl(联网检索)----
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "")

# ---- 路径 ----
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", os.path.join(PROJECT_ROOT, "output"))
KNOWLEDGE_DIR = os.path.join(PROJECT_ROOT, "knowledge")
KB_INDEX_DIR = os.path.join(PROJECT_ROOT, "kb_index")

# ---- 质量阈值 ----
QUALITY_THRESHOLD = 0.70

# ---- 记忆 ----
MEMORY_LAST_N = int(os.getenv("MEMORY_LAST_N", "10"))


def ensure_dirs():
    """确保运行所需目录存在(输出/知识库/向量索引)。"""
    for d in (OUTPUT_DIR, KNOWLEDGE_DIR, KB_INDEX_DIR):
        os.makedirs(d, exist_ok=True)


def safe_filename(name: str) -> str:
    """把公司名等用户输入转成安全的文件名片段(避免路径注入)。"""
    cleaned = re.sub(r"[^\w\-]+", "_", (name or "unknown").strip())
    return (cleaned[:50] or "unknown")
