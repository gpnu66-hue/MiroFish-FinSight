"""持久化 FAISS 知识库(满足大作业 RAG 必做:Embedding + 向量数据库,≥5 篇文档)。

首次运行从 knowledge/*.md 构建索引并落盘到 kb_index/;之后直接 load 复用。
检索函数 retrieve(query) 返回 top-k 相关文本片段,供 ReAct agent 的 rag_search 工具使用。
"""
import os
import re
import glob

from langchain_community.vectorstores import FAISS, DistanceStrategy
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

import config

_RETRIEVER_K = 4


def _embeddings():
    return OpenAIEmbeddings(
        base_url=config.SILICONFLOW_BASE_URL,
        api_key=config.SILICONFLOW_API_KEY,
        model=config.SILICONFLOW_RAG_EMBED_MODEL,
    )


def _load_docs():
    """读取 knowledge/ 下全部 .md 文档。"""
    paths = sorted(glob.glob(os.path.join(config.KNOWLEDGE_DIR, "*.md")))
    docs = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if text:
            docs.append(Document(page_content=text, metadata={"source": os.path.basename(p)}))
    return docs


def _chunk(docs, chunk_size=320, overlap=48):
    """按"行"贪心合并到 ~chunk_size:让标题/小节与其下要点附着,又不会把
    一整篇合并成一大块稀释语义,从而保证具体事实(如某指标数值)可被精准检索。"""
    chunks = []
    for d in docs:
        lines = [ln.strip() for ln in d.page_content.split("\n") if ln.strip()]
        buf = ""
        for ln in lines:
            if len(ln) > chunk_size:
                if buf:
                    chunks.append(Document(page_content=buf, metadata=d.metadata))
                    buf = ""
                start = 0
                while start < len(ln):
                    chunks.append(Document(page_content=ln[start:start + chunk_size], metadata=d.metadata))
                    start += chunk_size - overlap
            elif len(buf) + len(ln) + 1 <= chunk_size:
                buf = (buf + "\n" + ln) if buf else ln
            else:
                chunks.append(Document(page_content=buf, metadata=d.metadata))
                buf = ln
        if buf:
            chunks.append(Document(page_content=buf, metadata=d.metadata))
    return chunks


def build_kb():
    """从 knowledge/ 构建 FAISS 索引并持久化到 kb_index/。"""
    config.ensure_dirs()
    docs = _load_docs()
    if not docs:
        raise RuntimeError(f"知识库为空:请在 {config.KNOWLEDGE_DIR} 放置 ≥5 篇文档")
    chunks = _chunk(docs)
    vs = FAISS.from_documents(chunks, _embeddings(), distance_strategy=DistanceStrategy.COSINE)
    vs.save_local(config.KB_INDEX_DIR)
    print(f"[RAG] 知识库构建完成:{len(docs)} 篇文档 → {len(chunks)} 个块,索引保存到 {config.KB_INDEX_DIR}")
    return vs


def load_kb():
    """加载已持久化的索引;不存在则现场构建。"""
    index_faiss = os.path.join(config.KB_INDEX_DIR, "index.faiss")
    if os.path.exists(index_faiss):
        return FAISS.load_local(
            config.KB_INDEX_DIR, _embeddings(),
            allow_dangerous_deserialization=True,
            distance_strategy=DistanceStrategy.COSINE,
        )
    return build_kb()


def _char_bigrams(s: str):
    """字符二元组(无需分词器,中英文通用),用于关键词重叠计算。"""
    s = re.sub(r"\s+", "", (s or "").lower())
    return set(s[i:i + 2] for i in range(max(0, len(s) - 1)))


def retrieve(query: str, k: int = _RETRIEVER_K):
    """混合检索(向量召回 + 关键词重排序)。

    先用 FAISS 余弦召回较大候选池,再用"语义分 + 字符重叠分"重排,返回 top-k。
    纯语义在小模型/小库上区分度有限,加入关键词重叠能精准命中含专有名词/指标的片段,
    同时构成 RAG 重排序(对应加分项)。
    """
    vs = load_kb()
    pool = vs.similarity_search_with_score(query, k=max(k * 3, 12))  # (doc, distance) 越小越相关
    qb = _char_bigrams(query)
    scored = []
    for doc, dist in pool:
        sem = -float(dist)  # distance 越小 → 分数越高
        tb = _char_bigrams(doc.page_content)
        kw = (len(qb & tb) / len(qb)) if qb else 0.0
        scored.append((sem + 1.5 * kw, doc))
    scored.sort(key=lambda x: -x[0])
    return [doc for _, doc in scored[:k]]


if __name__ == "__main__":
    # 自测:构建/加载并检索一次
    load_kb()
    print("\n=== 检索测试:英伟达营收增长 ===")
    for d in retrieve("英伟达营收增长"):
        print(f"--- {d.metadata['source']} ---")
        print(d.page_content[:200], "\n")
