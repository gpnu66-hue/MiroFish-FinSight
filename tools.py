"""标准化工具(满足大作业"≥2 个自定义工具"必做)。

四个 LangChain Tool,由 ReAct 编排 agent 自主决定调用:
  - rag_search          检索本地金融知识库
  - web_search          联网搜索(Firecrawl;失败时返回友好提示)
  - financial_analysis  调用多智能体深度财报分析流水线(bull/bear/debate/quality)
  - draw_chart          依据结构化数据绘制营收柱状图

每个工具有明确的 name / description / 调用逻辑,框架使用 langchain_core.tools.tool。
"""
import json
import os

from langchain_core.tools import tool

import config
import rag


@tool
def rag_search(query: str) -> str:
    """Search the local financial knowledge base for relevant information.
    Use this to answer questions about financial metrics (revenue, gross margin, EPS,
    free cash flow), specific company financials, or investment-analysis concepts.
    Always try this first before searching the web.
    """
    try:
        docs = rag.retrieve(query)
    except Exception as e:
        return f"[rag_search] 知识库检索失败: {e}"
    if not docs:
        return "[rag_search] 未检索到相关内容。"
    blocks = []
    for d in docs:
        blocks.append(f"[来源: {d.metadata.get('source', '?')}]\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)


@tool
def web_search(query: str) -> str:
    """Search the web for the latest information (earnings, news, guidance).
    Use when the knowledge base lacks current or company-specific data.
    Requires network access and a configured FIRECRAWL_API_KEY.
    """
    key = config.FIRECRAWL_API_KEY
    if not key:
        return "[web_search] 未配置 FIRECRAWL_API_KEY(离线):请改用知识库 rag_search,或在 .env 配置 FIRECRAWL_API_KEY。"
    try:
        from firecrawl import FirecrawlApp
        app = FirecrawlApp(api_key=key)
        result = app.search(query, limit=3)
        rows = getattr(result, "web", None) or (result if isinstance(result, list) else [])
        items = []
        for r in rows[:3]:
            if isinstance(r, dict):
                title = r.get("title", "")
                url = r.get("url", "")
                desc = r.get("description") or r.get("content") or ""
            else:
                title = getattr(r, "title", "")
                url = getattr(r, "url", "")
                desc = getattr(r, "description", "")
            items.append(f"- {title}\n  {url}\n  {desc}")
        return "\n".join(items) if items else "[web_search] 无结果。"
    except Exception as e:
        return f"[web_search] 联网搜索失败({e});请改用知识库 rag_search。"


@tool
def financial_analysis(company: str) -> str:
    """Run the full multi-agent deep financial-analysis pipeline
    (DataFetcher → Bull Analyst → Bear Analyst → Debater → Quality Inspector)
    for a company and return a synthesized balanced report.
    This is slower (multiple LLM calls + web fetching); use for in-depth analysis, not quick facts.
    """
    try:
        from main import run_financial_analysis  # 延迟导入,避免与 main 的循环依赖
        from langchain_core.messages import AIMessage
        result = run_financial_analysis(company)
        if not result:
            return "[financial_analysis] 分析失败,请查看上方日志。"
        report = ""
        _skip = ("Quality Check", "Reflection:", "Successfully fetched", "[BULL",
                 "[BEAR", "[VISUALIZATION", "Visualization", "Bull analysis", "Bear analysis")
        for m in reversed(result.get("messages", [])):
            if isinstance(m, AIMessage) and m.content and not m.content.startswith(_skip):
                report = m.content
                break
        score = result.get("eval_score", 0.0)
        return (f"[financial_analysis] 完成度评分={score:.2f}\n\n{report}"
                if report else "[financial_analysis] 未取到报告文本,详见运行日志。")
    except Exception as e:
        return f"[financial_analysis] 运行出错: {e}"


@tool
def draw_chart(company: str, revenue_data_json: str) -> str:
    """Draw a revenue bar chart and save it to the output directory.
    revenue_data_json must be a JSON array of objects, e.g.:
    [{"period":"FY2023","revenue_billion":383,"yoy_percent":-3},
     {"period":"FY2024","revenue_billion":391,"yoy_percent":2}]
    """
    try:
        data = json.loads(revenue_data_json)
    except Exception as e:
        return f"[draw_chart] revenue_data_json 解析失败(需 JSON 数组): {e}"
    if not isinstance(data, list) or not data:
        return "[draw_chart] 数据为空。"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    periods = [d.get("period", str(i + 1)) for i, d in enumerate(data)]
    revenues = [float(d.get("revenue_billion", 0)) for d in data]
    yoys = [d.get("yoy_percent") for d in data]

    config.ensure_dirs()
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#34C759" if (y is not None and y > 0) else "#FF453A" for y in yoys]
    ax.bar(periods, revenues, color=colors, alpha=0.75, edgecolor="black")
    ax.set_title(f"{company} Revenue (Billion USD)")
    ax.set_ylabel("Revenue (Billion USD)")
    for i, r in enumerate(revenues):
        ax.text(i, r, f"${r:.0f}B", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(config.OUTPUT_DIR, f"{config.safe_filename(company)}_revenue.png")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    return f"[draw_chart] 图表已保存: {path}"


# 全部工具及按名映射(供 ReAct agent 与安全白名单使用)
ALL_TOOLS = [rag_search, web_search, financial_analysis, draw_chart]
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}
