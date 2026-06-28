import os
import re
import json
from typing import TypedDict, List
from langgraph.graph import MessagesState

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from firecrawl import FirecrawlApp
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

import _ragas_compat  # noqa: F401  兼容垫片:必须在 import ragas 之前加载
import config

from ragas import EvaluationDataset, evaluate
# ragas 0.4.x 的 collections 指标要求 InstructorLLM;改用旧 ragas.metrics 路径
# (旧 metric 类可直接接收 ChatOpenAI,避免降级到启发式)
from ragas.metrics import Faithfulness, AnswerRelevancy

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from dotenv import load_dotenv

load_dotenv()


def clean_text(text: str) -> str:
    if not text:
        return ""

    cleaned = text

    cleaned = re.sub(r'https?://\S+', '', cleaned)

    cleaned = re.sub(r'Skip to main content', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'Cookie preferences', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'Accept cookies', '', cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

    cleaned = cleaned.strip()

    return cleaned


class AgentState(MessagesState):
    research_context: str
    eval_score: float
    failure_reason: str
    iteration_count: int
    suggested_queries: List[str]
    bull_analysis: str
    bear_analysis: str


# -----------------------------
# Data Fetcher
# -----------------------------

def data_fetcher(state: AgentState):

    print("[DataFetcher] Running")

    try:
        app = FirecrawlApp(api_key=config.FIRECRAWL_API_KEY)

        messages = state["messages"]
        company_name = (
            messages[0].content
            if messages and isinstance(messages[0], HumanMessage)
            else "Unknown"
        )

        iteration = state.get("iteration_count", 0)

        print(f"\n[Iteration {iteration}] Fetching data for: {company_name}")
        print(f"[DataFetcher] Locked company name from initial input: {company_name}")

        suggested_queries = state.get("suggested_queries", [])

        if suggested_queries:
            print(f"[DataFetcher] Using {len(suggested_queries)} suggested queries from reflection")
            search_queries = suggested_queries
        else:
            print(f"[DataFetcher] Using default financial search queries (first iteration)")
            search_queries = [
                f"{company_name} earnings report",
                f"{company_name} revenue growth analysis",
                f"{company_name} financial results quarterly",
                f"{company_name} competition and risks analysis 2026"
            ]

        # Optimize for forward-looking data (2026 projections)
        if iteration > 0 or True:  # Apply 2026 forward-looking filter
            print(f"[DataFetcher] Applying 2026 forward-looking filter for projected financial data...")
            forward_looking_queries = [
                f"{company_name} 2026 revenue forecast projection guidance",
                f"{company_name} 2026 earnings guidance outlook",
                f"{company_name} AI innovation strategy 2026",
            ]
            search_queries.extend(forward_looking_queries)

        aggregated_content = []
        scrape_urls = []  # Track URLs to scrape for full content

        for q in search_queries:
            search_result = app.search(q, limit=3)

            if hasattr(search_result, 'web') and search_result.web:
                for idx, result in enumerate(search_result.web):
                    title = getattr(result, 'title', 'Unknown Source')
                    url = getattr(result, 'url', '')
                    description = getattr(result, 'description', '')

                    # Prioritize scraping first 3 URLs for full content
                    if idx < 1 and url and len(scrape_urls) < 3:
                        scrape_urls.append((url, title))

                    if description:
                        print(f"[DataFetcher] Raw description length: {len(description)}")
                        cleaned_desc = clean_text(description)
                        print(f"[DataFetcher] Cleaned description length: {len(cleaned_desc)}")
                        if len(cleaned_desc) >= 100:
                            source_marker = f"\n\n## Source: {title}\nURL: {url}\n\n"
                            aggregated_content.append(source_marker + cleaned_desc)
                            print(f"[DataFetcher] Added result: {title[:50]}... ({len(cleaned_desc)} chars)")
                        else:
                            print(f"[DataFetcher] Skipped result (too short): {title[:50]}... ({len(cleaned_desc)} chars)")
            elif hasattr(search_result, 'data'):
                for result in search_result.data:
                    title = getattr(result, 'title', 'Unknown Source')
                    url = getattr(result, 'url', '')
                    markdown = getattr(result, 'markdown', '')

                    if markdown:
                        print(f"[DataFetcher] Raw markdown length: {len(markdown)}")
                        cleaned_markdown = clean_text(markdown)
                        print(f"[DataFetcher] Cleaned markdown length: {len(cleaned_markdown)}")
                        if len(cleaned_markdown) >= 100:
                            source_marker = f"\n\n## Source: {title}\nURL: {url}\n\n"
                            aggregated_content.append(source_marker + cleaned_markdown)
                            print(f"[DataFetcher] Added result: {title[:50]}... ({len(cleaned_markdown)} chars)")
                        else:
                            print(f"[DataFetcher] Skipped result (too short): {title[:50]}... ({len(cleaned_markdown)} chars)")
            elif isinstance(search_result, dict):
                for result in search_result.get("data", []):
                    if isinstance(result, dict):
                        title = result.get("title", "Unknown Source")
                        url = result.get("url", "")
                        markdown = result.get("markdown", "")

                        if markdown:
                            print(f"[DataFetcher] Raw markdown length: {len(markdown)}")
                            cleaned_markdown = clean_text(markdown)
                            print(f"[DataFetcher] Cleaned markdown length: {len(cleaned_markdown)}")
                            if len(cleaned_markdown) >= 100:
                                source_marker = f"\n\n## Source: {title}\nURL: {url}\n\n"
                                aggregated_content.append(source_marker + cleaned_markdown)
                                print(f"[DataFetcher] Added result: {title[:50]}... ({len(cleaned_markdown)} chars)")
                            else:
                                print(f"[DataFetcher] Skipped result (too short): {title[:50]}... ({len(cleaned_markdown)} chars)")

        new_content = "\n\n---\n\n".join(aggregated_content)

        # Quality improvement: limit total results to prevent noise
        # Keep only top 12 search results to improve signal-to-noise ratio
        max_results = 12
        if len(aggregated_content) > max_results:
            print(f"[DataFetcher] Limiting from {len(aggregated_content)} to {max_results} best results")
            new_content = "\n\n---\n\n".join(aggregated_content[:max_results])

        # Attempt to scrape full content from top URLs for richer data
        print(f"[DataFetcher] Attempting to scrape {len(scrape_urls)} URLs for full content...")
        for url, title in scrape_urls:
            try:
                scrape_result = app.scrape(url, formats=["markdown"])
                if scrape_result and "markdown" in scrape_result:
                    full_content = scrape_result["markdown"]
                    if full_content:
                        cleaned_content = clean_text(full_content)
                        if len(cleaned_content) > len("") and len(cleaned_content) < 5000:
                            source_marker = f"\n\n## Source (Full Text): {title}\nURL: {url}\n\n"
                            new_content = source_marker + cleaned_content + "\n\n---\n\n" + new_content
                            print(f"[DataFetcher] Scraped full content from {title} ({len(cleaned_content)} chars)")
            except Exception as e:
                print(f"[DataFetcher] Scrape failed for {title}: {str(e)[:100]}")
                # Continue with search results if scrape fails

        existing_context = state.get("research_context", "")
        if existing_context:
            final_content = new_content + "\n\n---\n\n" + existing_context
        else:
            final_content = new_content

        final_content = final_content[:15000]

        print(f"[DataFetcher] Aggregated {len(aggregated_content)} search results, total context length: {len(final_content)} characters")

        return {
            "research_context": final_content,
            "iteration_count": iteration,
            "messages": state["messages"]
            + [
                AIMessage(
                    content=f"Successfully fetched data for {company_name}. Found {len(aggregated_content)} relevant articles."
                )
            ],
        }

    except Exception as e:

        return {
            "failure_reason": f"Data fetching failed: {str(e)}",
            "messages": state["messages"]
            + [AIMessage(content=f"Error fetching data: {str(e)}")],
        }


# -----------------------------
# Bull Analyst (Growth Investor)
# -----------------------------

def bull_analyst(state: AgentState):

    print("[BullAnalyst] Running bullish financial analysis")

    try:

        research_context = state.get("research_context", "")

        if research_context:
            print("[BullAnalyst] Filtering context for growth and positive indicators...")

            try:
                from langchain_community.vectorstores import FAISS

                embeddings = OpenAIEmbeddings(
                    base_url=config.SILICONFLOW_BASE_URL,
                    api_key=config.SILICONFLOW_API_KEY,
                    model=config.SILICONFLOW_EMBED_MODEL,
                    # BGE 模型上下文 512 token,langchain 默认批量=1000 容易触发 SiliconFlow 400
                    chunk_size=10,
                )

                chunks = research_context.split("\n\n---\n\n")
                documents = []
                _MAX_CHARS = 1200  # BGE 模型 512 token 限制,保险取 1200 字符
                for _chunk in chunks:
                    if not _chunk.strip():
                        continue
                    if len(_chunk) > _MAX_CHARS:
                        _step = _MAX_CHARS - 100
                        for _i in range(0, len(_chunk), _step):
                            _sub = _chunk[_i:_i + _MAX_CHARS]
                            if _sub.strip():
                                documents.append(Document(page_content=_sub))
                    else:
                        documents.append(Document(page_content=_chunk))

                if len(documents) > 0:
                    if len(documents) > 30:
                        print(f"[BullAnalyst] Too many chunks ({len(documents)}), selecting top 30")
                        documents = documents[:30]

                    vectorstore = FAISS.from_documents(documents, embeddings)

                    growth_queries = [
                        "revenue growth earnings increase",
                        "market share expansion AI leadership",
                        "quarterly results performance",
                        "positive outlook guidance"
                    ]

                    relevant_docs = []
                    for query in growth_queries:
                        docs = vectorstore.similarity_search(query, k=3)
                        relevant_docs.extend(docs)

                    seen_content = set()
                    unique_docs = []
                    for doc in relevant_docs:
                        content_hash = hash(doc.page_content)
                        if content_hash not in seen_content:
                            seen_content.add(content_hash)
                            unique_docs.append(doc)

                    top_k_docs = unique_docs[:10]

                    filtered_context = "\n\n---\n\n".join([doc.page_content for doc in top_k_docs])
                    research_context = filtered_context

                    print(f"[BullAnalyst] Filtered from {len(documents)} chunks to {len(top_k_docs)} most bullish chunks")
            except ImportError as e:
                print(f"[BullAnalyst] FAISS not installed, using keyword-based filter. Install with: pip install faiss-cpu")
                print(f"[BullAnalyst] Error details: {str(e)}")

                chunks = research_context.split("\n\n---\n\n")
                growth_keywords = ["growth", "increase", "revenue", "earnings", "profit", "expansion", "leadership", "positive", "outlook", "guidance", "record", "strong"]

                relevant_chunks = []
                for chunk in chunks:
                    chunk_lower = chunk.lower()
                    has_growth = any(keyword in chunk_lower for keyword in growth_keywords)

                    if has_growth:
                        relevant_chunks.append(chunk)

                if relevant_chunks:
                    research_context = "\n\n---\n\n".join(relevant_chunks)
                    print(f"[BullAnalyst] Keyword filter: {len(chunks)} -> {len(relevant_chunks)} bullish chunks")
            except Exception as e:
                print(f"[BullAnalyst] Vector filtering failed, using keyword-based filter: {str(e)}")

                chunks = research_context.split("\n\n---\n\n")
                growth_keywords = ["growth", "increase", "revenue", "earnings", "profit", "expansion", "leadership", "positive", "outlook", "guidance", "record", "strong"]

                relevant_chunks = []
                for chunk in chunks:
                    chunk_lower = chunk.lower()
                    has_growth = any(keyword in chunk_lower for keyword in growth_keywords)

                    if has_growth:
                        relevant_chunks.append(chunk)

                if relevant_chunks:
                    research_context = "\n\n---\n\n".join(relevant_chunks)
                    print(f"[BullAnalyst] Keyword filter: {len(chunks)} -> {len(relevant_chunks)} bullish chunks")

        llm = ChatOpenAI(
            base_url=config.SILICONFLOW_BASE_URL,
            api_key=config.SILICONFLOW_API_KEY,
            model=config.SILICONFLOW_LLM_MODEL,
            temperature=0.7,
        )

        system_prompt = """
You are an aggressive growth investor and bullish financial analyst.

CRITICAL RULES FOR DATA RECOGNITION:
1. ONLY use the provided context. Do NOT use any external knowledge.
2. NEVER hallucinate or invent data that is not in the context.
3. RECOGNIZE data in THREE categories:
   Category A: Explicitly stated (e.g., "Q4 2026 Revenue: $68.1 billion") → Report AS IS
   Category B: Projected/Expected/Guidance (e.g., "expected to reach $416B", "projected revenue", "guidance for 2026") → Report WITH [Source] notation
   Category C: Missing (not mentioned anywhere) → Mark as "数据不足" (Data not available)
4. PROHIBITED ACTIONS: Do NOT perform complex mathematical calculations or derivations from stated numbers.
5. If a number is NOT explicitly stated AND NOT in projected/guidance form, mark it as "未直接提及" (Not explicitly mentioned).
6. Your goal is to maximize faithfulness to the source material - 100% accuracy is required.

GROWTH ANALYST FOCUS:
7. Deeply dig into revenue growth, AI market leadership, and positive data from the context.
8. In the ## Key Financial Metrics section, prioritize:
   - YoY and QoQ growth rates that are explicitly mentioned in the context
   - Official guidance or projections for future periods (clearly label as "Projected" or "Guidance")
   - Historical actuals if available
9. WHEN you encounter projected data (e.g., "Apple is expected to achieve $416 billion in revenue for 2026"):
   - DO include it in analysis with clear source attribution
   - Format as: "**2026 Projected Revenue:** $416 billion (Source: Apple at the $4 Trillion Threshold: A 2026 Deep Dive...)"
   - This is NOT "hallucination" - it's forward-looking data from the source material

CITATION REQUIREMENT:
10. When presenting any data or claim, you MUST cite the exact phrase from the context using quotation marks.
11. Examples:
    - Actual: "As stated in the context: 'quarterly revenue of $102.5 billion, up 8 percent year over year'"
    - Projected: "As stated in the context: 'Apple is expected to achieve $416B in revenue in 2026'"

OUTPUT FORMAT (only include sections with actual data):
## Bullish Executive Summary
## Key Financial Metrics (Growth Focus)
## Market Leadership Analysis
## Growth Catalysts
## Bullish Investment Thesis

CRITICAL: Please distinguish between ACTUALS and PROJECTIONS in your data reporting.
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Context:\n{research_context}"),
        ]

        response = llm.invoke(messages)

        return {
            "messages": state["messages"] + [AIMessage(content=f"[BULL ANALYSIS]\n{response.content}")],
            "bull_analysis": response.content
        }

    except Exception as e:

        return {
            "failure_reason": f"Bull analysis failed: {str(e)}",
            "messages": state["messages"]
            + [AIMessage(content=f"Bull analysis error: {str(e)}")],
            "bull_analysis": ""
        }


# -----------------------------
# Bear Analyst (Short Seller)
# -----------------------------

def bear_analyst(state: AgentState):

    print("[BearAnalyst] Running bearish financial analysis")

    try:

        research_context = state.get("research_context", "")

        if research_context:
            print("[BearAnalyst] Filtering context for risks and negative indicators...")

            try:
                from langchain_community.vectorstores import FAISS

                embeddings = OpenAIEmbeddings(
                    base_url=config.SILICONFLOW_BASE_URL,
                    api_key=config.SILICONFLOW_API_KEY,
                    model=config.SILICONFLOW_EMBED_MODEL,
                    # BGE 模型上下文 512 token,langchain 默认批量=1000 容易触发 SiliconFlow 400
                    chunk_size=10,
                )

                chunks = research_context.split("\n\n---\n\n")
                documents = []
                _MAX_CHARS = 1200  # BGE 模型 512 token 限制,保险取 1200 字符
                for _chunk in chunks:
                    if not _chunk.strip():
                        continue
                    if len(_chunk) > _MAX_CHARS:
                        _step = _MAX_CHARS - 100
                        for _i in range(0, len(_chunk), _step):
                            _sub = _chunk[_i:_i + _MAX_CHARS]
                            if _sub.strip():
                                documents.append(Document(page_content=_sub))
                    else:
                        documents.append(Document(page_content=_chunk))

                if len(documents) > 0:
                    if len(documents) > 30:
                        print(f"[BearAnalyst] Too many chunks ({len(documents)}), selecting top 30")
                        documents = documents[:30]

                    vectorstore = FAISS.from_documents(documents, embeddings)

                    risk_queries = [
                        "risk assessment challenges threats",
                        "competition ASIC market share",
                        "downside vulnerability concerns",
                        "decline loss uncertainty"
                    ]

                    relevant_docs = []
                    for query in risk_queries:
                        docs = vectorstore.similarity_search(query, k=3)
                        relevant_docs.extend(docs)

                    seen_content = set()
                    unique_docs = []
                    for doc in relevant_docs:
                        content_hash = hash(doc.page_content)
                        if content_hash not in seen_content:
                            seen_content.add(content_hash)
                            unique_docs.append(doc)

                    top_k_docs = unique_docs[:10]

                    filtered_context = "\n\n---\n\n".join([doc.page_content for doc in top_k_docs])
                    research_context = filtered_context

                    print(f"[BearAnalyst] Filtered from {len(documents)} chunks to {len(top_k_docs)} most bearish chunks")
            except ImportError as e:
                print(f"[BearAnalyst] FAISS not installed, using keyword-based filter. Install with: pip install faiss-cpu")
                print(f"[BearAnalyst] Error details: {str(e)}")

                chunks = research_context.split("\n\n---\n\n")
                risk_keywords = ["risk", "challenge", "threat", "concern", "uncertainty", "downside", "vulnerability", "decline", "loss", "competition", "competitor", "asic", "weakness"]

                relevant_chunks = []
                for chunk in chunks:
                    chunk_lower = chunk.lower()
                    has_risk = any(keyword in chunk_lower for keyword in risk_keywords)

                    if has_risk:
                        relevant_chunks.append(chunk)

                if relevant_chunks:
                    research_context = "\n\n---\n\n".join(relevant_chunks)
                    print(f"[BearAnalyst] Keyword filter: {len(chunks)} -> {len(relevant_chunks)} bearish chunks")
            except Exception as e:
                print(f"[BearAnalyst] Vector filtering failed, using keyword-based filter: {str(e)}")

                chunks = research_context.split("\n\n---\n\n")
                risk_keywords = ["risk", "challenge", "threat", "concern", "uncertainty", "downside", "vulnerability", "decline", "loss", "competition", "competitor", "asic", "weakness"]

                relevant_chunks = []
                for chunk in chunks:
                    chunk_lower = chunk.lower()
                    has_risk = any(keyword in chunk_lower for keyword in risk_keywords)

                    if has_risk:
                        relevant_chunks.append(chunk)

                if relevant_chunks:
                    research_context = "\n\n---\n\n".join(relevant_chunks)
                    print(f"[BearAnalyst] Keyword filter: {len(chunks)} -> {len(relevant_chunks)} bearish chunks")

        llm = ChatOpenAI(
            base_url=config.SILICONFLOW_BASE_URL,
            api_key=config.SILICONFLOW_API_KEY,
            model=config.SILICONFLOW_LLM_MODEL,
            temperature=0.7,
        )

        system_prompt = """
You are a cautious short seller and bearish financial analyst.

CRITICAL RULES:
1. ONLY use the provided context. Do NOT use any external knowledge.
2. NEVER hallucinate or invent data that is not in the context.
3. If specific data is missing from the context, explicitly state "数据不足" (Data not available) for that section.
4. Do NOT generate generic descriptions or assumptions. If the context doesn't contain specific information, say "数据不足".
5. STRICTLY PROHIBITED: Do NOT perform complex mathematical calculations or derivations.
6. If a number is not explicitly stated in the context, mark it as "未直接提及" (Not explicitly mentioned) instead of calculating it.
7. Your goal is to maximize faithfulness to the source material - 100% accuracy is required.
8. FOCUS ON RISKS: Deeply analyze risk factors, competitor threats (like ASIC), and data unsustainability.
9. In the ## Key Financial Metrics section, prioritize listing year-over-year growth rates (YoY) and quarter-over-quarter growth rates (QoQ) that are explicitly mentioned in the context. If official guidance is available, include it.
10. CITATION REQUIREMENT: When presenting any data or claim, you MUST cite the exact phrase from the context using quotation marks.

Examples of PROHIBITED behavior:
- If context says "revenue grew 20% from Q3", do NOT calculate Q3 revenue
- If context says "up 73% from same period last year", do NOT calculate last year's revenue
- If context mentions percentage changes without base numbers, do NOT derive the base numbers
- If context mentions "Year Ago Sales" without explicit numbers, do NOT calculate them

Examples of CORRECT behavior:
- Report the exact numbers as stated in the context
- If context says "Q4 2026 Revenue: $68.1 billion", report exactly that
- If context doesn't mention net income, write "未直接提及" or "数据不足"
- If context mentions percentages without base numbers, report the percentages only
- If context mentions YoY or QoQ growth rates, list them explicitly in ## Key Financial Metrics
- If context mentions official guidance, include it in the analysis
- When citing data, use exact quotes: "As stated in the context: 'rising ASIC competition'"
- CRITICAL: Keep the original number format exactly as it appears in the context. If the context says "$68.1 billion", do NOT convert it to "681 亿" or any other format. Preserve the original units and number format.

Output format (only include sections with actual data):

## Bearish Executive Summary
## Key Financial Metrics (Risk Focus)
## Competitive Threats Analysis
## Risk Assessment
## Bearish Investment Thesis

For each section, if the context lacks sufficient information, write "数据不足" or "未直接提及" and move to the next section.
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Context:\n{research_context}"),
        ]

        response = llm.invoke(messages)

        return {
            "messages": state["messages"] + [AIMessage(content=f"[BEAR ANALYSIS]\n{response.content}")],
            "bear_analysis": response.content
        }

    except Exception as e:

        return {
            "failure_reason": f"Bear analysis failed: {str(e)}",
            "messages": state["messages"]
            + [AIMessage(content=f"Bear analysis error: {str(e)}")],
            "bear_analysis": ""
        }


# -----------------------------
# Debater Node (Final Synthesis)
# -----------------------------

def debater_node(state: AgentState):

    print("[Debater] Synthesizing bull and bear analyses")

    try:

        bull_analysis = state.get("bull_analysis", "")
        bear_analysis = state.get("bear_analysis", "")
        research_context = state.get("research_context", "")

        if not bull_analysis or not bear_analysis:
            print("[Debater] Missing bull or bear analysis, using fallback")
            return {
                "messages": state["messages"]
                + [AIMessage(content="Debate synthesis failed: missing analyses")]
            }

        llm = ChatOpenAI(
            base_url=config.SILICONFLOW_BASE_URL,
            api_key=config.SILICONFLOW_API_KEY,
            model=config.SILICONFLOW_LLM_MODEL,
            temperature=0.5,
        )

        system_prompt = """
You are an impartial financial analyst synthesizing a debate between a bullish investor and a bearish short seller. Act as a senior auditor cross-referencing data sources.

CRITICAL RULES:
1. ONLY use the provided context and both analyses. Do NOT use any external knowledge.
2. Extract ALL specific numerical data from both analyses with proper citations.
3. NEVER hallucinate or invent data not in the source material.
4. STRICTLY PROHIBITED: Do NOT perform complex mathematical calculations or derivations.
5. STRICTLY PROHIBITED: Do NOT fabricate or invent "official" data points, correction values, or reconciliation numbers.
   - If Bull cited 19% and Bear cited 18% but context has no reconciliation → Report BOTH and state explicitly "No official reconciliation found in provided sources"
   - Do NOT invent a middle value like "18.5%" or claim "Official source shows X%" without explicit citation
   - If you cannot verify which is correct, say: "Data conflict unresolved - Bull analysis claims 19%, Bear analysis claims 18%. Both require verification from official filings."

6. When data is missing, explicitly state the gap AND analyze its risk implication:
   - MISSING REVENUE → Risk: Inability to assess growth trajectory
   - MISSING MARGINS → Risk: Cannot evaluate profitability quality
   - MISSING GUIDANCE → Risk: Increased forecast uncertainty for investors

7. DATA CONFLICT RESOLUTION (Evidence-Based Only):
   a) Search the original_context for keywords like "posted quarterly revenue", "reported earnings", "fiscal year", "official filing"
   b) If original_context contains explicit correction/update, use that ONLY
   c) Identify which data point comes from the most recent dated source in the context
   d) Determine the data year/quarter for each conflicting claim
   e) ONLY if context explicitly provides official correction, state: "Bull analysis cited [older data], Bear analysis cited [newer data]. Official correction: [cite directly from source]"
   f) If NO official source resolves the conflict, explicitly report both values as unresolved: "Conflicting metrics detected: Value A from [Source 1], Value B from [Source 2]. Context does not provide official reconciliation."

8. Use clear structure with 5 mandatory sections: Executive Summary, Key Financial Metrics, Market Analysis, Risk Assessment, Investment Recommendation
9. CITATION FORMAT: Every numerical claim must cite as "metric value (Source: Title, Year: YYYY)" or [Source: Title, Year: YYYY]
10. OUTPUT REQUIREMENTS: Minimum 1200 characters with ≥15 citations, explicit data conflict resolutions, and risk analysis for each gap
11. Conclude with: "Key Risk Factors for [Growth/Conservative/Value] Investors: [list 3-4 specific risks]"

When presenting missing data, use:
- "Q3 2024 revenue: Not disclosed (Risk Implication: Inability to assess quarterly growth seasonality)"

When CANNOT resolve conflicting data, use:
- "Data Discrepancy: Q2 2024 margin reported as 19% (Bull source) vs 18% (Bear source). No official reconciliation in provided context. Both values require verification from SEC filings."

This converts conflicts into actionable insights while maintaining DATA INTEGRITY.
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
Original Context (Financial Data Source):
{research_context}

Bullish Analysis:
{bull_analysis}

Bearish Analysis:
{bear_analysis}

TASK: Synthesize into a comprehensive balanced report that:
1. Includes all disclosed financial metrics with citations
2. For missing metrics, explains risk implications
3. Resolves disagreements between analysts explicitly
4. Provides actionable investment guidance for different investor profiles

Please provide COMPLETE financial analysis with mandatory sections.
"""),
        ]

        response = llm.invoke(messages)

        return {
            "messages": state["messages"] + [AIMessage(content=response.content)]
        }

    except Exception as e:

        return {
            "failure_reason": f"Debate synthesis failed: {str(e)}",
            "messages": state["messages"]
            + [AIMessage(content=f"Debate synthesis error: {str(e)}")],
        }


# -----------------------------
# Revenue Visualization Node
# -----------------------------

def quality_inspector(state: AgentState):

    print("[QualityInspector] Running quality evaluation")

    try:

        research_context = state.get("research_context", "")

        messages = state["messages"]

        analyst_output = ""

        for msg in messages:
            if isinstance(msg, AIMessage):
                content = msg.content
                if not content.startswith("Successfully fetched data") and not content.startswith("Quality Check") and not content.startswith("Reflection:") and not content.startswith("[BULL ANALYSIS]") and not content.startswith("[BEAR ANALYSIS]"):
                    analyst_output = content

        if not analyst_output:
            analyst_output = "No analysis generated"

        company_name = (
            messages[0].content
            if messages and isinstance(messages[0], HumanMessage)
            else "Unknown"
        )

        print(f"[QualityInspector] Using RAGAS evaluation with SiliconFlow LLM and embeddings")

        llm = ChatOpenAI(
            base_url="https://api.siliconflow.cn/v1",
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            model="Qwen/Qwen2.5-7B-Instruct",
            temperature=0.0,
        )

        embeddings = OpenAIEmbeddings(
            base_url="https://api.siliconflow.cn/v1",
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            model="BAAI/bge-large-en-v1.5"
        )

        chunks = research_context.split("\n\n---\n\n")

        cleaned_contexts = []
        for chunk in chunks:
            if len(chunk.strip()) > 50 and len(chunk.strip()) < 4000:
                cleaned_contexts.append(chunk.strip())

        if len(cleaned_contexts) > 20:
            print(f"[QualityInspector] Too many contexts ({len(cleaned_contexts)}), selecting top 20")
            cleaned_contexts = cleaned_contexts[:20]

        key_keywords = ["Q4", "$", "billion", "revenue", "earnings", "EPS", "profit", "growth"]
        for chunk in chunks:
            chunk_stripped = chunk.strip()
            if chunk_stripped not in cleaned_contexts:
                has_keyword = any(keyword in chunk_stripped for keyword in key_keywords)
                if has_keyword:
                    cleaned_contexts.insert(0, chunk_stripped)
                    print(f"[QualityInspector] Forced inclusion of key segment with keywords: {chunk_stripped[:50]}...")

        if len(cleaned_contexts) > 25:
            print(f"[QualityInspector] After keyword inclusion, too many contexts ({len(cleaned_contexts)}), selecting top 25")
            cleaned_contexts = cleaned_contexts[:25]

        if not cleaned_contexts:
            cleaned_contexts = [research_context[:4000]]
            print(f"[QualityInspector] Using truncated context as fallback")

        dataset = EvaluationDataset.from_list(
            [
                {
                    "user_input": f"Provide a detailed financial analysis for {company_name}, including revenue growth, key financial metrics, industry competition risks, and investment recommendations. Please return evaluation results strictly in JSON format, without any leading words or explanatory text.",
                    "response": analyst_output,
                    "retrieved_contexts": cleaned_contexts,
                }
            ]
        )

        try:
            result = evaluate(
                dataset=dataset,
                metrics=[Faithfulness(llm=llm), AnswerRelevancy(llm=llm)],
                llm=llm,
                embeddings=embeddings,
                raise_exceptions=False,
                batch_size=20,
            )

            faith = result["faithfulness"]
            rel = result["answer_relevancy"]

            if isinstance(faith, list):
                faith = faith[0] if len(faith) > 0 else 0.0
            if isinstance(rel, list):
                rel = rel[0] if len(rel) > 0 else 0.0

            import math
            if math.isnan(faith):
                faith = 0.0
            if math.isnan(rel):
                rel = 0.0

            score = (faith + rel) / 2

            print(
                f"\n[Quality] Faithfulness {faith:.4f} | Relevancy {rel:.4f} | Avg {score:.4f}"
            )

            status = "PASS" if score >= 0.70 else "FAIL"

            return {
                "eval_score": score,
                "failure_reason": "" if score >= 0.70 else f"Score too low: {score}",
                "messages": state["messages"]
                + [
                    AIMessage(
                        content=f"Quality Check\nFaithfulness:{faith:.3f}\nRelevancy:{rel:.3f}\nScore:{score:.3f}\nStatus:{status}"
                    )
                ],
            }
        except Exception as ragas_error:
            print(f"[QualityInspector] RAGAS evaluation failed: {str(ragas_error)}")
            print(f"[QualityInspector] Falling back to heuristic evaluation")

            faith_score = 0.0
            rel_score = 0.0

            # Count structured elements
            has_executive_summary = "Executive Summary" in analyst_output
            has_key_metrics = "Key Financial Metrics" in analyst_output
            has_market_analysis = "Market Analysis" in analyst_output
            has_risk_assessment = "Risk Assessment" in analyst_output or "Competitive Threats" in analyst_output
            has_investment_thesis = "Investment Thesis" in analyst_output or "Investment Recommendation" in analyst_output

            # Count data elements
            has_revenue = any(keyword in analyst_output.lower() for keyword in ["revenue", "revenues", "$"])
            has_eps = any(keyword in analyst_output.lower() for keyword in ["eps", "earnings per share", "earning per share"])
            has_net_income = any(keyword in analyst_output.lower() for keyword in ["net income", "profit", "billion"])
            has_growth = any(keyword in analyst_output.lower() for keyword in ["growth", "increase", "decrease", "yoy", "qoq"])
            has_numbers = any(char.isdigit() for char in analyst_output)
            has_structure = any(keyword in analyst_output for keyword in ["##", "Executive Summary", "Key Financial"])
            has_citations = '[' in analyst_output and ']' in analyst_output  # Has [Source: ...] format

            # Count specific dollar amounts and percentages more strictly
            import re
            dollar_amounts = len(re.findall(r'\$[\d,\.]+\s*(billion|million|B|M)', analyst_output, re.IGNORECASE))
            percentages = len(re.findall(r'\d+(\.\d+)?%', analyst_output))

            # Calculate faithfulness: presence of data + structure + citations
            faith_score = 0.0
            if has_executive_summary:
                faith_score += 0.12
            if has_key_metrics:
                faith_score += 0.12
            if has_market_analysis:
                faith_score += 0.12
            if has_risk_assessment:
                faith_score += 0.10
            if has_investment_thesis:
                faith_score += 0.10
            if has_citations:
                faith_score += 0.15
            if dollar_amounts > 2:
                faith_score += 0.15
            if percentages > 2:
                faith_score += 0.04

            # Cap at 1.0
            faith_score = min(faith_score, 1.0)

            # Calculate relevancy: completeness + length + data density
            rel_score = 0.0

            # Structure + organization
            if has_structure and has_revenue and has_growth:
                rel_score += 0.30
            elif has_structure:
                rel_score += 0.20

            # Data presence
            if has_numbers and dollar_amounts >= 2:
                rel_score += 0.30
            elif has_numbers:
                rel_score += 0.15

            # Content length (longer = more complete/relevant)
            if len(analyst_output) > 2000:
                rel_score += 0.15
            elif len(analyst_output) > 1000:
                rel_score += 0.10

            # Section completeness bonus
            section_count = sum([
                has_executive_summary,
                has_key_metrics,
                has_market_analysis,
                has_risk_assessment,
                has_investment_thesis
            ])
            if section_count >= 4:
                rel_score += 0.10
            elif section_count >= 3:
                rel_score += 0.05

            rel_score = min(rel_score, 1.0)

            score = (faith_score + rel_score) / 2

            print(f"[Quality] Heuristic - Structure:{section_count}/5, Dollars:{dollar_amounts}, Percent:{percentages}, Faithfulness {faith_score:.4f} | Relevancy {rel_score:.4f} | Avg {score:.4f}")

            status = "PASS" if score >= 0.70 else "FAIL"

            return {
                "eval_score": score,
                "failure_reason": f"RAGAS failed: {str(ragas_error)[:100]}" if score < 0.70 else "",
                "messages": state["messages"]
                + [
                    AIMessage(
                        content=f"Quality Check (Heuristic Fallback)\nSections:{section_count}/5 | Data Points:{dollar_amounts+percentages}\nFaithfulness:{faith_score:.3f}\nRelevancy:{rel_score:.3f}\nScore:{score:.3f}\nStatus:{status}"
                    )
                ],
            }

    except Exception as e:

        print(f"[QualityInspector] Quality inspection failed: {str(e)}")
        print(f"[QualityInspector] Using minimal fallback score")

        return {
            "eval_score": 0.5,
            "failure_reason": f"Quality inspection failed: {str(e)}",
            "messages": state["messages"]
            + [
                AIMessage(
                    content=f"Quality Check (Emergency Fallback)\nError: {str(e)[:100]}\nScore:0.500\nStatus:FAIL"
                )
            ],
        }


# -----------------------------
# Reflection Agent
# -----------------------------



def _extract_structured_revenue(bull: str, bear: str, llm) -> list:
    """让 LLM 从多头/空头分析中提取结构化营收数据。
    返回 [{period, revenue_billion, yoy_percent}, ...];失败返回空列表。
    容忍 LLM 输出:markdown fence (\`\`\`json)、前后解释性文字、中文标点等。"""
    import json, re as _re

    def _clean(text):
        text = (text or "").strip()
        # 剥 markdown ```json ... ``` 围栏
        fence = _re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, _re.S)
        if fence:
            return fence.group(1).strip()
        # 找首个 '[' 到末尾最后一个 ']' (含跨行)
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            return text[start:end + 1]
        return text

    def _try_parse(text):
        try:
            return json.loads(text)
        except Exception:
            pass
        cleaned = (text or "")
        cleaned = cleaned.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
        cleaned = _re.sub(r",\s*([\]])", r"\1", cleaned)  # 去尾部逗号
        try:
            return json.loads(cleaned)
        except Exception:
            return None

    prompt = (
        "你是一名数据提取助手。请从下面的多头/空头分析中提取该公司的营收数据点。\n"
        "只输出一个 JSON 数组,前后不要任何其它文字、不要 markdown 包装、不要代码块标记。\n"
        "格式:[{\"period\":\"FY2024\",\"revenue_billion\":391.0,\"yoy_percent\":2.0}]\n"
        "其中:period 用 FY20XX 或 QXX 20XX;revenue_billion 与 yoy_percent 必须是数字(null 表示未知)。\n\n"
        f"[多头分析]\n{(bull or '')[:3000]}\n\n[空头分析]\n{(bear or '')[:3000]}"
    )
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        text = _clean(resp.content or "")
        data = _try_parse(text)
        if isinstance(data, list):
            out = []
            for it in data:
                if not isinstance(it, dict):
                    continue
                rev = it.get("revenue_billion")
                yoy = it.get("yoy_percent")
                out.append({
                    "period": str(it.get("period", f"#{len(out)+1}")),
                    "revenue_billion": float(rev) if isinstance(rev, (int, float)) else 0.0,
                    "yoy_percent": float(yoy) if isinstance(yoy, (int, float)) else None,
                })
            return out
    except Exception as e:
        print(f"[RevenueVisualizer] 结构化抽取失败: {e}")
    return []


def _fallback_regex_revenue(all_content: str) -> list:
    """正则兜底:抽取 $X billion / X% YoY 数据点(兼容旧项目逻辑)。"""
    import re as _re
    out = []
    pat = r"\$(\d+(?:\.\d+)?)\s*billion[^.]*?(\d+(?:\.\d+)?)\s*%\s*(?:YoY|year over year|growth)"
    for m in _re.finditer(pat, all_content, _re.IGNORECASE):
        try:
            out.append({"value": float(m.group(1)), "yoy": float(m.group(2))})
        except Exception:
            pass
    if not out:
        pat2 = r"\$(\d+(?:\.\d+)?)\s*(?:billion|B)"
        for m in _re.finditer(pat2, all_content):
            try:
                out.append({"value": float(m.group(1)), "yoy": None})
            except Exception:
                pass
    return out


def _render_revenue_chart(company_name: str, items: list) -> str:
    """画营收图(与 tools.draw_chart 渲染一致)。items 元素为
    {period, revenue_billion, yoy_percent} 或 {period, value, yoy}。"""
    config.ensure_dirs()
    periods = [str(it.get("period", f"#{i+1}")) for i, it in enumerate(items)]
    values = [float(it.get("revenue_billion", it.get("value", 0)) or 0) for it in items]
    yoys = [it.get("yoy_percent", it.get("yoy")) for it in items]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{company_name} Financial Analysis - Revenue Insights", fontsize=14, fontweight="bold")

    colors = ["#34C759" if (y is not None and y > 0) else "#FF453A" for y in yoys]
    ax1.barh(range(len(values)), values, color=colors, alpha=0.75, edgecolor="black")
    ax1.set_yticks(range(len(values)))
    ax1.set_yticklabels(periods)
    ax1.set_xlabel("Revenue (Billion USD)", fontweight="bold")
    ax1.set_title("Revenue by Period", fontweight="bold")
    ax1.grid(axis="x", alpha=0.3)
    for i, v in enumerate(values):
        ax1.text(v + max(values) * 0.01 + 0.5, i, f"${v:.1f}B", va="center", fontsize=9, fontweight="bold")

    yoy_data = [float(y) for y in yoys if y is not None]
    if yoy_data:
        colors_yoy = ["#34C759" if y > 0 else "#FF453A" for y in yoy_data]
        ax2.bar(range(len(yoy_data)), yoy_data, color=colors_yoy, alpha=0.75, edgecolor="black")
        ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax2.set_ylabel("YoY Growth (%)", fontweight="bold")
        ax2.set_title("Year-over-Year Growth", fontweight="bold")
        ax2.set_xticks(range(len(yoy_data)))
        ax2.set_xticklabels([p for p, y in zip(periods, yoys) if y is not None])
        ax2.grid(axis="y", alpha=0.3)
        for i, y in enumerate(yoy_data):
            ax2.text(i, y + (1 if y > 0 else -3), f"{y:.1f}%", ha="center", fontsize=9, fontweight="bold")
    else:
        ax2.text(0.5, 0.5, "YoY Growth Data\nNot Available", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=12, style="italic", color="gray")
        ax2.set_xticks([])
        ax2.set_yticks([])

    plt.tight_layout()
    path = os.path.join(config.OUTPUT_DIR, f"{config.safe_filename(company_name)}_revenue_analysis.png")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    return path


def revenue_visualizer(state: AgentState):
    """生成营收可视化:优先 LLM 结构化抽取,失败则正则兜底;渲染与 tools.draw_chart 一致。"""
    print("[RevenueVisualizer] Generating revenue trend visualization")
    try:
        messages = state.get("messages", [])
        bull_analysis = state.get("bull_analysis", "")
        bear_analysis = state.get("bear_analysis", "")
        company_name = (
            messages[0].content
            if messages and isinstance(messages[0], HumanMessage)
            else "Unknown"
        )

        # 1) 优先:LLM 结构化抽取
        llm = ChatOpenAI(
            base_url=config.SILICONFLOW_BASE_URL,
            api_key=config.SILICONFLOW_API_KEY,
            model=config.SILICONFLOW_LLM_MODEL,
            temperature=0.0,
        )
        items = _extract_structured_revenue(bull_analysis, bear_analysis, llm)
        if items and isinstance(items, list):
            print(f"[RevenueVisualizer] LLM 抽取到 {len(items)} 个数据点")
            path = _render_revenue_chart(company_name, items)
            visualization_msg = f"Revenue Visualization Generated\n- Data Points: {len(items)}\n- Chart: {path}"
        else:
            # 2) 兜底:正则抽取
            legacy = _fallback_regex_revenue(bull_analysis + "\n" + bear_analysis)
            if legacy:
                normalized = [
                    {"period": f"#{i+1}", "revenue_billion": r["value"], "yoy_percent": r.get("yoy")}
                    for i, r in enumerate(legacy)
                ]
                path = _render_revenue_chart(company_name, normalized)
                visualization_msg = (
                    f"Revenue Visualization (regex fallback)\n"
                    f"- Data Points: {len(legacy)}\n- Chart: {path}"
                )
            else:
                print("[RevenueVisualizer] No revenue data found in analyses")
                visualization_msg = "No revenue data found in analyses for visualization"
                path = None

        return {
            "messages": state["messages"] + [AIMessage(content=f"[VISUALIZATION]\n{visualization_msg}")]
        }
    except Exception as e:
        print(f"[RevenueVisualizer] Visualization failed: {str(e)}")
        return {
            "messages": state["messages"] + [AIMessage(content=f"Visualization error: {str(e)}")],
            "failure_reason": f"Visualization failed: {str(e)}",
        }


def reflect(state: AgentState):

    print("[Reflect] Running reflection")

    try:

        current_iteration = state.get("iteration_count", 0)
        iteration = current_iteration + 1

        llm = ChatOpenAI(
            base_url=config.SILICONFLOW_BASE_URL,
            api_key=config.SILICONFLOW_API_KEY,
            model=config.SILICONFLOW_LLM_MODEL,
            temperature=0.5,
        )

        failure_reason = state.get("failure_reason", "Unknown failure")
        research_context = state.get("research_context", "")
        eval_score = state.get("eval_score", 0.0)

        messages = state["messages"]
        company_name = (
            messages[0].content
            if messages and isinstance(messages[0], HumanMessage)
            else "Unknown"
        )

        has_financial_data = any(keyword in research_context for keyword in ["$", "%", "billion", "million", "revenue", "EPS", "net income"])
        context_length = len(research_context)
        has_dollar_sign = "$" in research_context

        if has_financial_data and context_length >= 12000 and has_dollar_sign:
            print("[Reflect] Context is rich with financial data (length >= 12,000 chars and contains $), focusing on format optimization")
            focus_instruction = """
CRITICAL: The context already contains substantial financial data (revenue, EPS, etc.) and dollar amounts.
STOP generating new search queries. The issue is likely that the analyst output is not well-structured or missing specific formatting requirements.
Focus on IMPROVING THE FORMAT and ORGANIZATION of the analysis report.
"""
        elif has_financial_data:
            print("[Reflect] Detected financial data in context, focusing on format optimization")
            focus_instruction = """
IMPORTANT: The context already contains financial data (revenue, EPS, etc.).
Focus on IMPROVING THE FORMAT and ORGANIZATION of the analysis rather than generating new search queries.
The issue is likely that analyst output is not well-structured or missing specific formatting requirements.
"""
        else:
            print("[Reflect] No financial data detected, focusing on data collection")
            focus_instruction = """
The context lacks sufficient financial data. Focus on generating specific search queries to gather missing information.
"""

        system_prompt = f"""
You are a reflection agent for a financial analysis system.

Your task is to analyze why the quality score is low and provide actionable improvements.

{focus_instruction}

CRITICAL: You MUST respond ONLY with valid JSON in this exact format, nothing else:

```json
{{
  "problem_analysis": "Brief explanation of why score is low",
  "queries": [
    "[Company] specific metric year/quarter",
    "[Company] another specific financial metric",
    "[Company] third financial data point",
    "[Company] fourth financial data point"
  ],
  "format_suggestions": ["suggestion 1", "suggestion 2", "suggestion 3"]
}}
```

RULES:
- Each query must be a concrete financial search term with company name, metric, and time period
- Never include generic words like "report", "results", "analysis" without specific metrics
- Format suggestions should only be included if format is the issue
- Always output valid JSON - no markdown code blocks, no extra text before or after
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=f"Company: {company_name}\nQuality Score: {eval_score:.4f}\nFailure Reason: {failure_reason}\nContext Length: {len(research_context)} characters"
            ),
        ]

        response = llm.invoke(messages)
        reflection_text = response.content

        print(f"[Reflect] Generated reflection, parsing JSON queries...")

        suggested_queries = []

        # Try to parse JSON response
        try:
            # Extract JSON from response (might have markdown wrappers)
            json_text = reflection_text
            if "```json" in json_text:
                json_text = json_text.split("```json")[1].split("```")[0]
            elif "```" in json_text:
                json_text = json_text.split("```")[1].split("```")[0]

            parsed = json.loads(json_text)

            if "queries" in parsed and isinstance(parsed["queries"], list):
                suggested_queries = [q.strip() for q in parsed["queries"] if q.strip() and len(q.strip()) > 10]
                print(f"[Reflect] JSON parsing successful, extracted {len(suggested_queries)} queries")
                for i, query in enumerate(suggested_queries[:4], 1):
                    print(f"[Reflect] Query {i}: {query}")
            else:
                print(f"[Reflect] JSON valid but no 'queries' field found")
        except json.JSONDecodeError as e:
            print(f"[Reflect] JSON parse failed: {str(e)[:100]}, falling back to strategy extraction")

            # Fallback: Extract queries using old strategy 3 but more carefully
            lines = reflection_text.split('\n')
            for line in lines:
                # Only accept lines that look like actual queries (have metric names + year/company)
                if any(metric in line.lower() for metric in ['revenue', 'earnings', 'income', 'margin', 'cash', 'growth', 'q1', 'q2', 'q3', 'q4', '2024', '2025']):
                    # Reject lines that look like format suggestions
                    if not any(format_word in line.lower() for format_word in ['format', 'structure', 'section', 'heading', 'bullet', 'table', 'example']):
                        line_clean = line.strip().replace('**', '').replace('-', '').strip()
                        if line_clean and len(line_clean) > 15 and line_clean not in suggested_queries:
                            suggested_queries.append(line_clean)
                            if len(suggested_queries) >= 4:
                                break

        # Ensure we have 4 queries
        if len(suggested_queries) < 4:
            print(f"[Reflect] Only have {len(suggested_queries)} queries, adding defaults...")
            defaults = [
                f"{company_name} Q4 2024 revenue earnings per share net income",
                f"{company_name} gross margin operating margin profit 2024",
                f"{company_name} free cash flow competitive advantages 2025",
                f"{company_name} risk factors regulatory challenges market position"
            ]
            for default in defaults:
                if len(suggested_queries) >= 4:
                    break
                if default not in suggested_queries:
                    suggested_queries.append(default)

        suggested_queries = suggested_queries[:4]
        print(f"[Reflect] Final query count: {len(suggested_queries)}")

        return {
            "iteration_count": iteration,
            "suggested_queries": suggested_queries,
            "messages": state["messages"]
            + [AIMessage(content=f"Reflection:\n{reflection_text}")]
        }

    except Exception as e:

        return {
            "messages": state["messages"]
            + [AIMessage(content=f"Reflection error: {str(e)}")]
        }
