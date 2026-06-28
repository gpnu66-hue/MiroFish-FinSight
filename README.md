# MiroFish FinSight

基于国产大模型(Qwen)的**金融分析智能体**:以 ReAct 编排为"大脑",自主决策调用工具完成问答与深度分析;深度财报分析由多智能体(多头/空头/辩论/质检/反思)流水线完成。

> 在一个轻量 LangGraph 多智能体金融分析项目基础上做"课程化改造",补齐了 ReAct 决策、标准化工具、RAG 知识库、记忆与安全白名单,使其满足智能体大作业的全部必做要求。

## 核心特性
- **国产模型驱动**:全程使用国产 Qwen2.5-72B-Instruct via SiliconFlow(全栈统一 72B,保证 RAGAS / 结构化 JSON 输出稳定)。
- **ReAct 自主决策**:智能体"思考→行动→观察→回答",自主选择是否调用工具。
- **多智能体深度分析**:DataFetcher → Bull → Bear → Debater → QualityInspector → Reflect,带 RAGAS 评分与反思迭代。
- **RAG 知识库**:本地金融文档库(≥5 篇),Embedding + FAISS 向量检索 + 关键词重排序。
- **记忆机制**:短期记忆(截最近 N 轮)+ 可选长期记忆(跨会话 JSON)。
- **安全防护**:角色权限白名单 + Prompt Injection 检测。

## 架构

```
CLI 聊天循环 (main.py)
  └─ ReAct 编排 agent (create_react_agent · Qwen2.5-72B)   ← Agent 大脑
        ├─ rag_search         检索本地知识库 (rag.py + knowledge/ + FAISS)
        ├─ web_search         联网搜索 (Firecrawl)
        ├─ financial_analysis 多智能体深度财报分析 (agents.py 流水线)
        └─ draw_chart         绘制营收柱状图
  + memory.py    短期/长期记忆
  + security.py  角色白名单 + 注入检测
  + config.py    集中配置
```

## 必做模块对照

| 大作业要求 | 实现位置 |
|---|---|
| 国产模型 | `config.py`(Qwen,可配置) |
| ReAct 循环 | `main.py` `build_react_agent` / `chat_loop` |
| ≥2 自定义工具 | `tools.py`(rag_search / web_search / financial_analysis / draw_chart) |
| RAG 知识库(≥5 文档) | `rag.py` + `knowledge/`(6 篇)+ FAISS 余弦 + 关键词重排 |
| 短期记忆 | `memory.py` `trim_messages` |
| 安全白名单 | `security.py` `ROLE_TOOLS` / `filter_tools` |
| (加分)反思机制 | `agents.py` `reflect` 节点 |
| (加分)Prompt Injection 防护 | `security.py` `screen_injection` |
| (加分)边界测试 | `benchmark_tasks.md` |
| (加分)多模型切换 | 全栈 72B(可按角色切换) |

## 环境配置
1. Python 环境(本项目开发环境:`D:\conda_envs\dataAgent`)。
2. 安装依赖:`pip install -r requirements.txt`
3. 复制并填写密钥:`copy .env_example .env`,至少填 `SILICONFLOW_API_KEY`。
   - `FIRECRAWL_API_KEY` 可选;不填则 `web_search`/深度分析退化为仅知识库。

## 运行
```bash
# 交互式 ReAct 智能体(主入口)
python main.py

# 直接跑多智能体深度财报分析流水线(旧入口/工具内部也调用)
python main.py --analyze "Apple Inc"
```

CLI 内命令:`/help` `/role <analyst|guest>` `/reset` `/remember <内容>` `/recall` `/quit`

## 目录结构
```
MiroFish-FinSight/
  main.py              ReAct 编排 + CLI;深度分析流水线
  agents.py            多智能体节点(bull/bear/debate/quality/reflect/visualizer)
  tools.py             4 个标准化工具
  rag.py               持久化 FAISS 知识库(构建/加载/混合检索)
  memory.py            短期记忆 trim + 长期记忆
  security.py          角色白名单 + 注入检测
  config.py            集中配置
  _ragas_compat.py     ragas/langchain-community 兼容垫片
  knowledge/           金融知识库(≥5 篇 .md)
  benchmark_tasks.md   演示评测任务(≥10 条)
```

## 工具列表
| 工具 | 说明 |
|---|---|
| `rag_search(query)` | 检索本地金融知识库 |
| `web_search(query)` | 联网搜索(Firecrawl) |
| `financial_analysis(company)` | 多智能体深度财报分析 |
| `draw_chart(company, revenue_data_json)` | 依据结构化数据画营收柱状图 |

## 备注
- `import agents` 依赖 `_ragas_compat.py` 解决 ragas 0.4.3 与 langchain-community 0.4.2 的 import 冲突(项目不使用 VertexAI)。
- RAG 知识库首次构建后会落盘到 `kb_index/`;修改 `knowledge/` 后删除 `kb_index/` 重建。
- 图表与索引输出到 `output/`、`kb_index/`(可由 `OUTPUT_DIR` 覆盖)。
