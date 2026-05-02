# MiroFish FinSight

MiroFish FinSight 是一个基于 LangGraph 的多智能体金融分析引擎，面向公司财报、行业资讯与投资研究场景，自动完成数据抓取、正反观点辩论、质量评估、反思迭代和结果可视化。

项目核心流程包括：
- 从网络抓取企业相关财务与市场信息
- 分别生成多头分析和空头分析
- 通过辩论节点合成平衡观点
- 使用 RAGAS 评估分析结果的忠实度与相关性
- 在评分不足时自动反思并重试检索
- 输出营收趋势可视化图表

## 特点
- 多智能体协作：Data Fetcher / Bull Analyst / Bear Analyst / Debater / Quality Inspector / Reflector
- 强调可追溯性：尽量基于原始上下文生成分析
- 支持迭代优化：根据质量评分自动补充检索查询
- 自动生成图表：提取营收相关数据并绘制趋势图
- 适合研究型场景：公司分析、财报摘要、投资观点整理

## 技术栈
- Python
- LangGraph
- LangChain
- Firecrawl
- RAGAS
- Matplotlib
- OpenAI / SiliconFlow
