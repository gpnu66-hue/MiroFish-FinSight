"""安全机制(满足大作业"权限白名单"必做 + Prompt Injection 防护加分)。

- 权限白名单:不同角色(role)可调用不同工具,越权调用被拒(对应"越权调用"防护)。
- 注入检测:对用户输入及检索/抓取内容做基础 Prompt Injection 屏蔽(加分项 +3)。
"""
import re

# 角色 → 允许调用的工具名白名单。演示用:访客(guest)只能知识库问答,
# 不能联网抓取 / 跑深度财报分析 / 画图 —— 用于演示"越权被拒"。
ROLE_TOOLS = {
    "analyst": {"rag_search", "web_search", "financial_analysis", "draw_chart"},
    "guest": {"rag_search"},
}

DEFAULT_ROLE = "analyst"

# 常见 Prompt Injection / 越狱特征(中英文)
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"disregard\s+(all\s+)?(previous|prior)\s+",
    r"you\s+are\s+now\s+(a|an)\s+",
    r"(reveal|show|print|dump)\s+(your|the|all)\s+(system\s+)?prompt",
    r"jailbreak",
    r"developer\s+mode",
    r"</?\s*(system|prompt|instruction)\s*>",
    r"忽略.{0,4}(之前|上面|以上|前面).{0,2}(指令|提示|规则|约束)",
    r"你现在(是|扮演)",
    r"(输出|打印|显示|告诉我)(你的)?系统提示",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


def allowed_tools(role):
    """返回该角色允许调用的工具名集合。"""
    return set(ROLE_TOOLS.get(role, ROLE_TOOLS[DEFAULT_ROLE]))


def check_permission(role, tool_name):
    """检查 role 是否被允许调用 tool_name。"""
    return tool_name in allowed_tools(role)


def filter_tools(role, tools):
    """按角色白名单过滤工具对象列表(工具需有 .name 属性)。

    ReAct agent 只拿到被允许的工具,从结构上就无法越权调用 —— 这是
    最稳健的白名单实现方式。
    """
    names = allowed_tools(role)
    return [t for t in tools if getattr(t, "name", None) in names]


def screen_injection(text):
    """检测 text 是否含 Prompt Injection 特征。

    返回 (is_safe, matched):is_safe=False 时 matched 为命中片段。
    """
    if not text:
        return True, None
    m = _INJECTION_RE.search(text)
    if m:
        return False, m.group(0)
    return True, None
