"""Agent 核心 — 构建与运行，三步链路：识别 → 精确查找 → 语义检索"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from config import load_config
from database import ShipDatabase
from tools import build_tools

logger = logging.getLogger(__name__)

# ── System Prompt ──────────────────────────────

SYSTEM_PROMPT = """你是船弦号识别助手。严格按以下三步链路工作：

【输入类型】
如果输入包含 "VLM 识别结果"，说明 VLM 已完成预识别，直接使用其中的弦号和描述信息执行后续步骤。
如果弦号为空字符串 ""，跳过第二步，直接进入第三步语义检索。

【三步链路】
1. 调用 recognize_ship 识别图像中的弦号和船只描述
   → 有弦号：进入第二步
   → 无弦号：跳过第二步，直接进入第三步

2. 调用 lookup_by_hull_number 精确查找（仅在有弦号时）
   → found=true：直接返回「库内确定id：{hull_number}，描述：{description}」
   → found=false：返回「未知id：{hull_number}，描述：{description}」，然后进入第三步

3. 调用 retrieve_by_description 语义检索
   → 用 recognize_ship 返回的 description 作为查询
   → 有结果：返回所有匹配「可能id：{弦号1}/{弦号2}/...」
   → 无结果：返回「未识别」

【禁止】
- 不要编造弦号或描述
- 不要跳过任何步骤
- 不要同时调用多个工具
"""

# ── 无 Few-shot 示例（避免误导 Agent）──


class AgentResult:
    """Agent 运行结果，包含结构化信息供 pipeline 使用。"""

    def __init__(
        self,
        hull_number: str = "",
        description: str = "",
        match_type: str = "none",
        semantic_match_ids: list[str] | None = None,
        answer: str = "",
    ):
        self.hull_number = hull_number
        self.description = description
        self.match_type = match_type        # "exact" | "semantic" | "none"
        self.semantic_match_ids = semantic_match_ids or []
        self.answer = answer


class ShipHullAgent:
    """船弦号识别 Agent 封装。三步链路：recognize_ship → lookup → retrieve。"""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or load_config()

        app_cfg = self.config.get("app", {})
        llm_cfg = self.config.get("llm", {})

        logging.basicConfig(
            level=getattr(logging, app_cfg.get("log_level", "INFO").upper(), logging.INFO),
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        )

        self.db = ShipDatabase(config=self.config)
        self.tools = build_tools(self.db)

        self._llm = ChatOpenAI(
            model=llm_cfg.get("model", "Qwen/Qwen3-VL-4B-AWQ"),
            api_key=llm_cfg.get("api_key", "abc123"),
            base_url=llm_cfg.get("base_url", "http://localhost:7890/v1"),
            temperature=llm_cfg.get("temperature", 0.0),
        )

        self._agent = create_react_agent(
            model=self._llm,
            tools=self.tools,
            prompt=SYSTEM_PROMPT,
        )

    def run(self, query: str) -> str:
        """运行 Agent，返回自然语言回答。"""
        logger.info("收到查询: %s", query[:100])
        try:
            result = self._agent.invoke({"messages": [HumanMessage(content=query)]})
            answer = result["messages"][-1].content
            logger.info("回答: %s", answer[:100])
            return answer
        except Exception as e:
            logger.exception("Agent 执行失败")
            return f"查询执行失败: {e}"

    def run_with_result(self, query: str) -> AgentResult:
        """运行 Agent，返回结构化结果（供 pipeline 使用）。"""
        try:
            # 判断是否为 VLM 预识别结果（文本模式）
            if query.startswith("VLM 识别结果"):
                return self._run_text_mode(query)

            result = self._agent.invoke({"messages": [HumanMessage(content=query)]})
            return self._parse_result(result)
        except Exception as e:
            logger.exception("Agent 执行失败")
            return AgentResult(answer=f"Agent 执行失败: {e}")

    def _run_text_mode(self, query: str) -> AgentResult:
        """
        文本模式：VLM 已完成预识别，直接执行 lookup → retrieve 两步。

        避免将 base64 图片传入 Agent 造成 token 超限。
        """
        import re

        # 解析 VLM 结果
        hull_match = re.search(r'弦号="([^"]*)"', query)
        desc_match = re.search(r'描述="([^"]*)"', query)
        hull_number = hull_match.group(1) if hull_match else ""
        description = desc_match.group(1) if desc_match else ""

        if not hull_number and not description:
            return AgentResult(answer="VLM 未返回有效结果")

        match_type = "none"
        semantic_ids: list[str] = []

        # 第二步：精确查找（有弦号时）
        if hull_number:
            desc_in_db = self.db.lookup(hull_number)
            if desc_in_db is not None:
                match_type = "exact"
                description = description or desc_in_db
                return AgentResult(
                    hull_number=hull_number,
                    description=description,
                    match_type=match_type,
                    semantic_match_ids=[],
                    answer=f"库内确定id：{hull_number}，描述：{description}",
                )
            elif description:
                # 第三步：语义检索（弦号未匹配时）
                results = self.db.semantic_search_filtered(description)
                semantic_ids = [r["hull_number"] for r in results if r.get("hull_number")]
                if semantic_ids:
                    match_type = "semantic"
        elif description:
            # 无弦号，直接第三步：语义检索
            results = self.db.semantic_search_filtered(description)
            semantic_ids = [r["hull_number"] for r in results if r.get("hull_number")]
            if semantic_ids:
                match_type = "semantic"

        answer_parts = []
        if hull_number:
            answer_parts.append(f"未知id：{hull_number}")
        else:
            answer_parts.append("未识别到弦号")
        if description:
            answer_parts.append(f"描述：{description}")
        if semantic_ids:
            answer_parts.append(f"可能id：{'/'.join(semantic_ids[:3])}")
        else:
            answer_parts.append("无语义匹配结果")

        logger.info(
            "文本模式完成: 弦号=%s, 匹配=%s, 语义候选=%s",
            hull_number or "(无)", match_type, semantic_ids,
        )

        return AgentResult(
            hull_number=hull_number,
            description=description,
            match_type=match_type,
            semantic_match_ids=semantic_ids,
            answer="，".join(answer_parts),
        )

    @staticmethod
    def _parse_result(result: dict) -> AgentResult:
        """从 Agent 消息历史中提取结构化结果。"""
        msgs = result.get("messages", [])
        hull_number = ""
        description = ""
        match_type = "none"
        semantic_match_ids: list[str] = []
        answer = msgs[-1].content if msgs else ""

        for msg in msgs:
            if not isinstance(msg, ToolMessage):
                continue
            try:
                data = json.loads(msg.content)
            except (json.JSONDecodeError, TypeError):
                continue

            # recognize_ship 结果
            if "hull_number" in data and "description" in data and "found" not in data and "results" not in data:
                hull_number = data.get("hull_number", "")
                description = data.get("description", "")

            # lookup_by_hull_number 精确匹配
            if data.get("found") is True:
                match_type = "exact"
                hull_number = data.get("hull_number", hull_number)
                description = data.get("description", description)

            # retrieve_by_description 语义匹配
            if "results" in data:
                results = data["results"]
                if results:
                    semantic_match_ids = [
                        r.get("hull_number", "") for r in results if r.get("hull_number")
                    ]
                    if match_type != "exact":
                        match_type = "semantic"

        return AgentResult(
            hull_number=hull_number,
            description=description,
            match_type=match_type,
            semantic_match_ids=semantic_match_ids,
            answer=answer,
        )

    def run_verbose(self, query: str) -> list[dict]:
        try:
            result = self._agent.invoke({"messages": [HumanMessage(content=query)]})
            trace = []
            for msg in result["messages"]:
                entry = {"type": msg.type, "content": msg.content}
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    entry["tool_calls"] = [
                        {"name": tc["name"], "args": tc["args"]}
                        for tc in msg.tool_calls
                    ]
                if hasattr(msg, "tool_call_id"):
                    entry["tool_call_id"] = msg.tool_call_id
                trace.append(entry)
            return trace
        except Exception as e:
            logger.exception("Agent 执行失败")
            return [{"type": "error", "content": f"查询执行失败: {e}"}]


_agent_instance: ShipHullAgent | None = None
_agent_config_hash: int = 0


def create_agent(config: dict[str, Any] | None = None) -> ShipHullAgent:
    global _agent_instance, _agent_config_hash
    config_hash = hash(str(config)) if config is not None else 0
    if _agent_instance is None or config_hash != _agent_config_hash:
        _agent_instance = ShipHullAgent(config)
        _agent_config_hash = config_hash
    return _agent_instance
