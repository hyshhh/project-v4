"""LangChain 工具定义 — 三步链路：识别 → 精确查找 → 语义检索"""

from __future__ import annotations

import json
import logging
from typing import Annotated

import cv2
import httpx
import numpy as np
from langchain_core.tools import tool

from database import ShipDatabase
from config import load_config

logger = logging.getLogger(__name__)


def _vlm_infer(image_b64: str, prompt_mode: str = "detailed") -> dict:
    """调用 VLM 进行弦号识别，返回 {hull_number, description}。"""
    config = load_config()
    llm_cfg = config.get("llm", {})

    api_url = f"{llm_cfg.get('base_url', 'http://localhost:7890/v1').rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {llm_cfg.get('api_key', 'abc123')}",
        "Content-Type": "application/json",
    }

    if prompt_mode == "brief":
        prompt = (
            "识别船体侧面文字编号。返回 JSON：\n"
            '{"hull_number": "弦号（无则空）", "description": "简短描述"}'
        )
    else:
        prompt = (
            "你是船只弦号识别专家。读取船体侧面的文字编号。\n"
            "不要评价图片质量。即使模糊，也必须尝试读取任何可见文字、数字。\n"
            "返回 JSON（不要其他文字）：\n"
            '{"hull_number": "弦号编号（无则空字符串）", '
            '"description": "船型+船体颜色+上层建筑颜色+特殊标志"}'
        )

    payload = {
        "model": llm_cfg.get("model", "Qwen/Qwen3-VL-4B-AWQ"),
        "temperature": llm_cfg.get("temperature", 0.0),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            }
        ],
    }

    resp = httpx.post(api_url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]

    # 解析 JSON
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

    import re
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        result = json.loads(match.group()) if match else {}

    return {
        "hull_number": str(result.get("hull_number") or "").strip(),
        "description": str(result.get("description") or "").strip(),
    }


def build_tools(db: ShipDatabase) -> list:
    """构建三步链路工具：recognize_ship → lookup_by_hull_number → retrieve_by_description。"""

    @tool
    def recognize_ship(
        image_base64: Annotated[str, "裁剪的船只图像 base64 编码字符串（JPEG）"],
    ) -> str:
        """
        第一步：对船只图像进行弦号识别。
        调用视觉大模型分析图像，返回识别到的弦号和船只描述。
        有弦号时接下来调 lookup_by_hull_number；无弦号时调 retrieve_by_description。
        """
        try:
            result = _vlm_infer(image_base64)
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            logger.exception("船只识别失败")
            return json.dumps({"error": str(e), "hull_number": "", "description": ""}, ensure_ascii=False)

    @tool
    def lookup_by_hull_number(
        hull_number: Annotated[str, "要查询的船弦号，例如 '0014'"],
    ) -> str:
        """
        第二步：通过弦号精确查找船只描述。
        recognize_ship 返回有弦号时调用此工具。found=true 则结束；found=false 进入第三步。
        """
        hull_number = hull_number.strip()
        desc = db.lookup(hull_number)
        if desc is not None:
            return json.dumps(
                {"found": True, "hull_number": hull_number, "description": desc},
                ensure_ascii=False,
            )
        return json.dumps({"found": False, "hull_number": hull_number}, ensure_ascii=False)

    @tool
    def retrieve_by_description(
        target_description: Annotated[str, "对目标船的外观文字描述，越详细越好"],
    ) -> str:
        """
        第三步：基于 FAISS 向量库的语义检索。
        当弦号查不到（第二步 found=false），或 recognize_ship 未识别到弦号时调用。
        输入对船只的外观描述，返回最匹配的结果。
        """
        try:
            results = db.semantic_search_filtered(target_description)
            if not results:
                raw = db.semantic_search(target_description)
                if raw:
                    return json.dumps(
                        {"note": "以下结果相似度较低，仅供参考", "results": raw},
                        ensure_ascii=False,
                    )
                return json.dumps({"error": "未找到匹配结果"}, ensure_ascii=False)

            return json.dumps({"results": results}, ensure_ascii=False)
        except Exception as e:
            logger.exception("语义检索失败")
            return json.dumps({"error": f"语义检索失败: {e}"}, ensure_ascii=False)

    return [recognize_ship, lookup_by_hull_number, retrieve_by_description]
