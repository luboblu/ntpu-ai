"""路由核心：難度判斷（judge）與級距／模型選擇。"""
import json
import logging
import re
import time
from typing import Optional

import httpx

import store
from config import (
    HISTORY_LIMIT,
    JUDGE_MODEL_ALIAS,
    MODEL_CANDIDATES,
    MODEL_NOTES,
    MODEL_TO_ROUTE,
    NO_TOOL_MODELS,
    is_local_alias,
)
from llm import call_litellm
from prompts import JUDGE_SYSTEM_PROMPT

logger = logging.getLogger("router.routing")

# ------------------------------------------------------------------
# Routing config（Firestore 讀取，30 秒快取）
# ------------------------------------------------------------------
DEFAULT_ROUTING_CONFIG = {
    "threshold_tiny": None,
    "threshold_medium": 4.0,
    "threshold_large": 7.0,
    "force_model": None,
    "prefer_local": False,
}
_routing_cache: dict = dict(DEFAULT_ROUTING_CONFIG)
_routing_cache_ts: float = 0.0


async def get_routing_config() -> dict:
    global _routing_cache, _routing_cache_ts
    if not store.firebase_ready:
        return _routing_cache
    if time.time() - _routing_cache_ts > 30:
        _routing_cache = await store.fetch_routing_config(DEFAULT_ROUTING_CONFIG)
        _routing_cache_ts = time.time()
    return _routing_cache


def invalidate_routing_cache():
    global _routing_cache_ts
    _routing_cache_ts = 0


async def save_routing_config(data: dict):
    """儲存路由設定；未接 Firebase（本機開發）時直接寫入記憶體快取，
    否則管理員面板的設定永遠存不進去（get_routing_config 在無 Firebase
    時只讀記憶體，不會去讀 Firestore）。"""
    global _routing_cache
    if not store.firebase_ready:
        _routing_cache = dict(data)
        return
    await store.save_routing_config(data)
    invalidate_routing_cache()


# ------------------------------------------------------------------
# 級距 / 模型選擇
# ------------------------------------------------------------------
def _route_from_score(config: dict, score: float) -> str:
    t_tiny = config.get("threshold_tiny")
    t_medium = float(config.get("threshold_medium") or 4.0)
    t_large = float(config.get("threshold_large") or 7.0)

    if t_tiny is not None and score < float(t_tiny):
        return "tiny"
    if score >= t_large:
        return "large"
    if score >= t_medium:
        return "medium"
    return "small"


def default_model_for_route(route: str) -> str:
    candidates = MODEL_CANDIDATES.get(route) or MODEL_CANDIDATES["small"]
    return candidates[0]


def pick_tool_capable(route: str, model_alias: str) -> tuple[str, str]:
    """啟用工具時，若選到不支援 function calling 的模型，改挑同級距其他可用模型；
    若整個級距都不支援，退到 medium 的預設模型。"""
    if model_alias not in NO_TOOL_MODELS:
        return route, model_alias
    for alias in MODEL_CANDIDATES.get(route, []):
        if alias not in NO_TOOL_MODELS:
            return route, alias
    return "medium", default_model_for_route("medium")


def _model_options_text() -> str:
    lines = []
    for route in ("small", "medium", "large"):
        candidates = MODEL_CANDIDATES.get(route, [])
        if candidates:
            lines.append(f"{route}:")
            for alias in candidates:
                lines.append(f"- {alias}: {MODEL_NOTES.get(alias, '可用回答模型')}")
    if MODEL_CANDIDATES.get("tiny"):
        lines.append("tiny:")
        for alias in MODEL_CANDIDATES["tiny"]:
            lines.append(f"- {alias}: 開源或自訂小模型；僅在低風險、低複雜度且不需工具時使用")
    return "\n".join(lines)


def select_route(config: dict, judge: dict, force: Optional[str]) -> tuple[str, str]:
    """依 judge 結果、管理員強制設定與 prefer_local 選出 (route, model_alias)。"""
    score = float(judge.get("score", 5.0))
    requested_route = judge.get("route")
    requested_model = judge.get("model")

    if force in MODEL_TO_ROUTE:
        return MODEL_TO_ROUTE[force], force

    if force in ("small", "medium", "large", "tiny"):
        route = force
    elif requested_route in MODEL_CANDIDATES:
        route = requested_route
    else:
        route = _route_from_score(config, score)

    candidates = MODEL_CANDIDATES.get(route) or MODEL_CANDIDATES["small"]

    # 地端優先：該級距有 local-* 候選時，改在地端候選中挑（資料不出機房）。
    # 之後接 Ollama / vLLM 時，把 alias 加進 LOCAL_*_MODEL_ALIASES 並在管理面板開啟 prefer_local 即可。
    if config.get("prefer_local"):
        local_candidates = [a for a in candidates if is_local_alias(a)]
        if local_candidates:
            candidates = local_candidates

    model_alias = requested_model if requested_model in candidates else candidates[0]
    return route, model_alias


# ------------------------------------------------------------------
# Judge：AI 判斷式難度評分
# ------------------------------------------------------------------
async def model_classify(client: httpx.AsyncClient, text: str, history: list) -> dict:
    if history:
        lines = [f"{'使用者' if m['role'] == 'user' else 'AI'}：{store.content_str(m['content'])[:400]}"
                 for m in history[-HISTORY_LIMIT:]]
        context_block = "對話歷史（供參考）：\n\"\"\"\n" + "\n".join(lines) + "\n\"\"\"\n\n"
    else:
        context_block = ""
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": f"{context_block}可選模型：\n{_model_options_text()}\n\n"
                                    f"請評估以下最新訊息的難度並選擇級距與模型：\n```\n{text}\n```"},
    ]
    raw, judge_usage = await call_litellm(client, JUDGE_MODEL_ALIAS, messages,
                                          max_tokens=1024, json_mode=True)

    parsed = _parse_judge_json(raw)
    if parsed:
        score = max(0.0, min(10.0, float(parsed.get("score", 5))))
        route = parsed.get("route")
        model = parsed.get("model")
        if route not in MODEL_CANDIDATES:
            route = None
        if route and model not in MODEL_CANDIDATES.get(route, []):
            model = None
        return {
            "score": score,
            "route": route,
            "model": model,
            "reason": parsed.get("reason", ""),
            "normalized": score / 10.0,
            "_usage": judge_usage,
        }

    logger.warning("[judge] parse failed, raw=%r", raw[:300])
    return {"score": 5.0, "route": None, "model": None, "reason": "", "normalized": 0.5, "_usage": judge_usage}


def _parse_judge_json(raw: str) -> Optional[dict]:
    """先嘗試直接解析，再剝除 code fence，最後用 regex 抓出 {...}。"""
    for candidate in [raw, re.sub(r"```(?:json)?|```", "", raw).strip()]:
        try:
            return json.loads(candidate)
        except Exception:
            pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)  # greedy: first { to last }
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return None
