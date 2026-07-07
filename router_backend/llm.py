"""LiteLLM Proxy 的呼叫封裝：一般呼叫、串流、tool-calling 迴圈。"""
import json
import logging

import httpx

from config import LITELLM_API_KEY, LITELLM_BASE_URL, MAX_ANSWER_TOKENS
from tools import run_search_tool

logger = logging.getLogger("router.llm")

_AUTH_HEADERS = {"Authorization": f"Bearer {LITELLM_API_KEY}"}
STREAM_TIMEOUT = httpx.Timeout(connect=30, read=300, write=30, pool=10)


async def call_litellm(client: httpx.AsyncClient, model_alias: str, messages: list,
                       max_tokens: int = MAX_ANSWER_TOKENS) -> tuple[str, dict]:
    """非串流呼叫，回傳 (content, usage)。"""
    resp = await client.post(
        f"{LITELLM_BASE_URL}/v1/chat/completions",
        headers=_AUTH_HEADERS,
        json={"model": model_alias, "messages": messages, "max_tokens": max_tokens},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    if content is None:
        content = data["choices"][0]["message"].get("reasoning_content") or ""
    usage = data.get("usage", {})
    return content, {"input_tokens": usage.get("prompt_tokens", 0),
                     "output_tokens": usage.get("completion_tokens", 0)}


def stream_litellm(client: httpx.AsyncClient, model_alias: str, messages: list,
                   max_tokens: int = MAX_ANSWER_TOKENS):
    """回傳可 async with 的串流 request（SSE），由呼叫端解析 chunk。"""
    return client.stream(
        "POST",
        f"{LITELLM_BASE_URL}/v1/chat/completions",
        headers=_AUTH_HEADERS,
        json={"model": model_alias, "messages": messages, "max_tokens": max_tokens,
              "stream": True, "stream_options": {"include_usage": True}},
        timeout=STREAM_TIMEOUT,
    )


async def run_tools(client: httpx.AsyncClient, model_alias: str, messages: list,
                    tools: list, max_tokens: int = MAX_ANSWER_TOKENS, max_iters: int = 5):
    """驅動 tool-calling 迴圈。

    以 async generator 形式產出事件：
      {"type": "tool_running", "name": <工具名>}  — 某個工具即將執行
      {"type": "final", "content": <答案>, "usage": {...}}  — 最終回覆（usage 為各回合累加）
    """
    total_in = total_out = 0
    for _ in range(max_iters):
        resp = await client.post(
            f"{LITELLM_BASE_URL}/v1/chat/completions",
            headers=_AUTH_HEADERS,
            json={"model": model_alias, "messages": messages,
                  "max_tokens": max_tokens, "tools": tools},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        msg = data["choices"][0]["message"]
        usage = data.get("usage", {})
        total_in  += usage.get("prompt_tokens", 0)
        total_out += usage.get("completion_tokens", 0)

        if msg.get("tool_calls"):
            messages.append(msg)
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                yield {"type": "tool_running", "name": fn.get("name", "")}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except Exception:
                    args = {}
                try:
                    result = await run_search_tool(fn.get("name", ""), args)
                except Exception:
                    logger.exception("搜尋工具執行失敗：%s", fn.get("name", ""))
                    result = "（工具呼叫失敗，請稍後再試）"
                messages.append({"role": "tool", "tool_call_id": tc.get("id"), "content": result})
            continue

        content = msg.get("content")
        if content is None:
            content = msg.get("reasoning_content") or ""
        yield {"type": "final", "content": content,
               "usage": {"input_tokens": total_in, "output_tokens": total_out}}
        return

    yield {"type": "final",
           "content": "（搜尋工具呼叫次數過多，請換個方式再問一次）",
           "usage": {"input_tokens": total_in, "output_tokens": total_out}}
