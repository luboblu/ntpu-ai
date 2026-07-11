"""LiteLLM Proxy 的呼叫封裝：一般呼叫、串流、tool-calling 迴圈。

雲端模型走 LITELLM_BASE_URL；地端模型（local-* alias）走獨立的
LOCAL_LLM_BASE_URL（通常是經 Cloudflare Tunnel 對外的自架 LiteLLM），
並額外帶 Cloudflare Access Service Token 通過閘道驗證。
"""
import json
import logging

import httpx

from config import (
    CF_ACCESS_CLIENT_ID,
    CF_ACCESS_CLIENT_SECRET,
    LITELLM_API_KEY,
    LITELLM_BASE_URL,
    LOCAL_LLM_API_KEY,
    LOCAL_LLM_BASE_URL,
    MAX_ANSWER_TOKENS,
    is_local_alias,
)
from tools import run_search_tool

logger = logging.getLogger("router.llm")

_CLOUD_HEADERS = {"Authorization": f"Bearer {LITELLM_API_KEY}"}
_CLOUD_URL = f"{LITELLM_BASE_URL}/v1/chat/completions"

_LOCAL_HEADERS = {"Authorization": f"Bearer {LOCAL_LLM_API_KEY}"}
if CF_ACCESS_CLIENT_ID and CF_ACCESS_CLIENT_SECRET:
    _LOCAL_HEADERS["CF-Access-Client-Id"] = CF_ACCESS_CLIENT_ID
    _LOCAL_HEADERS["CF-Access-Client-Secret"] = CF_ACCESS_CLIENT_SECRET
_LOCAL_URL = f"{LOCAL_LLM_BASE_URL}/chat/completions" if LOCAL_LLM_BASE_URL else _CLOUD_URL

STREAM_TIMEOUT = httpx.Timeout(connect=30, read=300, write=30, pool=10)


def _endpoint_for(model_alias: str) -> tuple[str, dict]:
    """依 alias 決定要打哪個 LiteLLM 端點與 headers。"""
    if LOCAL_LLM_BASE_URL and is_local_alias(model_alias):
        return _LOCAL_URL, _LOCAL_HEADERS
    return _CLOUD_URL, _CLOUD_HEADERS


async def _chat_completion(client: httpx.AsyncClient, payload: dict) -> dict:
    """非串流 chat completion，回傳解析後的 JSON。"""
    url, headers = _endpoint_for(payload.get("model", ""))
    resp = await client.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def _usage_of(data: dict) -> tuple[int, int]:
    usage = data.get("usage", {})
    return usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


def _message_text(msg: dict) -> str:
    """取出訊息文字；部分推理模型把輸出放在 reasoning_content。"""
    content = msg.get("content")
    if content is None:
        content = msg.get("reasoning_content") or ""
    return content


async def call_litellm(client: httpx.AsyncClient, model_alias: str, messages: list,
                       max_tokens: int = MAX_ANSWER_TOKENS,
                       json_mode: bool = False) -> tuple[str, dict]:
    """非串流呼叫，回傳 (content, usage)。

    json_mode=True 時啟用供應商原生的 JSON 輸出模式（response_format），
    確保回應是合法 JSON；不支援的供應商由 litellm 的 drop_params 直接忽略。
    """
    payload = {"model": model_alias, "messages": messages, "max_tokens": max_tokens}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    data = await _chat_completion(client, payload)
    input_tokens, output_tokens = _usage_of(data)
    return _message_text(data["choices"][0]["message"]), {
        "input_tokens": input_tokens, "output_tokens": output_tokens,
    }


def stream_litellm(client: httpx.AsyncClient, model_alias: str, messages: list,
                   max_tokens: int = MAX_ANSWER_TOKENS):
    """回傳可 async with 的串流 request（SSE），由呼叫端解析 chunk。"""
    url, headers = _endpoint_for(model_alias)
    return client.stream(
        "POST",
        url,
        headers=headers,
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
        data = await _chat_completion(client, {
            "model": model_alias, "messages": messages,
            "max_tokens": max_tokens, "tools": tools,
        })
        msg = data["choices"][0]["message"]
        input_tokens, output_tokens = _usage_of(data)
        total_in  += input_tokens
        total_out += output_tokens

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

        yield {"type": "final", "content": _message_text(msg),
               "usage": {"input_tokens": total_in, "output_tokens": total_out}}
        return

    yield {"type": "final",
           "content": "（搜尋工具呼叫次數過多，請換個方式再問一次）",
           "usage": {"input_tokens": total_in, "output_tokens": total_out}}
