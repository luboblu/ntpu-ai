"""Chat 串流編排：judge 評分 → 選模型 → 產生回答 → 儲存歷史與用量。

app.py 只負責 HTTP 端點與身分驗證，實際對話流程集中在這裡，
每個步驟拆成獨立函式，方便閱讀與單獨測試。
"""
import asyncio
import json
import logging
import time

import httpx

import memory
import store
from attachments import build_user_content
from config import HISTORY_LIMIT, JUDGE_MODEL_ALIAS, TAVILY_KEY
from llm import run_tools, stream_litellm
from prompts import build_system_prompt
from routing import (
    default_model_for_route,
    get_routing_config,
    model_classify,
    pick_tool_capable,
    select_route,
)
from schemas import ChatRequest
from tools import build_search_tools

logger = logging.getLogger("router.chat")

_EMPTY_USAGE = {"input_tokens": 0, "output_tokens": 0}


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _chunk_text(text: str, size: int = 24):
    for i in range(0, len(text), size):
        yield text[i:i + size]


async def _decide_model(client: httpx.AsyncClient, req: ChatRequest,
                        llm_history: list, config: dict, is_admin: bool):
    """Judge 評估難度後選定模型；啟用工具時避開不支援 function calling 的模型。

    回傳 (route, model_alias, judge, judge_elapsed_ms, tools)。
    """
    t0 = time.time()
    judge = await model_classify(client, req.message, llm_history)
    judge_elapsed_ms = int((time.time() - t0) * 1000)

    force = config.get("force_model") if is_admin else None
    route, model_alias = select_route(config, judge, force)

    tools = build_search_tools(req) if TAVILY_KEY else []
    if tools:
        if route == "tiny":
            route, model_alias = "small", default_model_for_route("small")
        route, model_alias = pick_tool_capable(route, model_alias)
    judge["route"] = route
    judge["model"] = model_alias
    return route, model_alias, judge, judge_elapsed_ms, tools


async def _build_answer_messages(uid: str, req: ChatRequest, llm_history: list,
                                 has_tools: bool) -> list:
    profile, user_content = await asyncio.gather(
        store.get_user_profile_fields(uid),
        build_user_content(req.message, req.file_gcs_path, req.file_mime_type),
    )
    messages = [{"role": "system", "content": build_system_prompt(
        profile.get("system_prompt", ""), has_tools=has_tools,
        long_term_memory=profile.get("memory", ""))}]
    return messages + llm_history[-HISTORY_LIMIT:] + [{"role": "user", "content": user_content}]


async def _answer_events(client: httpx.AsyncClient, model_alias: str,
                         messages: list, tools: list):
    """統一「工具路徑」與「純串流路徑」，產出相同格式的事件：

      {"type": "search"}                        — 搜尋工具執行中
      {"type": "token", "content": <str>}       — 增量文字
      {"type": "final", "content", "usage"}     — 完整答案與 token 用量（必為最後一筆）
    """
    if tools:
        # 工具路徑：先讓模型決定是否呼叫搜尋，解析完再把最終答案逐段送出
        full_content, usage = "", dict(_EMPTY_USAGE)
        async for ev in run_tools(client, model_alias, messages, tools):
            if ev["type"] == "tool_running":
                yield {"type": "search"}
            elif ev["type"] == "final":
                full_content, usage = ev["content"], ev["usage"]
        for piece in _chunk_text(full_content):
            yield {"type": "token", "content": piece}
        yield {"type": "final", "content": full_content, "usage": usage}
        return

    full_content, reasoning_fallback, usage = "", "", dict(_EMPTY_USAGE)
    async with stream_litellm(client, model_alias, messages) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            raw = line[6:].strip()
            if raw == "[DONE]":
                break
            try:
                chunk = json.loads(raw)
            except Exception:
                continue
            # 最後一個 chunk 含 usage
            if chunk.get("usage"):
                u = chunk["usage"]
                usage = {"input_tokens": u.get("prompt_tokens", 0),
                         "output_tokens": u.get("completion_tokens", 0)}
            if chunk.get("choices"):
                delta = chunk["choices"][0].get("delta", {})
                # 推理模型（Nemotron Omni、gpt-oss 等）在正式作答前會先吐一段
                # reasoning_content（內部思考過程，通常是英文），不能當成答案
                # 顯示；只累積 content，reasoning_content 另外存著備用
                content = delta.get("content") or ""
                if content:
                    full_content += content
                    yield {"type": "token", "content": content}
                else:
                    reasoning_fallback += delta.get("reasoning_content") or ""
    if not full_content and reasoning_fallback:
        # 極端情況：整段回應都在推理階段就被截斷（例如 max_tokens 用完），
        # 完全沒有輸出正式答案時，退而求其次顯示推理內容，避免整則訊息空白
        full_content = reasoning_fallback
        for piece in _chunk_text(full_content):
            yield {"type": "token", "content": piece}
    yield {"type": "final", "content": full_content, "usage": usage}


async def _persist(uid: str, email: str, req: ChatRequest, history: list,
                   full_content: str, route: str, model_alias: str, judge: dict,
                   judge_usage: dict, answer_usage: dict):
    """寫入對話歷史；已登入使用者另外在背景記錄 token 用量、視情況壓縮長期記憶。"""
    user_entry: dict = {"role": "user", "content": req.message}
    if req.file_name:
        user_entry["_file_name"]      = req.file_name
        user_entry["_file_mime_type"] = req.file_mime_type or ""
        user_entry["_file_gcs_path"]  = req.file_gcs_path or ""
    new_history = history + [
        user_entry,
        {"role": "assistant", "content": full_content,
         "_route": route, "_model": model_alias,
         "_score": judge["score"], "_reason": judge.get("reason", "")},
    ]
    await store.save_history(uid, req.session_id, new_history)

    if uid != "anonymous":
        store.log_usage_background(
            uid, email, req.session_id, route, judge["score"], model_alias,
            judge_usage["input_tokens"] + answer_usage["input_tokens"],
            judge_usage["output_tokens"] + answer_usage["output_tokens"],
            JUDGE_MODEL_ALIAS, judge_usage["input_tokens"], judge_usage["output_tokens"],
            answer_usage["input_tokens"], answer_usage["output_tokens"],
        )
        memory.maybe_compress_background(uid, req.session_id, new_history)


async def stream_chat(req: ChatRequest, decoded: dict):
    """/chat/stream 的 SSE 事件產生器。

    任何一步失敗都只回傳一般化錯誤訊息；詳細錯誤留在 server log，
    避免把內部資訊（URL、金鑰、堆疊）洩漏給前端。
    """
    uid   = decoded.get("uid", "anonymous")
    email = decoded.get("email", "")
    try:
        history = await store.get_history(uid, req.session_id)
        llm_history = [{"role": m["role"], "content": m["content"]} for m in history]
        config = await get_routing_config()

        async with httpx.AsyncClient() as client:
            route, model_alias, judge, judge_elapsed_ms, tools = await _decide_model(
                client, req, llm_history, config, bool(decoded.get("admin")))

            # 先送 judge metadata（usage 只記帳，不傳給前端）
            judge_usage = judge.pop("_usage", dict(_EMPTY_USAGE))
            yield sse({"type": "judge", "route": route, "model": model_alias,
                       "judge": judge, "judge_elapsed_ms": judge_elapsed_ms})

            answer_messages = await _build_answer_messages(uid, req, llm_history, bool(tools))
            t1 = time.time()
            full_content, answer_usage = "", dict(_EMPTY_USAGE)
            async for ev in _answer_events(client, model_alias, answer_messages, tools):
                if ev["type"] == "final":
                    full_content, answer_usage = ev["content"], ev["usage"]
                else:
                    yield sse(ev)

            yield sse({"type": "done", "answer_elapsed_ms": int((time.time() - t1) * 1000)})
            yield "data: [DONE]\n\n"

            await _persist(uid, email, req, history, full_content,
                           route, model_alias, judge, judge_usage, answer_usage)
    except Exception:
        logger.exception("chat_stream 失敗 uid=%s session=%s", uid, req.session_id)
        yield sse({"type": "error", "message": "系統暫時無法回應，請稍後再試"})
