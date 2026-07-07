"""AI Router Backend — FastAPI 組裝與 API 端點。

模組分工：
  config.py      環境變數與模型設定
  schemas.py     API 請求資料模型
  security.py    Token 驗證、速率限制、安全標頭
  store.py       Firebase / Firestore / GCS 存取
  prompts.py     系統提示與 judge 提示
  routing.py     難度判斷與級距／模型選擇
  llm.py         LiteLLM 呼叫（一般／串流／tool-calling）
  tools.py       搜尋工具（Tavily、北大官網）
  attachments.py 附件解析
"""
import asyncio
import datetime
import json
import logging
import os
import re
import time
import uuid
from typing import Optional

import httpx
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse

import store
from attachments import OFFICE_MIMES, build_user_content, extract_office_text, is_text_mime
from config import (
    CHAT_RATE_LIMIT_PER_MINUTE,
    CORS_ORIGINS,
    HISTORY_LIMIT,
    JUDGE_MODEL_ALIAS,
    OPENAI_API_KEY,
    TAVILY_KEY,
    TRANSCRIBE_MAX_BYTES,
    UPLOAD_MAX_BYTES,
    UPLOAD_RATE_LIMIT_PER_MINUTE,
)
from llm import run_tools, stream_litellm
from prompts import build_system_prompt
from routing import (
    default_model_for_route,
    get_routing_config,
    invalidate_routing_cache,
    model_classify,
    pick_tool_capable,
    select_route,
)
from schemas import ChatRequest, RoutingConfig, UserProfileRequest
from security import (
    RateLimiter,
    SecurityHeadersMiddleware,
    decode_token,
    require_admin,
    verify_token,
)
from tools import build_search_tools

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("router.app")

app = FastAPI(title="AI Router Backend")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(SecurityHeadersMiddleware)

_chat_limiter   = RateLimiter(CHAT_RATE_LIMIT_PER_MINUTE)
_upload_limiter = RateLimiter(UPLOAD_RATE_LIMIT_PER_MINUTE)


# ------------------------------------------------------------------
# 首頁（前端唯一正本是 frontend/index.html；Docker build 會 COPY 到 app.py 旁邊）
# ------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_INDEX_CANDIDATES = [
    os.path.join(_HERE, "index.html"),
    os.path.join(_HERE, "..", "frontend", "index.html"),
]


def _index_html_path() -> str:
    for p in _INDEX_CANDIDATES:
        if os.path.exists(p):
            return p
    return _INDEX_CANDIDATES[0]


@app.get("/")
async def index():
    return FileResponse(_index_html_path())


@app.get("/health")
async def health():
    return {"status": "ok"}


# ------------------------------------------------------------------
# Chat（SSE 串流）
# ------------------------------------------------------------------
def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _chunk_text(text: str, size: int = 24):
    for i in range(0, len(text), size):
        yield text[i:i + size]


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, authorization: Optional[str] = Header(None)):
    decoded = await decode_token(authorization)
    uid   = decoded.get("uid", "anonymous")
    email = decoded.get("email", "")
    _chat_limiter.check(uid)

    history = await store.get_history(uid, req.session_id)
    llm_history = [{"role": m["role"], "content": m["content"]} for m in history]
    config = await get_routing_config()

    async def generate():
        try:
            async with httpx.AsyncClient() as client:
                # 1. Judge（不串流）
                t0 = time.time()
                judge = await model_classify(client, req.message, llm_history)
                judge_elapsed_ms = int((time.time() - t0) * 1000)

                # 2. 路由
                force = config.get("force_model") if decoded.get("admin") else None
                route, model_alias = select_route(config, judge, force)
                judge["route"] = route
                judge["model"] = model_alias

                tools = build_search_tools(req) if TAVILY_KEY else []
                # 工具啟用時避免路由到不支援 function calling 的模型
                if tools and route == "tiny":
                    route, model_alias = "small", default_model_for_route("small")
                if tools:
                    route, model_alias = pick_tool_capable(route, model_alias)
                    judge["route"] = route
                    judge["model"] = model_alias

                # 3. 先送 judge metadata（順便取出 usage，不傳給前端）
                judge_usage = judge.pop("_usage", {"input_tokens": 0, "output_tokens": 0})
                yield _sse({"type": "judge", "route": route, "model": model_alias,
                            "judge": judge, "judge_elapsed_ms": judge_elapsed_ms})

                # 4. 準備訊息（系統提示；搜尋改由 LLM 透過工具自行觸發）
                sys_prompt = await store.get_user_system_prompt(uid)
                user_content = await build_user_content(req.message, req.file_gcs_path, req.file_mime_type)
                answer_messages = [{"role": "system", "content": build_system_prompt(sys_prompt, has_tools=bool(tools))}]
                answer_messages += llm_history[-HISTORY_LIMIT:] + [{"role": "user", "content": user_content}]
                t1 = time.time()
                full_content = ""
                answer_input_tokens  = 0
                answer_output_tokens = 0

                if tools:
                    # 工具路徑：先讓模型決定是否呼叫搜尋，解析完再把最終答案逐段送出
                    async for ev in run_tools(client, model_alias, answer_messages, tools):
                        if ev["type"] == "tool_running":
                            yield _sse({"type": "search"})
                        elif ev["type"] == "final":
                            full_content = ev["content"]
                            answer_input_tokens  = ev["usage"]["input_tokens"]
                            answer_output_tokens = ev["usage"]["output_tokens"]
                    for piece in _chunk_text(full_content):
                        yield _sse({"type": "token", "content": piece})
                else:
                    async with stream_litellm(client, model_alias, answer_messages) as resp:
                        resp.raise_for_status()
                        async for line in resp.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            raw = line[6:].strip()
                            if raw == "[DONE]":
                                break
                            try:
                                chunk = json.loads(raw)
                                # 最後一個 chunk 含 usage
                                if chunk.get("usage"):
                                    u = chunk["usage"]
                                    answer_input_tokens  = u.get("prompt_tokens", 0)
                                    answer_output_tokens = u.get("completion_tokens", 0)
                                if chunk.get("choices"):
                                    delta = chunk["choices"][0].get("delta", {})
                                    content = delta.get("content") or delta.get("reasoning_content") or ""
                                    if content:
                                        full_content += content
                                        yield _sse({"type": "token", "content": content})
                            except Exception:
                                pass

                answer_elapsed_ms = int((time.time() - t1) * 1000)
                total_input  = judge_usage["input_tokens"]  + answer_input_tokens
                total_output = judge_usage["output_tokens"] + answer_output_tokens

                yield _sse({"type": "done", "answer_elapsed_ms": answer_elapsed_ms})
                yield "data: [DONE]\n\n"

                # 5. 儲存歷史
                user_entry: dict = {"role": "user", "content": req.message}
                if req.file_name:
                    user_entry["_file_name"]      = req.file_name
                    user_entry["_file_mime_type"] = req.file_mime_type or ""
                    user_entry["_file_gcs_path"]  = req.file_gcs_path or ""
                new_history = history + [
                    user_entry,
                    {"role": "assistant", "content": full_content,
                     "_route": route, "_score": judge["score"], "_reason": judge.get("reason", "")},
                ]
                await store.save_history(uid, req.session_id, new_history)

                if uid != "anonymous":
                    store.log_usage_background(
                        uid, email, req.session_id, route, judge["score"], model_alias,
                        total_input, total_output,
                        JUDGE_MODEL_ALIAS, judge_usage["input_tokens"], judge_usage["output_tokens"],
                        answer_input_tokens, answer_output_tokens,
                    )

        except Exception:
            # 詳細錯誤只留在 server log，避免把內部資訊（URL、金鑰、堆疊）洩漏給前端
            logger.exception("chat_stream 失敗 uid=%s session=%s", uid, req.session_id)
            yield _sse({"type": "error", "message": "系統暫時無法回應，請稍後再試"})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ------------------------------------------------------------------
# 對話管理
# ------------------------------------------------------------------
@app.post("/reset")
async def reset(session_id: str, authorization: Optional[str] = Header(None)):
    uid = await verify_token(authorization)
    await store.delete_session(uid, session_id)
    return {"ok": True}


@app.get("/conversations")
async def list_conversations(authorization: Optional[str] = Header(None)):
    uid = await verify_token(authorization)
    if uid == "anonymous":
        return []
    return await store.list_sessions(uid)


@app.get("/conversations/{session_id}")
async def get_conversation(session_id: str, authorization: Optional[str] = Header(None)):
    uid = await verify_token(authorization)
    if uid == "anonymous":
        raise HTTPException(status_code=401)
    data = await store.get_session_data(uid, session_id)
    return {"session_id": session_id, "history": data.get("history", []), "title": data.get("title", "對話")}


@app.post("/conversations/{session_id}/share")
async def share_conversation(session_id: str, authorization: Optional[str] = Header(None)):
    decoded = await decode_token(authorization)
    uid = decoded.get("uid", "anonymous")
    if uid == "anonymous":
        raise HTTPException(status_code=401)
    history = await store.get_history(uid, session_id)
    if not history:
        raise HTTPException(status_code=404, detail="對話不存在")
    title = next((store.content_str(m["content"])[:40] for m in history if m["role"] == "user"), "對話")
    public_history = [
        {"role": m["role"], "content": m["content"],
         "_route": m.get("_route"), "_score": m.get("_score"), "_reason": m.get("_reason"),
         "_file_name": m.get("_file_name")}
        for m in history
    ]
    share_id = await store.create_share(uid, session_id, public_history, title)
    return {"share_id": share_id}


@app.get("/share/{share_id}")
async def get_shared_conversation(share_id: str):
    data = await store.get_share(share_id)
    if not data:
        raise HTTPException(status_code=404, detail="分享連結不存在或已失效")
    return {"history": data.get("history", []), "title": data.get("title", "對話")}


# ------------------------------------------------------------------
# 使用者
# ------------------------------------------------------------------
@app.get("/me")
async def get_me(authorization: Optional[str] = Header(None)):
    """Verify token and return basic user info (also enforces domain check)."""
    decoded = await decode_token(authorization)
    return {
        "uid":   decoded.get("uid", "anonymous"),
        "email": decoded.get("email", ""),
        "admin": bool(decoded.get("admin")),
    }


@app.get("/user/profile")
async def get_profile(authorization: Optional[str] = Header(None)):
    decoded = await decode_token(authorization)
    uid = decoded.get("uid", "anonymous")
    if uid == "anonymous":
        return {"system_prompt": ""}
    profile = await store.get_user_profile(uid)
    return {"system_prompt": profile.get("system_prompt", "")}


@app.post("/user/profile")
async def set_profile(data: UserProfileRequest, authorization: Optional[str] = Header(None)):
    decoded = await decode_token(authorization)
    uid = decoded.get("uid", "anonymous")
    if uid == "anonymous":
        raise HTTPException(status_code=401)
    await store.set_user_profile(uid, {"system_prompt": data.system_prompt or ""})
    return {"ok": True}


# ------------------------------------------------------------------
# 檔案上傳 / 預覽 / 語音轉文字
# ------------------------------------------------------------------
_SAFE_EXT_RE = re.compile(r"^\.[A-Za-z0-9]{1,10}$")


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(None),
):
    decoded = await decode_token(authorization)
    uid = decoded.get("uid", "anonymous")
    _upload_limiter.check(uid)
    if not store.gcs_ready:
        raise HTTPException(status_code=503, detail="Storage not configured")
    content = await file.read()
    if len(content) > UPLOAD_MAX_BYTES:
        raise HTTPException(status_code=413, detail="檔案超過 20MB 上限")
    # 副檔名只允許英數字，避免把奇怪字元寫進 GCS 路徑
    ext = os.path.splitext(file.filename or "")[1].lower()
    if not _SAFE_EXT_RE.match(ext):
        ext = ""
    gcs_path = f"uploads/{uid}/{int(time.time())}_{uuid.uuid4().hex[:8]}{ext}"
    await store.gcs_upload(gcs_path, content, file.content_type)
    return {
        "gcs_path":  gcs_path,
        "filename":  file.filename,
        "mime_type": file.content_type,
        "size":      len(content),
    }


@app.get("/file-preview")
async def file_preview(path: str, authorization: Optional[str] = Header(None)):
    decoded = await decode_token(authorization)
    uid = decoded.get("uid", "anonymous")
    if not store.gcs_ready:
        raise HTTPException(status_code=503)
    # 只允許讀自己的上傳目錄，避免跨使用者存取
    if not path.startswith(f"uploads/{uid}/"):
        raise HTTPException(status_code=403)
    try:
        file_bytes, mime = await store.gcs_download(path)
    except Exception:
        logger.exception("file_preview 下載失敗 path=%s", path)
        raise HTTPException(status_code=500, detail="無法讀取檔案")
    if mime in OFFICE_MIMES:
        text = await asyncio.to_thread(extract_office_text, file_bytes, mime)
        return Response(content=text or "（無法取出文字內容）", media_type="text/plain; charset=utf-8")
    if is_text_mime(mime):
        return Response(content=file_bytes, media_type=mime)
    return Response(content=file_bytes, media_type=mime)


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    lang: str = Form("zh"),
    authorization: Optional[str] = Header(None),
):
    decoded = await decode_token(authorization)
    _upload_limiter.check(decoded.get("uid", "anonymous"))
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="語音轉文字功能未啟用")
    audio_bytes = await file.read()
    if len(audio_bytes) > TRANSCRIBE_MAX_BYTES:
        raise HTTPException(status_code=413, detail="音檔過大（上限 25 MB）")
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    lang_code = "zh" if lang.startswith("zh") else "en"
    transcript = await client.audio.transcriptions.create(
        model="whisper-1",
        file=(file.filename or "audio.webm", audio_bytes, file.content_type or "audio/webm"),
        language=lang_code,
    )
    return {"text": transcript.text}


# ------------------------------------------------------------------
# Admin API
# ------------------------------------------------------------------
@app.post("/admin/setup")
async def admin_setup(authorization: Optional[str] = Header(None)):
    """第一次使用：若系統中無管理員，授予請求者管理員權限。
    先用 Auth 掃描擋掉既有管理員，再用 Firestore 交易防止兩個「首次」請求同時搶權（TOCTOU 防護）。"""
    if not store.firebase_ready:
        raise HTTPException(status_code=503)
    decoded = await decode_token(authorization)
    uid = decoded.get("uid")
    if not uid or uid == "anonymous":
        raise HTTPException(status_code=401)

    # 既有部署可能在此 flag 機制上線前就已有管理員，需先擋掉
    if await store.any_admin_exists():
        raise HTTPException(status_code=403, detail="管理員已存在，請聯絡現有管理員授權")

    if not await store.try_claim_first_admin(uid):
        raise HTTPException(status_code=403, detail="管理員已存在，請聯絡現有管理員授權")

    await store.set_admin_claim(uid, True)
    return {"ok": True}


@app.get("/admin/config")
async def admin_get_config(authorization: Optional[str] = Header(None)):
    await require_admin(authorization)
    return await get_routing_config()


@app.post("/admin/config")
async def admin_set_config(config: RoutingConfig, authorization: Optional[str] = Header(None)):
    await require_admin(authorization)
    await store.save_routing_config(config.model_dump())
    invalidate_routing_cache()
    return {"ok": True}


def _parse_iso_dt(s: Optional[str]) -> Optional[datetime.datetime]:
    if not s:
        return None
    try:
        dt = datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None
    # 一律轉成 UTC aware，才能跟 Firestore 的 SERVER_TIMESTAMP 比較
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone(datetime.timezone.utc)


@app.get("/admin/stats")
async def admin_stats(
    start: Optional[str] = None,
    end: Optional[str] = None,
    authorization: Optional[str] = Header(None),
):
    await require_admin(authorization)
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    start_dt = _parse_iso_dt(start) or (now - datetime.timedelta(days=30))
    end_dt = _parse_iso_dt(end) or now
    return await store.get_stats(start_dt, end_dt)


@app.get("/admin/users")
async def admin_list_users(authorization: Optional[str] = Header(None)):
    await require_admin(authorization)
    return await store.list_auth_users()


@app.delete("/admin/users/{uid}")
async def admin_delete_user(uid: str, authorization: Optional[str] = Header(None)):
    admin_uid = await require_admin(authorization)
    if uid == admin_uid:
        raise HTTPException(status_code=400, detail="不能刪除自己")
    await store.delete_auth_user(uid)
    return {"ok": True}


@app.post("/admin/users/{uid}/toggle-admin")
async def admin_toggle_admin(uid: str, is_admin: bool = True, authorization: Optional[str] = Header(None)):
    await require_admin(authorization)
    await store.set_admin_claim(uid, is_admin)
    return {"ok": True}
