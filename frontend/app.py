import os
import re
import json
import time
import base64
import asyncio
import datetime
from typing import Optional

import uuid
import secrets
import httpx
from fastapi import FastAPI, HTTPException, Header, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

# ------------------------------------------------------------------
# 設定
# ------------------------------------------------------------------
LITELLM_BASE_URL  = os.environ.get("LITELLM_BASE_URL", "http://localhost:4000")
LITELLM_API_KEY   = os.environ.get("LITELLM_MASTER_KEY", "sk-1234")
SMALL_MODEL_ALIAS = os.environ.get("SMALL_MODEL_ALIAS", "cloud-small")
LARGE_MODEL_ALIAS = os.environ.get("LARGE_MODEL_ALIAS", "cloud-large")
JUDGE_MODEL_ALIAS = os.environ.get("JUDGE_MODEL_ALIAS", "judge-model")
TINY_MODEL_ALIAS  = os.environ.get("TINY_MODEL_ALIAS", "")  # 開源小模型，選填
HISTORY_LIMIT     = 10
GCS_BUCKET        = os.environ.get("GCS_BUCKET", "ntpu-ai-uploads")
UPLOAD_MAX_BYTES  = 20 * 1024 * 1024  # 20 MB
SERPER_KEY        = os.environ.get("SERPER_API_KEY", "")


# ------------------------------------------------------------------
# Google Cloud Storage
# ------------------------------------------------------------------
try:
    from google.cloud import storage as _gcs_lib
    _gcs_client = _gcs_lib.Client()
    _gcs_ready  = True
except Exception:
    _gcs_client = None
    _gcs_ready  = False


# ------------------------------------------------------------------
# Firebase Admin
# ------------------------------------------------------------------
import firebase_admin
from firebase_admin import credentials, auth as fb_auth, firestore as fb_firestore

_sa_b64 = os.environ.get("FIREBASE_SERVICE_ACCOUNT_B64", "")
if _sa_b64:
    _sa_dict = json.loads(base64.b64decode(_sa_b64).decode())
    firebase_admin.initialize_app(credentials.Certificate(_sa_dict))
    _db = fb_firestore.client()
    _firebase_ready = True
else:
    _db = None
    _firebase_ready = False

# ------------------------------------------------------------------
# FastAPI
# ------------------------------------------------------------------
app = FastAPI(title="AI Router Backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ChatRequest(BaseModel):
    session_id: str
    message: str
    file_gcs_path:  Optional[str] = None
    file_mime_type: Optional[str] = None
    file_name:      Optional[str] = None
    search_enabled: bool = False


class RoutingConfig(BaseModel):
    threshold_tiny: Optional[float] = None   # None = 不啟用開源小模型層
    threshold_large: float = 6.0
    force_model: Optional[str] = None        # None/"small"/"large"/"tiny"


class UserProfileRequest(BaseModel):
    system_prompt: Optional[str] = None


# ------------------------------------------------------------------
# Auth helpers
# ------------------------------------------------------------------
async def decode_token(authorization: Optional[str]) -> dict:
    if not _firebase_ready:
        return {"uid": "anonymous", "email": ""}
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing auth token")
    try:
        return await asyncio.to_thread(fb_auth.verify_id_token, authorization[7:])
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid auth token")


async def verify_token(authorization: Optional[str] = Header(None)) -> str:
    return (await decode_token(authorization)).get("uid", "anonymous")


async def require_admin(authorization: Optional[str] = Header(None)) -> str:
    decoded = await decode_token(authorization)
    if not decoded.get("admin"):
        raise HTTPException(status_code=403, detail="Admin only")
    return decoded["uid"]


# ------------------------------------------------------------------
# Routing config（Firestore 讀取，30 秒快取）
# ------------------------------------------------------------------
_routing_cache: dict = {"threshold_tiny": None, "threshold_large": 6.0, "force_model": None}
_routing_cache_ts: float = 0.0


def _fs_get_routing_config() -> dict:
    doc = _db.collection("config").document("routing").get()
    return doc.to_dict() if doc.exists else {"threshold_tiny": None, "threshold_large": 6.0, "force_model": None}


def _fs_set_routing_config(data: dict):
    _db.collection("config").document("routing").set(data)


async def get_routing_config() -> dict:
    global _routing_cache, _routing_cache_ts
    if not _firebase_ready:
        return _routing_cache
    if time.time() - _routing_cache_ts > 30:
        _routing_cache = await asyncio.to_thread(_fs_get_routing_config)
        _routing_cache_ts = time.time()
    return _routing_cache


# ------------------------------------------------------------------
# Firestore helpers
# ------------------------------------------------------------------
def _ts_to_str(ts):
    if ts is None:
        return None
    try:
        return ts.isoformat()
    except AttributeError:
        return None


def _ms_to_str(ms):
    if ms is None:
        return None
    try:
        return datetime.datetime.fromtimestamp(ms / 1000, tz=datetime.timezone.utc).isoformat()
    except Exception:
        return None


def _fs_get_history(uid: str, session_id: str) -> list:
    doc = _db.collection("users").document(uid).collection("sessions").document(session_id).get()
    return doc.to_dict().get("history", []) if doc.exists else []


def _content_str(c) -> str:
    if isinstance(c, list):
        return next((p.get("text", "") for p in c if p.get("type") == "text"), "")
    return c or ""


def _fs_save_history(uid: str, session_id: str, history: list):
    title = next((_content_str(m["content"])[:60] for m in history if m["role"] == "user"), "對話")
    _db.collection("users").document(uid).collection("sessions").document(session_id).set({
        "history": history,
        "title": title,
        "updated_at": fb_firestore.SERVER_TIMESTAMP,
    })


def _fs_delete_session(uid: str, session_id: str):
    _db.collection("users").document(uid).collection("sessions").document(session_id).delete()


def _fs_get_user_profile(uid: str) -> dict:
    doc = _db.collection("users").document(uid).get()
    return (doc.to_dict() or {}) if doc.exists else {}


def _fs_set_user_profile(uid: str, data: dict):
    _db.collection("users").document(uid).set(data, merge=True)


def _fs_create_share(uid: str, session_id: str, history: list, title: str) -> str:
    share_id = secrets.token_urlsafe(12)
    _db.collection("public_shares").document(share_id).set({
        "uid": uid, "session_id": session_id,
        "history": history, "title": title,
        "created_at": fb_firestore.SERVER_TIMESTAMP,
    })
    return share_id


def _fs_get_share(share_id: str) -> dict:
    doc = _db.collection("public_shares").document(share_id).get()
    return doc.to_dict() if doc.exists else {}


async def get_user_system_prompt(uid: str) -> str:
    if not _firebase_ready or uid == "anonymous":
        return ""
    try:
        profile = await asyncio.to_thread(_fs_get_user_profile, uid)
        return profile.get("system_prompt", "")
    except Exception:
        return ""


def _fs_list_sessions(uid: str) -> list:
    docs = (
        _db.collection("users").document(uid).collection("sessions")
        .order_by("updated_at", direction="DESCENDING").limit(50).stream()
    )
    result = []
    for d in docs:
        data = d.to_dict()
        result.append({
            "session_id": d.id,
            "title": data.get("title", "對話"),
            "updated_at": _ts_to_str(data.get("updated_at")),
        })
    return result


def _fs_get_session_data(uid: str, session_id: str) -> dict:
    doc = _db.collection("users").document(uid).collection("sessions").document(session_id).get()
    return doc.to_dict() if doc.exists else {}


def _fs_log_usage(uid: str, email: str, session_id: str, route: str, score: float,
                  model: str, input_tokens: int = 0, output_tokens: int = 0):
    _db.collection("usage_logs").add({
        "uid": uid, "email": email, "session_id": session_id,
        "route": route, "score": score, "model": model,
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "timestamp": fb_firestore.SERVER_TIMESTAMP,
    })


def _fs_get_stats() -> list:
    cutoff = datetime.datetime.now(tz=datetime.timezone.utc) - datetime.timedelta(days=30)
    docs = _db.collection("usage_logs").where("timestamp", ">=", cutoff).stream()
    stats: dict = {}
    for doc in docs:
        d = doc.to_dict()
        uid = d.get("uid", "?")
        if uid not in stats:
            stats[uid] = {
                "uid": uid, "email": d.get("email", ""),
                "total": 0, "small": 0, "large": 0, "tiny": 0,
                "input_tokens": 0, "output_tokens": 0,
            }
        stats[uid]["total"] += 1
        route = d.get("route", "small")
        if route in stats[uid]:
            stats[uid][route] += 1
        stats[uid]["input_tokens"]  += d.get("input_tokens", 0)
        stats[uid]["output_tokens"] += d.get("output_tokens", 0)
    return sorted(stats.values(), key=lambda x: x["total"], reverse=True)


def _fs_list_auth_users() -> list:
    result = []
    page = fb_auth.list_users()
    while page:
        for u in page.users:
            result.append({
                "uid": u.uid,
                "email": u.email or "",
                "is_admin": bool(u.custom_claims and u.custom_claims.get("admin")),
                "created_at": _ms_to_str(u.user_metadata.creation_timestamp),
            })
        page = page.get_next_page()
    return result


async def get_history(uid: str, session_id: str) -> list:
    if not _firebase_ready or uid == "anonymous":
        return []
    return await asyncio.to_thread(_fs_get_history, uid, session_id)


async def save_history(uid: str, session_id: str, history: list):
    if not _firebase_ready or uid == "anonymous":
        return
    await asyncio.to_thread(_fs_save_history, uid, session_id, history)


async def delete_session(uid: str, session_id: str):
    if not _firebase_ready or uid == "anonymous":
        return
    await asyncio.to_thread(_fs_delete_session, uid, session_id)


# ------------------------------------------------------------------
# LiteLLM 呼叫
# ------------------------------------------------------------------
async def call_litellm(client: httpx.AsyncClient, model_alias: str, messages: list, max_tokens: int = 65536) -> str:
    resp = await client.post(
        f"{LITELLM_BASE_URL}/v1/chat/completions",
        headers={"Authorization": f"Bearer {LITELLM_API_KEY}"},
        json={"model": model_alias, "messages": messages, "max_tokens": max_tokens},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    if content is None:
        content = data["choices"][0]["message"].get("reasoning_content") or ""
    usage = data.get("usage", {})
    return content, {"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0)}


# ------------------------------------------------------------------
# File attachment → multimodal content
# ------------------------------------------------------------------
TEXT_MIMES = {"text/", "application/json", "application/xml", "application/javascript",
              "application/x-python", "application/x-sh"}

# Gemini 支援直接送 base64 的格式
INLINE_MIMES = {
    "image/jpeg", "image/png", "image/gif", "image/webp",
    "application/pdf",
    "audio/wav", "audio/mp3", "audio/mpeg", "audio/aiff", "audio/aac", "audio/ogg", "audio/flac",
    "video/mp4", "video/mpeg", "video/mov", "video/avi", "video/webm", "video/3gpp",
}

OFFICE_MIMES = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": (
        "word/document.xml",
        "http://schemas.openxmlformats.org/wordprocessingml/2006/main", "w",
    ),
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": (
        None,  # pptx: 多個 slide xml
        "http://schemas.openxmlformats.org/drawingml/2006/main", "a",
    ),
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": (
        "xl/sharedStrings.xml",
        "http://schemas.openxmlformats.org/spreadsheetml/2006/main", "s",
    ),
}


def _extract_office_text(file_bytes: bytes, mime: str) -> str:
    import zipfile, xml.etree.ElementTree as ET
    from io import BytesIO
    try:
        with zipfile.ZipFile(BytesIO(file_bytes)) as z:
            names = z.namelist()
            parts = []

            if "word/document.xml" in names:  # docx
                with z.open("word/document.xml") as f:
                    tree = ET.parse(f)
                    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
                    parts = [t.text or "" for t in tree.findall(".//w:t", ns)]

            elif any(n.startswith("ppt/slides/slide") for n in names):  # pptx
                slides = sorted(n for n in names if n.startswith("ppt/slides/slide") and n.endswith(".xml"))
                for slide in slides:
                    with z.open(slide) as f:
                        tree = ET.parse(f)
                        ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
                        parts += [t.text or "" for t in tree.findall(".//a:t", ns)]

            elif "xl/sharedStrings.xml" in names:  # xlsx
                with z.open("xl/sharedStrings.xml") as f:
                    tree = ET.parse(f)
                    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
                    parts = [t.text or "" for t in tree.findall(".//s:t", ns)]

            return "\n".join(p for p in parts if p.strip())
    except Exception:
        return ""


async def build_user_content(message: str, file_gcs_path: Optional[str],
                              file_mime_type: Optional[str]):
    if not file_gcs_path or not _gcs_ready:
        return message
    try:
        blob = _gcs_client.bucket(GCS_BUCKET).blob(file_gcs_path)
        file_bytes = await asyncio.to_thread(blob.download_as_bytes)
        mime = (file_mime_type or "application/octet-stream").lower()

        # 純文字類型：直接附在訊息裡
        is_text = any(mime.startswith(t) if t.endswith("/") else mime == t for t in TEXT_MIMES)
        if is_text:
            text_content = file_bytes.decode("utf-8", errors="replace")
            return f"{message}\n\n```\n{text_content[:50000]}\n```"

        # Office Open XML（.docx / .pptx / .xlsx）：解析文字
        if mime in OFFICE_MIMES:
            doc_text = await asyncio.to_thread(_extract_office_text, file_bytes, mime)
            if doc_text:
                return f"{message}\n\n以下是文件內容：\n\n{doc_text[:50000]}"
            return message

        # Gemini 原生支援的二進位格式（圖片、PDF、音訊、影片）
        if mime in INLINE_MIMES:
            b64 = base64.b64encode(file_bytes).decode()
            return [
                {"type": "text", "text": message},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ]

        # 不支援的格式（.doc/.ppt/.xls 舊格式等）
        ext = os.path.splitext(file_gcs_path)[1].upper() or mime
        return f"{message}\n\n（系統無法解析 {ext} 格式，請改用 .docx、.pptx、.xlsx、PDF 或圖片）"
    except Exception:
        return message


# ------------------------------------------------------------------
# Web search (Serper)
# ------------------------------------------------------------------
async def web_search(query: str, count: int = 5):
    """Returns search results str, "" on generic error, None on quota exceeded."""
    if not SERPER_KEY:
        return ""
    try:
        async with httpx.AsyncClient() as c:
            resp = await c.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": SERPER_KEY, "Content-Type": "application/json"},
                json={"q": query, "num": count, "gl": "tw", "hl": "zh-tw"},
                timeout=10,
            )
            if resp.status_code in (403, 429):
                return None  # quota exceeded
            resp.raise_for_status()
            results = resp.json().get("organic", [])
            lines = [
                f"• {r.get('title','')}\n  {r.get('snippet','')}\n  {r.get('link','')}"
                for r in results[:count] if r.get("title")
            ]
            return "\n\n".join(lines)
    except Exception:
        return ""


def _inject_search(user_content, search_ctx: str):
    suffix = f"\n\n【網路搜尋結果】\n{search_ctx}"
    if isinstance(user_content, str):
        return user_content + suffix
    parts = list(user_content)
    parts[0] = {"type": "text", "text": parts[0]["text"] + suffix}
    return parts


# ------------------------------------------------------------------
# Judge
# ------------------------------------------------------------------
JUDGE_SYSTEM_PROMPT = """你是一個路由決策模型，專門負責評估使用者訊息的任務難度，決定要交給輕量模型（Flash）還是強力模型（Pro）處理。

你的唯一工作是輸出難度分數，不要回答使用者的問題。

評分標準（0–10）：
- 0–3：閒聊、問候、簡單查詢、是非題、單一事實查詢
- 4–6：需要解釋概念、簡單摘要、基本程式碼片段、一般性建議
- 7–10：多步驟推理、複雜程式實作、數學證明、需要深度分析或跨領域整合的任務

注意事項：
- 若訊息本身簡短，但對話脈絡顯示是複雜任務的延伸（如「幫我改一下」接在程式碼討論後），請評估整個任務的難度
- 評分要保守：寧可低估讓 Flash 先試，也不要動輒給高分浪費 Pro

輸出格式（嚴格遵守，不得有多餘文字）：
{"score": 數字, "reason": "一句話說明"}"""


async def model_classify(client: httpx.AsyncClient, text: str, history: list) -> dict:
    if history:
        lines = [f"{'使用者' if m['role']=='user' else 'AI'}：{m['content'][:400]}" for m in history[-HISTORY_LIMIT:]]
        context_block = "對話歷史（供參考）：\n\"\"\"\n" + "\n".join(lines) + "\n\"\"\"\n\n"
    else:
        context_block = ""
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": f"{context_block}請評估以下最新訊息的難度：\n```\n{text}\n```"},
    ]
    raw, judge_usage = await call_litellm(client, JUDGE_MODEL_ALIAS, messages, max_tokens=1024)
    # 先嘗試直接解析，再用 regex 從回應中抓出 {...}
    parsed = None
    for candidate in [raw, re.sub(r"```(?:json)?|```", "", raw).strip()]:
        try:
            parsed = json.loads(candidate)
            break
        except Exception:
            pass
    if parsed is None:
        m = re.search(r"\{.*\}", raw, re.DOTALL)  # greedy: first { to last }
        if m:
            try:
                parsed = json.loads(m.group())
            except Exception:
                pass
    if parsed:
        score = max(0.0, min(10.0, float(parsed.get("score", 5))))
        return {"score": score, "reason": parsed.get("reason", ""), "normalized": score / 10.0, "_usage": judge_usage}
    # 解析失敗：記錄 raw 幫助偵錯
    import logging
    logging.warning(f"[judge] parse failed, raw={raw[:300]!r}")
    return {"score": 5.0, "reason": "", "normalized": 0.5, "_usage": judge_usage}


# ------------------------------------------------------------------
# Chat API
# ------------------------------------------------------------------
@app.get("/")
async def index():
    return FileResponse("index.html")


@app.post("/chat")
async def chat(req: ChatRequest, authorization: Optional[str] = Header(None)):
    decoded = await decode_token(authorization)
    uid   = decoded.get("uid", "anonymous")
    email = decoded.get("email", "")

    history = await get_history(uid, req.session_id)
    llm_history = [{"role": m["role"], "content": m["content"]} for m in history]
    config = await get_routing_config()

    async with httpx.AsyncClient() as client:
        t0 = time.time()
        judge = await model_classify(client, req.message, llm_history)
        judge_elapsed_ms = int((time.time() - t0) * 1000)

        # force_model 只對管理員帳號生效，一般使用者維持正常路由
        force     = config.get("force_model") if decoded.get("admin") else None
        t_large   = float(config.get("threshold_large") or 6.0)
        t_tiny    = config.get("threshold_tiny")

        if force in ("small", "large", "tiny"):
            route = force
        elif t_tiny is not None and judge["score"] < float(t_tiny):
            route = "tiny"
        elif judge["score"] >= t_large:
            route = "large"
        else:
            route = "small"

        model_alias = {
            "tiny":  TINY_MODEL_ALIAS or SMALL_MODEL_ALIAS,
            "small": SMALL_MODEL_ALIAS,
            "large": LARGE_MODEL_ALIAS,
        }[route]

        judge_usage = judge.pop("_usage", {"input_tokens": 0, "output_tokens": 0})
        sys_prompt = await get_user_system_prompt(uid)
        search_ctx = ""
        if req.search_enabled and SERPER_KEY:
            result = await web_search(req.message)
            search_ctx = result or ""  # None (quota) treated as empty
        user_content = await build_user_content(req.message, req.file_gcs_path, req.file_mime_type)
        if search_ctx:
            user_content = _inject_search(user_content, search_ctx)
        answer_messages = []
        if sys_prompt:
            answer_messages.append({"role": "system", "content": sys_prompt})
        answer_messages += llm_history[-HISTORY_LIMIT:] + [{"role": "user", "content": user_content}]
        t1 = time.time()
        answer, answer_usage = await call_litellm(client, model_alias, answer_messages)
        answer_elapsed_ms = int((time.time() - t1) * 1000)

    total_input  = judge_usage["input_tokens"]  + answer_usage["input_tokens"]
    total_output = judge_usage["output_tokens"] + answer_usage["output_tokens"]

    user_entry: dict = {"role": "user", "content": req.message}
    if req.file_name:
        user_entry["_file_name"]      = req.file_name
        user_entry["_file_mime_type"] = req.file_mime_type or ""
        user_entry["_file_gcs_path"]  = req.file_gcs_path or ""
    new_history = history + [
        user_entry,
        {"role": "assistant", "content": answer,
         "_route": route, "_score": judge["score"], "_reason": judge.get("reason", "")},
    ]
    await save_history(uid, req.session_id, new_history)

    if uid != "anonymous":
        asyncio.create_task(asyncio.to_thread(
            _fs_log_usage, uid, email, req.session_id, route, judge["score"], model_alias,
            total_input, total_output,
        ))

    return {
        "route": route, "model": model_alias, "judge": judge,
        "session_score": judge["normalized"],
        "judge_elapsed_ms": judge_elapsed_ms,
        "answer_elapsed_ms": answer_elapsed_ms,
        "answer": answer,
    }


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, authorization: Optional[str] = Header(None)):
    decoded = await decode_token(authorization)
    uid   = decoded.get("uid", "anonymous")
    email = decoded.get("email", "")

    history = await get_history(uid, req.session_id)
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
                force   = config.get("force_model") if decoded.get("admin") else None
                t_large = float(config.get("threshold_large") or 6.0)
                t_tiny  = config.get("threshold_tiny")

                if force in ("small", "large", "tiny"):
                    route = force
                elif t_tiny is not None and judge["score"] < float(t_tiny):
                    route = "tiny"
                elif judge["score"] >= t_large:
                    route = "large"
                else:
                    route = "small"

                model_alias = {
                    "tiny":  TINY_MODEL_ALIAS or SMALL_MODEL_ALIAS,
                    "small": SMALL_MODEL_ALIAS,
                    "large": LARGE_MODEL_ALIAS,
                }[route]

                # 3. 先送 judge metadata（順便取出 usage，不傳給前端）
                judge_usage = judge.pop("_usage", {"input_tokens": 0, "output_tokens": 0})
                yield f"data: {json.dumps({'type':'judge','route':route,'model':model_alias,'judge':judge,'judge_elapsed_ms':judge_elapsed_ms})}\n\n"

                # 4. 準備訊息（含系統提示 + 網路搜尋）
                sys_prompt = await get_user_system_prompt(uid)
                search_ctx = ""
                if req.search_enabled and SERPER_KEY:
                    yield f"data: {json.dumps({'type':'search'})}\n\n"
                    result = await web_search(req.message)
                    if result is None:
                        yield f"data: {json.dumps({'type':'search_quota'})}\n\n"
                    else:
                        search_ctx = result
                user_content = await build_user_content(req.message, req.file_gcs_path, req.file_mime_type)
                if search_ctx:
                    user_content = _inject_search(user_content, search_ctx)
                answer_messages = []
                if sys_prompt:
                    answer_messages.append({"role": "system", "content": sys_prompt})
                answer_messages += llm_history[-HISTORY_LIMIT:] + [{"role": "user", "content": user_content}]
                t1 = time.time()
                full_content = ""
                answer_input_tokens  = 0
                answer_output_tokens = 0

                async with client.stream(
                    "POST",
                    f"{LITELLM_BASE_URL}/v1/chat/completions",
                    headers={"Authorization": f"Bearer {LITELLM_API_KEY}"},
                    json={"model": model_alias, "messages": answer_messages, "max_tokens": 65536,
                          "stream": True, "stream_options": {"include_usage": True}},
                    timeout=httpx.Timeout(connect=30, read=300, write=30, pool=10),
                ) as resp:
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
                                    yield f"data: {json.dumps({'type':'token','content':content})}\n\n"
                        except Exception:
                            pass

                answer_elapsed_ms = int((time.time() - t1) * 1000)
                total_input  = judge_usage["input_tokens"]  + answer_input_tokens
                total_output = judge_usage["output_tokens"] + answer_output_tokens

                yield f"data: {json.dumps({'type':'done','answer_elapsed_ms':answer_elapsed_ms})}\n\n"
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
                await save_history(uid, req.session_id, new_history)

                if uid != "anonymous":
                    asyncio.create_task(asyncio.to_thread(
                        _fs_log_usage, uid, email, req.session_id, route, judge["score"], model_alias,
                        total_input, total_output,
                    ))

        except Exception as e:
            yield f"data: {json.dumps({'type':'error','message':str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/reset")
async def reset(session_id: str, authorization: Optional[str] = Header(None)):
    uid = await verify_token(authorization)
    await delete_session(uid, session_id)
    return {"ok": True}


@app.get("/conversations")
async def list_conversations(authorization: Optional[str] = Header(None)):
    uid = await verify_token(authorization)
    if uid == "anonymous":
        return []
    return await asyncio.to_thread(_fs_list_sessions, uid)


@app.get("/conversations/{session_id}")
async def get_conversation(session_id: str, authorization: Optional[str] = Header(None)):
    uid = await verify_token(authorization)
    if uid == "anonymous":
        raise HTTPException(status_code=401)
    data = await asyncio.to_thread(_fs_get_session_data, uid, session_id)
    return {"session_id": session_id, "history": data.get("history", []), "title": data.get("title", "對話")}


@app.get("/user/profile")
async def get_profile(authorization: Optional[str] = Header(None)):
    decoded = await decode_token(authorization)
    uid = decoded.get("uid", "anonymous")
    if uid == "anonymous":
        return {"system_prompt": ""}
    profile = await asyncio.to_thread(_fs_get_user_profile, uid)
    return {"system_prompt": profile.get("system_prompt", "")}


@app.post("/user/profile")
async def set_profile(data: UserProfileRequest, authorization: Optional[str] = Header(None)):
    decoded = await decode_token(authorization)
    uid = decoded.get("uid", "anonymous")
    if uid == "anonymous":
        raise HTTPException(status_code=401)
    await asyncio.to_thread(_fs_set_user_profile, uid, {"system_prompt": data.system_prompt or ""})
    return {"ok": True}


@app.post("/conversations/{session_id}/share")
async def share_conversation(session_id: str, authorization: Optional[str] = Header(None)):
    decoded = await decode_token(authorization)
    uid = decoded.get("uid", "anonymous")
    if uid == "anonymous":
        raise HTTPException(status_code=401)
    history = await get_history(uid, session_id)
    if not history:
        raise HTTPException(status_code=404, detail="對話不存在")
    title = next((_content_str(m["content"])[:40] for m in history if m["role"] == "user"), "對話")
    public_history = [
        {"role": m["role"], "content": m["content"],
         "_route": m.get("_route"), "_score": m.get("_score"), "_reason": m.get("_reason"),
         "_file_name": m.get("_file_name")}
        for m in history
    ]
    share_id = await asyncio.to_thread(_fs_create_share, uid, session_id, public_history, title)
    return {"share_id": share_id}


@app.get("/share/{share_id}")
async def get_shared_conversation(share_id: str):
    data = await asyncio.to_thread(_fs_get_share, share_id)
    if not data:
        raise HTTPException(status_code=404, detail="分享連結不存在或已失效")
    return {"history": data.get("history", []), "title": data.get("title", "對話")}


# ------------------------------------------------------------------
# Admin API
# ------------------------------------------------------------------
@app.post("/admin/setup")
async def admin_setup(authorization: Optional[str] = Header(None)):
    """第一次使用：若系統中無管理員，授予請求者管理員權限"""
    if not _firebase_ready:
        raise HTTPException(status_code=503)
    decoded = await decode_token(authorization)
    uid = decoded.get("uid")
    if not uid or uid == "anonymous":
        raise HTTPException(status_code=401)

    has_admin = False
    page = fb_auth.list_users()
    while page:
        for u in page.users:
            if u.custom_claims and u.custom_claims.get("admin"):
                has_admin = True
                break
        if has_admin:
            break
        page = page.get_next_page()

    if has_admin:
        raise HTTPException(status_code=403, detail="管理員已存在，請聯絡現有管理員授權")

    await asyncio.to_thread(fb_auth.set_custom_user_claims, uid, {"admin": True})
    return {"ok": True}


@app.get("/admin/config")
async def admin_get_config(authorization: Optional[str] = Header(None)):
    await require_admin(authorization)
    return await get_routing_config()


@app.post("/admin/config")
async def admin_set_config(config: RoutingConfig, authorization: Optional[str] = Header(None)):
    global _routing_cache_ts
    await require_admin(authorization)
    await asyncio.to_thread(_fs_set_routing_config, config.dict())
    _routing_cache_ts = 0
    return {"ok": True}


@app.get("/admin/stats")
async def admin_stats(authorization: Optional[str] = Header(None)):
    await require_admin(authorization)
    return await asyncio.to_thread(_fs_get_stats)


@app.get("/admin/users")
async def admin_list_users(authorization: Optional[str] = Header(None)):
    await require_admin(authorization)
    return await asyncio.to_thread(_fs_list_auth_users)


@app.delete("/admin/users/{uid}")
async def admin_delete_user(uid: str, authorization: Optional[str] = Header(None)):
    admin_uid = await require_admin(authorization)
    if uid == admin_uid:
        raise HTTPException(status_code=400, detail="不能刪除自己")
    await asyncio.to_thread(fb_auth.delete_user, uid)
    return {"ok": True}


@app.post("/admin/users/{uid}/toggle-admin")
async def admin_toggle_admin(uid: str, is_admin: bool = True, authorization: Optional[str] = Header(None)):
    await require_admin(authorization)
    claims = {"admin": True} if is_admin else {}
    await asyncio.to_thread(fb_auth.set_custom_user_claims, uid, claims)
    return {"ok": True}


@app.get("/file-preview")
async def file_preview(path: str, authorization: Optional[str] = Header(None)):
    from fastapi.responses import Response
    decoded = await decode_token(authorization)
    uid = decoded.get("uid", "anonymous")
    if not _gcs_ready:
        raise HTTPException(status_code=503)
    if not path.startswith(f"uploads/{uid}/"):
        raise HTTPException(status_code=403)
    try:
        blob = _gcs_client.bucket(GCS_BUCKET).blob(path)
        file_bytes = await asyncio.to_thread(blob.download_as_bytes)
        mime = (blob.content_type or "application/octet-stream").lower()
        if mime in OFFICE_MIMES:
            text = await asyncio.to_thread(_extract_office_text, file_bytes, mime)
            return Response(content=text or "（無法取出文字內容）", media_type="text/plain; charset=utf-8")
        is_text = any(mime.startswith(t) if t.endswith("/") else mime == t for t in TEXT_MIMES)
        if is_text:
            return Response(content=file_bytes, media_type=mime)
        return Response(content=file_bytes, media_type=mime)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(None),
):
    decoded = await decode_token(authorization)
    uid = decoded.get("uid", "anonymous")
    if not _gcs_ready:
        raise HTTPException(status_code=503, detail="Storage not configured")
    content = await file.read()
    if len(content) > UPLOAD_MAX_BYTES:
        raise HTTPException(status_code=413, detail="檔案超過 20MB 上限")
    ext      = os.path.splitext(file.filename or "")[1].lower()
    gcs_path = f"uploads/{uid}/{int(time.time())}_{uuid.uuid4().hex[:8]}{ext}"
    blob     = _gcs_client.bucket(GCS_BUCKET).blob(gcs_path)
    await asyncio.to_thread(blob.upload_from_string, content, file.content_type)
    return {
        "gcs_path":  gcs_path,
        "filename":  file.filename,
        "mime_type": file.content_type,
        "size":      len(content),
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
