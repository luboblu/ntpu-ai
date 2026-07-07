import os
import re
import json
import time
import base64
import asyncio
import datetime
import html as _htmllib
import urllib.parse
from typing import Optional

import uuid
import secrets
import httpx
from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

# ------------------------------------------------------------------
# 設定
# ------------------------------------------------------------------
LITELLM_BASE_URL  = os.environ.get("LITELLM_BASE_URL", "http://localhost:4000")
LITELLM_API_KEY   = os.environ.get("LITELLM_MASTER_KEY", "sk-1234")
SMALL_MODEL_ALIAS = os.environ.get("SMALL_MODEL_ALIAS", "cloud-small-claude")
MEDIUM_MODEL_ALIAS = os.environ.get("MEDIUM_MODEL_ALIAS", "cloud-medium-claude")
LARGE_MODEL_ALIAS = os.environ.get("LARGE_MODEL_ALIAS", "cloud-large-claude")
SMALL_MODEL_ALIASES = os.environ.get("SMALL_MODEL_ALIASES", f"{SMALL_MODEL_ALIAS},cloud-small-gemini")
MEDIUM_MODEL_ALIASES = os.environ.get("MEDIUM_MODEL_ALIASES", f"{MEDIUM_MODEL_ALIAS},cloud-medium-gemini")
LARGE_MODEL_ALIASES = os.environ.get("LARGE_MODEL_ALIASES", f"{LARGE_MODEL_ALIAS},cloud-large-gemini")
JUDGE_MODEL_ALIAS = os.environ.get("JUDGE_MODEL_ALIAS", "judge-model")
TINY_MODEL_ALIAS  = os.environ.get("TINY_MODEL_ALIAS", "")  # 開源小模型，選填
HISTORY_LIMIT     = 10
GCS_BUCKET        = os.environ.get("GCS_BUCKET", "ntpu-ai-uploads")
UPLOAD_MAX_BYTES  = 20 * 1024 * 1024  # 20 MB
TAVILY_KEY        = os.environ.get("TAVILY_API_KEY", "")
OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY", "")
MAX_ANSWER_TOKENS = 64000


def _alias_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


MODEL_CANDIDATES = {
    "small": _alias_list(SMALL_MODEL_ALIASES),
    "medium": _alias_list(MEDIUM_MODEL_ALIASES),
    "large": _alias_list(LARGE_MODEL_ALIASES),
}
if TINY_MODEL_ALIAS:
    MODEL_CANDIDATES["tiny"] = [TINY_MODEL_ALIAS]

MODEL_NOTES = {
    "cloud-small-claude": "Claude Haiku 4.5：快速、省成本，適合簡單問答與短任務",
    "cloud-small-gemini": "Gemini 3.1 Flash-Lite：最快速、成本低，適合大量輕量任務",
    "cloud-small-gpt": "GPT-4o mini（OpenRouter）：便宜快速，適合簡單問答與短任務",
    "cloud-medium-claude": "Claude Sonnet 5：品質穩定，適合一般推理、寫作與程式任務",
    "cloud-medium-gemini": "Gemini 3.5 Flash：低延遲且能力均衡，適合中等複雜任務",
    "cloud-medium-deepseek": "DeepSeek V3（OpenRouter）：高 CP 值、成本低，通用推理與程式能力強，適合中等任務",
    "cloud-medium-llama": "Llama 3.3 70B（OpenRouter）：開源模型、成本低，適合中等難度的一般任務",
    "cloud-large-claude": "Claude Opus 4.8：高品質深度推理、長任務與複雜 coding",
    "cloud-large-gemini": "Gemini 2.5 Pro：深度推理與 coding，適合複雜任務",
    "cloud-large-r1": "DeepSeek R1（OpenRouter）：強推理模型、成本低，適合數學證明與多步推理（不擅長工具呼叫）",
}

MODEL_TO_ROUTE = {
    alias: route
    for route, aliases in MODEL_CANDIDATES.items()
    for alias in aliases
}


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
_cors_origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()] or ["*"]
app.add_middleware(CORSMiddleware, allow_origins=_cors_origins, allow_methods=["*"], allow_headers=["*"])


class ChatRequest(BaseModel):
    session_id: str
    message: str
    file_gcs_path:  Optional[str] = None
    file_mime_type: Optional[str] = None
    file_name:      Optional[str] = None
    search_enabled: bool = False
    ntpu_search_enabled: bool = False


class RoutingConfig(BaseModel):
    threshold_tiny: Optional[float] = None   # None = 不啟用開源小模型層
    threshold_medium: float = 4.0
    threshold_large: float = 7.0
    force_model: Optional[str] = None        # None/"small"/"medium"/"large"/"tiny"


class UserProfileRequest(BaseModel):
    system_prompt: Optional[str] = None


# ------------------------------------------------------------------
# Auth helpers
# ------------------------------------------------------------------
ALLOWED_DOMAINS = {"gm.ntpu.edu.tw", "ms.ntpu.edu.tw"}


async def decode_token(authorization: Optional[str]) -> dict:
    if not _firebase_ready:
        return {"uid": "anonymous", "email": ""}
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing auth token")
    try:
        decoded = await asyncio.to_thread(fb_auth.verify_id_token, authorization[7:])
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid auth token")

    # 網域限制套用到所有非管理員，不分登入方式（google / password 一視同仁），
    # 避免有人用 email/password 自行註冊任意信箱繞過「僅限 NTPU 帳號」的限制。
    if not decoded.get("admin"):
        email = decoded.get("email", "")
        domain = email.split("@")[-1] if "@" in email else ""
        if domain not in ALLOWED_DOMAINS:
            raise HTTPException(
                status_code=403,
                detail="僅限 NTPU 師生帳號（@gm.ntpu.edu.tw）登入",
            )

    return decoded


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
_routing_cache: dict = {
    "threshold_tiny": None,
    "threshold_medium": 4.0,
    "threshold_large": 7.0,
    "force_model": None,
}
_routing_cache_ts: float = 0.0


def _fs_get_routing_config() -> dict:
    doc = _db.collection("config").document("routing").get()
    return doc.to_dict() if doc.exists else dict(_routing_cache)


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


def _default_model_for_route(route: str) -> str:
    candidates = MODEL_CANDIDATES.get(route) or MODEL_CANDIDATES["small"]
    return candidates[0]


# 不支援 function calling 的模型（例如 DeepSeek R1）；啟用搜尋工具時要避開，
# 否則工具呼叫會失敗。可用環境變數覆寫。
NO_TOOL_MODELS = set(_alias_list(os.environ.get("NO_TOOL_MODELS", "cloud-large-r1")))


def _pick_tool_capable(route: str, model_alias: str) -> tuple[str, str]:
    """啟用工具時，若選到不支援 function calling 的模型，改挑同級距其他可用模型；
    若整個級距都不支援，退到 medium 的預設模型。"""
    if model_alias not in NO_TOOL_MODELS:
        return route, model_alias
    for alias in MODEL_CANDIDATES.get(route, []):
        if alias not in NO_TOOL_MODELS:
            return route, alias
    return "medium", _default_model_for_route("medium")


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


def _select_route(config: dict, judge: dict, force: Optional[str]) -> tuple[str, str]:
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
    model_alias = requested_model if requested_model in candidates else candidates[0]
    return route, model_alias


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


def build_system_prompt(user_sys_prompt: str, has_tools: bool = False) -> str:
    today = datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=8))).strftime("%Y-%m-%d")
    parts = [f"今天的日期是 {today}（台灣時間）。"]
    if has_tools:
        parts.append(
            "你可以使用搜尋工具查詢即時或校內資訊。當使用者詢問臺北大學（NTPU）的公告、"
            "最新消息、課程、系所、行政規定、活動、師資等校內資訊，或需要最新／即時的網路"
            "資訊時，請呼叫對應的搜尋工具。若使用者想要「最新／近期」的北大公告，呼叫"
            "search_ntpu_web 時把 latest 設為 true；想瀏覽全校最新消息可用 get_ntpu_news；"
            "若搜尋摘要不足、需要某頁的完整內容，用 fetch_ntpu_page 讀取該網址全文。"
            "呼叫工具後，只能根據工具回傳的結果回答，"
            "不得用訓練資料編造日期、公告、活動或連結；若工具查無資料，請如實告知查不到，"
            "並建議前往官方網站確認。與搜尋無關的一般問題（閒聊、翻譯、寫作、算式等）"
            "直接回答即可，不需呼叫工具。回答涉及搜尋結果時請附上來源網址。"
        )
    else:
        parts.append(
            "你沒有即時上網或搜尋能力。若使用者詢問需要即時或最新資訊的問題"
            "（例如最新公告、近期新聞、活動、股價、天氣等），請明確告訴對方"
            "你無法查詢即時資料，並建議點選輸入框旁的「＋」開啟「網路搜尋」"
            "或「NTPU 校內搜尋」功能後再問一次。切勿假裝已經搜尋，"
            "也不要用訓練資料編造最新公告、日期、活動或連結。"
        )
    if user_sys_prompt:
        parts.append(user_sys_prompt)
    return "\n\n".join(parts)


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
                  model: str, input_tokens: int = 0, output_tokens: int = 0,
                  judge_model: str = "", judge_input_tokens: int = 0, judge_output_tokens: int = 0,
                  answer_input_tokens: int = 0, answer_output_tokens: int = 0):
    _db.collection("usage_logs").add({
        "uid": uid, "email": email, "session_id": session_id,
        "route": route, "score": score, "model": model,
        # input_tokens/output_tokens 為 judge+回答加總（維持舊版相容的每人統計）
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        # 分開記錄，才能精準做「依模型」計價
        "judge_model": judge_model,
        "judge_input_tokens": judge_input_tokens,
        "judge_output_tokens": judge_output_tokens,
        "answer_input_tokens": answer_input_tokens,
        "answer_output_tokens": answer_output_tokens,
        "timestamp": fb_firestore.SERVER_TIMESTAMP,
    })


def _fs_get_stats(start_dt: datetime.datetime, end_dt: datetime.datetime) -> dict:
    docs = (
        _db.collection("usage_logs")
        .where("timestamp", ">=", start_dt)
        .where("timestamp", "<", end_dt)
        .limit(20000).stream()
    )
    stats: dict = {}
    sessions: dict = {}   # uid -> set(session_id)，用來算「對話數」（不重複的 session）
    by_model: dict = {}   # model alias -> {input, output, requests}

    def _add_model(alias: str, inp: int, out: int):
        if not alias:
            return
        b = by_model.setdefault(alias, {"model": alias, "input_tokens": 0, "output_tokens": 0, "requests": 0})
        b["input_tokens"]  += inp
        b["output_tokens"] += out
        b["requests"]      += 1

    for doc in docs:
        d = doc.to_dict()
        uid = d.get("uid", "?")
        if uid not in stats:
            stats[uid] = {
                "uid": uid, "email": d.get("email", ""),
                "total": 0, "conversations": 0,
                "small": 0, "medium": 0, "large": 0, "tiny": 0,
                "input_tokens": 0, "output_tokens": 0,
            }
            sessions[uid] = set()
        stats[uid]["total"] += 1
        sid = d.get("session_id")
        if sid:
            sessions[uid].add(sid)
        route = d.get("route", "small")
        if route in stats[uid]:
            stats[uid][route] += 1
        stats[uid]["input_tokens"]  += d.get("input_tokens", 0)
        stats[uid]["output_tokens"] += d.get("output_tokens", 0)

        # 依模型分列：新版 log 有拆開 judge / 回答的 token，舊版則整筆歸給回答模型
        if "answer_input_tokens" in d or "judge_model" in d:
            _add_model(d.get("judge_model", ""), d.get("judge_input_tokens", 0), d.get("judge_output_tokens", 0))
            _add_model(d.get("model", ""), d.get("answer_input_tokens", 0), d.get("answer_output_tokens", 0))
        else:
            _add_model(d.get("model", ""), d.get("input_tokens", 0), d.get("output_tokens", 0))

    for uid, s in stats.items():
        s["conversations"] = len(sessions[uid])

    users = sorted(stats.values(), key=lambda x: x["total"], reverse=True)
    models = sorted(by_model.values(),
                    key=lambda x: x["input_tokens"] + x["output_tokens"], reverse=True)
    return {"users": users, "by_model": models}


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
async def call_litellm(client: httpx.AsyncClient, model_alias: str, messages: list, max_tokens: int = MAX_ANSWER_TOKENS) -> str:
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
async def web_search(query: str, count: int = 5, ntpu_only: bool = False,
                     sort_by_date: bool = False) -> dict:
    """Returns {"status": "ok"|"empty"|"quota"|"error", "text": str}."""
    if not TAVILY_KEY:
        return {"status": "error", "text": ""}
    try:
        payload: dict = {
            "api_key": TAVILY_KEY,
            "query": query,
            "max_results": count,
            "include_answer": True,
            "search_depth": "advanced",
        }
        if ntpu_only:
            payload["include_domains"] = ["ntpu.edu.tw"]
        else:
            payload["days"] = 30
        if sort_by_date:
            payload["time_range"] = "year"  # 偏向近一年，提升「最新公告」查詢的新鮮度
        async with httpx.AsyncClient() as c:
            resp = await c.post(
                "https://api.tavily.com/search",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=20,
            )
            if resp.status_code in (401, 403, 429):
                try:
                    msg = str(resp.json()).lower()
                    if any(k in msg for k in ("quota", "limit", "exceeded", "plan")):
                        return {"status": "quota", "text": ""}
                except Exception:
                    pass
                return {"status": "error", "text": ""}
            resp.raise_for_status()
            data = resp.json()
            today = datetime.datetime.now().strftime("%Y-%m-%d")
            lines = [f"（搜尋日期：{today}）"]
            if data.get("answer"):
                lines.append(f"摘要：{data['answer']}")
            results = data.get("results", [])
            if sort_by_date:
                # Tavily 預設依相似度排序；抓最新公告時改依發布日期新→舊，無日期者排最後
                results = sorted(results, key=lambda r: r.get("published_date") or "", reverse=True)
            found = 0
            for r in results[:count]:
                title   = r.get("title", "")
                content = r.get("content", "")[:300]
                url     = r.get("url", "")
                pub     = r.get("published_date", "")
                date_str = f" [{pub}]" if pub else ""
                if title:
                    lines.append(f"• {title}{date_str}\n  {content}\n  {url}")
                    found += 1
            # 只有摘要（answer）但沒有任何實際結果時，視為查無資料，避免 AI 拿空殼編造
            if found == 0:
                return {"status": "empty", "text": ""}
            return {"status": "ok", "text": "\n\n".join(lines)}
    except Exception:
        return {"status": "error", "text": ""}


# ------------------------------------------------------------------
# Tool calling（讓 LLM 自己決定何時搜尋）
# ------------------------------------------------------------------
NTPU_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_ntpu_web",
        "description": (
            "搜尋國立臺北大學（NTPU）官方網站（ntpu.edu.tw）內的資訊。"
            "當使用者詢問北大的公告、最新消息、課程、系所、行政規定、活動、"
            "師資或校內任何事務時使用。與北大無關的一般問題不要呼叫此工具。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜尋關鍵字，可用中文"},
                "latest": {
                    "type": "boolean",
                    "description": "使用者想要『最新／近期』的公告或消息（重視時間新舊）時設為 true，"
                                   "會改依發布日期由新到舊排序；一般查詢維持 false。",
                },
            },
            "required": ["query"],
        },
    },
}

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": (
            "搜尋即時網路資訊。當使用者詢問需要最新或即時的一般資訊"
            "（新聞、天氣、活動、非北大的時事等）時使用。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜尋關鍵字"},
            },
            "required": ["query"],
        },
    },
}


FETCH_PAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_ntpu_page",
        "description": (
            "打開指定的臺北大學（ntpu.edu.tw）網頁並讀取整頁全文。當你已從搜尋結果拿到某個北大網址、"
            "但摘要不足以回答使用者問題時，用此工具讀取該頁完整內容。只接受 ntpu.edu.tw 網域的網址。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "要讀取的臺北大學頁面完整網址（需為 ntpu.edu.tw 網域）"},
            },
            "required": ["url"],
        },
    },
}

NEWS_TOOL = {
    "type": "function",
    "function": {
        "name": "get_ntpu_news",
        "description": (
            "取得臺北大學首頁的最新消息列表（全校各單位的公告、新聞、活動），依網站排序（通常最新在前）。"
            "當使用者想瀏覽最新／近期消息，或不確定該用什麼關鍵字搜尋時使用。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "count": {"type": "integer", "description": "要回傳幾則，預設 15，最多 30"},
            },
        },
    },
}


# ---- 北大主站抓取（httpx；主站為 Angular SSR，新聞/文章頁內容在 HTML 內） ----
_NTPU_HOME = "https://new.ntpu.edu.tw/"
_BROWSER_UA = "Mozilla/5.0 (compatible; NTPU-AI/1.0)"


def _is_ntpu_url(url: str) -> bool:
    try:
        host = (urllib.parse.urlparse(url).hostname or "").lower()
    except Exception:
        return False
    return host == "ntpu.edu.tw" or host.endswith(".ntpu.edu.tw")


def _html_to_text(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", _htmllib.unescape(raw)).strip()


async def _fetch_ntpu(url: str) -> str:
    if not _is_ntpu_url(url):
        return "（只能讀取 ntpu.edu.tw 網域的頁面）"
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=20,
                                     headers={"User-Agent": _BROWSER_UA}) as c:
            resp = await c.get(url)
            resp.raise_for_status()
            raw = resp.text
            final_url = str(resp.url)
    except Exception:
        return f"（無法開啟該頁面：{url}）"
    text = _html_to_text(raw)
    if len(text) < 40:
        return f"（該頁面沒有可讀取的文字內容，可能需要瀏覽器才能顯示：{final_url}）"
    m = re.search(r"發布日期[：: ]*([0-9]{4}\s*/\s*[0-9]{1,2}\s*/\s*[0-9]{1,2})", raw)
    head = f"來源：{final_url}\n"
    if m:
        clean_date = re.sub(r"\s+", "", m.group(1))
        head += f"發布日期：{clean_date}\n"
    return head + "\n" + text[:6000]


async def _ntpu_latest_news(count: int = 15) -> str:
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=20,
                                     headers={"User-Agent": _BROWSER_UA}) as c:
            resp = await c.get(_NTPU_HOME)
            resp.raise_for_status()
            raw = resp.text
    except Exception:
        return "（無法取得臺北大學最新消息）"
    anchors = re.findall(r'<a[^>]+href="([^"]*?/news/[^"]+)"[^>]*>(.*?)</a>', raw, re.S | re.I)
    seen: set = set()
    items: list = []
    for href, inner in anchors:
        parts = [p for p in urllib.parse.urlparse(href).path.split("/") if p]
        if len(parts) < 2:
            continue
        nid = parts[1]
        if nid in seen or nid == "news":
            continue
        title_from_url = urllib.parse.unquote(parts[-1]) if len(parts) >= 3 else ""
        inner_text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", inner)).strip()
        title = inner_text or title_from_url
        if not title or title.startswith("SYSTEM."):
            continue
        seen.add(nid)
        full = href if href.startswith("http") else "https://new.ntpu.edu.tw/" + href.lstrip("/")
        items.append(f"• {title}\n  {full}")
        if len(items) >= count:
            break
    if not items:
        return "（目前抓不到最新消息列表）"
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    return (f"（臺北大學首頁最新消息，擷取日期 {today}，依網站排序，通常最新在前）\n\n"
            + "\n".join(items))


def _build_search_tools(req: "ChatRequest") -> list:
    tools = []
    if req.ntpu_search_enabled:
        tools += [NTPU_SEARCH_TOOL, FETCH_PAGE_TOOL, NEWS_TOOL]
    if req.search_enabled:
        tools.append(WEB_SEARCH_TOOL)
    return tools


async def _run_search_tool(name: str, args: dict) -> str:
    if name == "fetch_ntpu_page":
        return await _fetch_ntpu((args.get("url") or "").strip())
    if name == "get_ntpu_news":
        try:
            n = int(args.get("count") or 15)
        except Exception:
            n = 15
        return await _ntpu_latest_news(max(1, min(30, n)))
    query = (args.get("query") or "").strip()
    if not query:
        return "（未提供搜尋關鍵字）"
    if name == "search_ntpu_web":
        sr = await web_search(query, count=8, ntpu_only=True, sort_by_date=bool(args.get("latest")))
        scope = "臺北大學官網"
    elif name == "search_web":
        sr = await web_search(query, count=5)
        scope = "網路"
    else:
        return "（未知的工具）"
    if sr["status"] == "ok":
        return sr["text"]
    if sr["status"] == "quota":
        return f"（{scope}搜尋額度已用盡，暫時無法查詢，請稍後再試）"
    return f"（在{scope}查無「{query}」的相關資料）"


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
            headers={"Authorization": f"Bearer {LITELLM_API_KEY}"},
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
                    result = await _run_search_tool(fn.get("name", ""), args)
                except Exception:
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


def _chunk_text(text: str, size: int = 24):
    for i in range(0, len(text), size):
        yield text[i:i + size]


# ------------------------------------------------------------------
# Judge
# ------------------------------------------------------------------
JUDGE_SYSTEM_PROMPT = """你是一個路由決策模型，專門負責評估使用者訊息的任務難度，決定要交給哪個級距與哪個回答模型處理。

你的唯一工作是輸出路由 JSON，不要回答使用者的問題。

評分標準（0–10）：
- 0–3：閒聊、問候、簡單查詢、是非題、單一事實查詢
- 4–6：需要解釋概念、簡單摘要、基本程式碼片段、一般性建議
- 7–10：多步驟推理、複雜程式實作、數學證明、需要深度分析或跨領域整合的任務

級距選擇：
- small：0–3 分，低成本快速回答
- medium：4–6 分，品質與速度平衡
- large：7–10 分，深度推理、複雜 coding、長任務

注意事項：
- 若訊息本身簡短，但對話脈絡顯示是複雜任務的延伸（如「幫我改一下」接在程式碼討論後），請評估整個任務的難度
- 評分要保守：寧可低估讓較小模型先試，也不要動輒給高分浪費大模型
- model 必須從使用者訊息提供的「可選模型」清單中挑選，不得自創模型名稱
- 若同級距有多個模型，依任務特性挑選：Claude 偏向穩定寫作、深度 coding、長對話；Gemini 偏向低延遲、多模態、Google 生態與高吞吐

輸出格式（嚴格遵守，不得有多餘文字）：
{"score": 數字, "route": "small|medium|large", "model": "模型 alias", "reason": "一句話說明"}"""


async def model_classify(client: httpx.AsyncClient, text: str, history: list) -> dict:
    if history:
        lines = [f"{'使用者' if m['role']=='user' else 'AI'}：{m['content'][:400]}" for m in history[-HISTORY_LIMIT:]]
        context_block = "對話歷史（供參考）：\n\"\"\"\n" + "\n".join(lines) + "\n\"\"\"\n\n"
    else:
        context_block = ""
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": f"{context_block}可選模型：\n{_model_options_text()}\n\n請評估以下最新訊息的難度並選擇級距與模型：\n```\n{text}\n```"},
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
    # 解析失敗：記錄 raw 幫助偵錯
    import logging
    logging.warning(f"[judge] parse failed, raw={raw[:300]!r}")
    return {"score": 5.0, "route": None, "model": None, "reason": "", "normalized": 0.5, "_usage": judge_usage}


# ------------------------------------------------------------------
# Chat API
# ------------------------------------------------------------------
# 唯一正本是 frontend/index.html。Docker build 會把它 COPY 到 app.py 旁邊（/app/index.html）；
# 本機從 repo 直接跑時則往上一層找 frontend/index.html。
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
                force = config.get("force_model") if decoded.get("admin") else None
                route, model_alias = _select_route(config, judge, force)
                judge["route"] = route
                judge["model"] = model_alias

                tools = _build_search_tools(req) if TAVILY_KEY else []
                # 工具啟用時避免路由到不支援 function calling 的模型
                if tools and route == "tiny":
                    route, model_alias = "small", _default_model_for_route("small")
                if tools:
                    route, model_alias = _pick_tool_capable(route, model_alias)
                    judge["route"] = route
                    judge["model"] = model_alias

                # 3. 先送 judge metadata（順便取出 usage，不傳給前端）
                judge_usage = judge.pop("_usage", {"input_tokens": 0, "output_tokens": 0})
                yield f"data: {json.dumps({'type':'judge','route':route,'model':model_alias,'judge':judge,'judge_elapsed_ms':judge_elapsed_ms})}\n\n"

                # 4. 準備訊息（系統提示；搜尋改由 LLM 透過工具自行觸發）
                sys_prompt = await get_user_system_prompt(uid)
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
                            yield f"data: {json.dumps({'type':'search'})}\n\n"
                        elif ev["type"] == "final":
                            full_content = ev["content"]
                            answer_input_tokens  = ev["usage"]["input_tokens"]
                            answer_output_tokens = ev["usage"]["output_tokens"]
                    for piece in _chunk_text(full_content):
                        yield f"data: {json.dumps({'type':'token','content':piece})}\n\n"
                else:
                    async with client.stream(
                        "POST",
                        f"{LITELLM_BASE_URL}/v1/chat/completions",
                        headers={"Authorization": f"Bearer {LITELLM_API_KEY}"},
                        json={"model": model_alias, "messages": answer_messages, "max_tokens": MAX_ANSWER_TOKENS,
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
                        JUDGE_MODEL_ALIAS, judge_usage["input_tokens"], judge_usage["output_tokens"],
                        answer_input_tokens, answer_output_tokens,
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
def _any_admin_exists() -> bool:
    """掃描 Firebase Auth，判斷系統是否已有任何管理員。
    用來擋掉「flag 機制上線前就已設定管理員」的既有部署被新使用者搶權。"""
    page = fb_auth.list_users()
    while page:
        for u in page.users:
            if u.custom_claims and u.custom_claims.get("admin"):
                return True
        page = page.get_next_page()
    return False


def _try_claim_first_admin(uid: str) -> bool:
    """Firestore 交易：若 first_admin_flag 尚不存在，原子地建立並回傳 True；已存在則回傳 False。"""
    flag_ref = _db.collection("config").document("first_admin_flag")

    @fb_firestore.transactional
    def _run(tx):
        snap = flag_ref.get(transaction=tx)
        if snap.exists:
            return False
        tx.set(flag_ref, {"uid": uid, "claimed_at": fb_firestore.SERVER_TIMESTAMP})
        return True

    return _run(_db.transaction())


@app.post("/admin/setup")
async def admin_setup(authorization: Optional[str] = Header(None)):
    """第一次使用：若系統中無管理員，授予請求者管理員權限。
    先用 Auth 掃描擋掉既有管理員，再用 Firestore 交易防止兩個「首次」請求同時搶權（TOCTOU 防護）。"""
    if not _firebase_ready:
        raise HTTPException(status_code=503)
    decoded = await decode_token(authorization)
    uid = decoded.get("uid")
    if not uid or uid == "anonymous":
        raise HTTPException(status_code=401)

    # 既有部署可能在此 flag 機制上線前就已有管理員，需先擋掉
    if await asyncio.to_thread(_any_admin_exists):
        raise HTTPException(status_code=403, detail="管理員已存在，請聯絡現有管理員授權")

    claimed = await asyncio.to_thread(_try_claim_first_admin, uid)
    if not claimed:
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
    return await asyncio.to_thread(_fs_get_stats, start_dt, end_dt)


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


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    lang: str = Form("zh"),
    authorization: Optional[str] = Header(None),
):
    await decode_token(authorization)
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="語音轉文字功能未啟用")
    audio_bytes = await file.read()
    if len(audio_bytes) > 25 * 1024 * 1024:
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


@app.get("/health")
async def health():
    return {"status": "ok"}
