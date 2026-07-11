"""外部儲存層：Firebase Auth / Firestore / Google Cloud Storage。

所有 Firestore 同步呼叫都以 `_fs_` 開頭，由對應的 async 包裝函式
用 asyncio.to_thread 呼叫，避免阻塞 event loop。
Firebase 未設定時全部降級為 no-op（匿名模式），方便本機開發。
"""
import asyncio
import base64
import datetime
import json
import logging
import secrets

import firebase_admin
from firebase_admin import auth as fb_auth
from firebase_admin import credentials
from firebase_admin import firestore as fb_firestore

from config import FIREBASE_SERVICE_ACCOUNT_B64, GCS_BUCKET

logger = logging.getLogger("router.store")

# ------------------------------------------------------------------
# Google Cloud Storage
# ------------------------------------------------------------------
try:
    from google.cloud import storage as _gcs_lib
    _gcs_client = _gcs_lib.Client()
    gcs_ready = True
except Exception:
    _gcs_client = None
    gcs_ready = False
    logger.warning("GCS 未設定，檔案上傳功能停用")

# ------------------------------------------------------------------
# Firebase Admin
# ------------------------------------------------------------------
if FIREBASE_SERVICE_ACCOUNT_B64:
    _sa_dict = json.loads(base64.b64decode(FIREBASE_SERVICE_ACCOUNT_B64).decode())
    firebase_admin.initialize_app(credentials.Certificate(_sa_dict))
    _db = fb_firestore.client()
    firebase_ready = True
else:
    _db = None
    firebase_ready = False
    logger.warning("Firebase 未設定，以匿名模式運作（不驗證登入、不保存歷史）")


# ------------------------------------------------------------------
# GCS helpers
# ------------------------------------------------------------------
async def gcs_download(path: str) -> tuple[bytes, str]:
    """下載 GCS 物件，回傳 (bytes, content_type)。"""
    blob = _gcs_client.bucket(GCS_BUCKET).blob(path)
    data = await asyncio.to_thread(blob.download_as_bytes)
    return data, (blob.content_type or "application/octet-stream").lower()


async def gcs_upload(path: str, content: bytes, content_type: str):
    blob = _gcs_client.bucket(GCS_BUCKET).blob(path)
    await asyncio.to_thread(blob.upload_from_string, content, content_type)


# ------------------------------------------------------------------
# 共用小工具
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


def content_str(c) -> str:
    """從 message content（str 或 multimodal list）取出純文字部分。"""
    if isinstance(c, list):
        return next((p.get("text", "") for p in c if p.get("type") == "text"), "")
    return c or ""


# ------------------------------------------------------------------
# Routing config（config/routing 文件）
# ------------------------------------------------------------------
def _fs_get_routing_config(default: dict) -> dict:
    doc = _db.collection("config").document("routing").get()
    return doc.to_dict() if doc.exists else dict(default)


def _fs_set_routing_config(data: dict):
    _db.collection("config").document("routing").set(data)


async def fetch_routing_config(default: dict) -> dict:
    return await asyncio.to_thread(_fs_get_routing_config, default)


async def save_routing_config(data: dict):
    await asyncio.to_thread(_fs_set_routing_config, data)


# ------------------------------------------------------------------
# 對話歷史（users/{uid}/sessions/{session_id}）
# ------------------------------------------------------------------
def _fs_get_history(uid: str, session_id: str) -> list:
    doc = _db.collection("users").document(uid).collection("sessions").document(session_id).get()
    return doc.to_dict().get("history", []) if doc.exists else []


def _fs_save_history(uid: str, session_id: str, history: list):
    title = next((content_str(m["content"])[:60] for m in history if m["role"] == "user"), "對話")
    _db.collection("users").document(uid).collection("sessions").document(session_id).set({
        "history": history,
        "title": title,
        "updated_at": fb_firestore.SERVER_TIMESTAMP,
    })


def _fs_delete_session(uid: str, session_id: str):
    _db.collection("users").document(uid).collection("sessions").document(session_id).delete()


def _fs_list_sessions(uid: str) -> list:
    docs = (
        _db.collection("users").document(uid).collection("sessions")
        .order_by("updated_at", direction="DESCENDING").limit(50).stream()
    )
    return [{
        "session_id": d.id,
        "title": d.to_dict().get("title", "對話"),
        "updated_at": _ts_to_str(d.to_dict().get("updated_at")),
    } for d in docs]


def _fs_get_session_data(uid: str, session_id: str) -> dict:
    doc = _db.collection("users").document(uid).collection("sessions").document(session_id).get()
    return doc.to_dict() if doc.exists else {}


async def get_history(uid: str, session_id: str) -> list:
    if not firebase_ready or uid == "anonymous":
        return []
    return await asyncio.to_thread(_fs_get_history, uid, session_id)


async def save_history(uid: str, session_id: str, history: list):
    if not firebase_ready or uid == "anonymous":
        return
    await asyncio.to_thread(_fs_save_history, uid, session_id, history)


async def delete_session(uid: str, session_id: str):
    if not firebase_ready or uid == "anonymous":
        return
    await asyncio.to_thread(_fs_delete_session, uid, session_id)


async def list_sessions(uid: str) -> list:
    return await asyncio.to_thread(_fs_list_sessions, uid)


async def get_session_data(uid: str, session_id: str) -> dict:
    return await asyncio.to_thread(_fs_get_session_data, uid, session_id)


# ------------------------------------------------------------------
# 長期記憶：每個 session 各自的「待壓縮」狀態（見 memory.py）
# ------------------------------------------------------------------
def _fs_get_session_memory_pending_since(uid: str, session_id: str):
    doc = _db.collection("users").document(uid).collection("sessions").document(session_id).get()
    return doc.to_dict().get("memory_pending_since") if doc.exists else None


def _fs_set_session_memory_pending(uid: str, session_id: str, pending_since):
    _db.collection("users").document(uid).collection("sessions").document(session_id).set(
        {"memory_pending_since": pending_since}, merge=True)


async def get_session_memory_pending_since(uid: str, session_id: str):
    if not firebase_ready or uid == "anonymous":
        return None
    return await asyncio.to_thread(_fs_get_session_memory_pending_since, uid, session_id)


async def set_session_memory_pending(uid: str, session_id: str, pending_since):
    await asyncio.to_thread(_fs_set_session_memory_pending, uid, session_id, pending_since)


# ------------------------------------------------------------------
# 使用者 profile
# ------------------------------------------------------------------
def _fs_get_user_profile(uid: str) -> dict:
    doc = _db.collection("users").document(uid).get()
    return (doc.to_dict() or {}) if doc.exists else {}


def _fs_set_user_profile(uid: str, data: dict):
    _db.collection("users").document(uid).set(data, merge=True)


async def get_user_profile(uid: str) -> dict:
    return await asyncio.to_thread(_fs_get_user_profile, uid)


async def set_user_profile(uid: str, data: dict):
    await asyncio.to_thread(_fs_set_user_profile, uid, data)


async def get_user_profile_fields(uid: str) -> dict:
    """回傳這位使用者的 profile doc（含 system_prompt、memory、
    memory_updated_at 等欄位）；未設定 Firebase 或匿名一律回傳空 dict。
    一次讀取整份文件，避免呼叫端各自讀一次同一份 doc。"""
    if not firebase_ready or uid == "anonymous":
        return {}
    try:
        return await get_user_profile(uid)
    except Exception:
        logger.exception("讀取使用者 profile 失敗")
        return {}


def _fs_set_user_memory(uid: str, memory: str):
    _db.collection("users").document(uid).set(
        {"memory": memory, "memory_updated_at": fb_firestore.SERVER_TIMESTAMP},
        merge=True)


async def set_user_memory(uid: str, memory: str):
    await asyncio.to_thread(_fs_set_user_memory, uid, memory)


# ------------------------------------------------------------------
# 公開分享（public_shares/{share_id}）
# ------------------------------------------------------------------
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


async def create_share(uid: str, session_id: str, history: list, title: str) -> str:
    return await asyncio.to_thread(_fs_create_share, uid, session_id, history, title)


async def get_share(share_id: str) -> dict:
    return await asyncio.to_thread(_fs_get_share, share_id)


# ------------------------------------------------------------------
# 用量紀錄與統計（usage_logs）
# ------------------------------------------------------------------
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


def log_usage_background(*args, **kwargs):
    """在背景 thread 寫入用量 log，不阻塞回應。"""
    asyncio.create_task(asyncio.to_thread(_fs_log_usage, *args, **kwargs))


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


async def get_stats(start_dt: datetime.datetime, end_dt: datetime.datetime) -> dict:
    if not firebase_ready:
        return {"users": [], "by_model": []}
    return await asyncio.to_thread(_fs_get_stats, start_dt, end_dt)


# ------------------------------------------------------------------
# Firebase Auth 使用者管理
# ------------------------------------------------------------------
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


async def list_auth_users() -> list:
    if not firebase_ready:
        return []
    return await asyncio.to_thread(_fs_list_auth_users)


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


async def any_admin_exists() -> bool:
    return await asyncio.to_thread(_any_admin_exists)


async def try_claim_first_admin(uid: str) -> bool:
    return await asyncio.to_thread(_try_claim_first_admin, uid)


async def set_admin_claim(uid: str, is_admin: bool):
    claims = {"admin": True} if is_admin else {}
    await asyncio.to_thread(fb_auth.set_custom_user_claims, uid, claims)


async def delete_auth_user(uid: str):
    await asyncio.to_thread(fb_auth.delete_user, uid)
