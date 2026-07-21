"""安全相關：Token 驗證、網域限制、速率限制、安全回應標頭。"""
import asyncio
import os
import time
from collections import defaultdict, deque
from typing import Optional

from fastapi import Header, HTTPException
from firebase_admin import auth as fb_auth
from starlette.middleware.base import BaseHTTPMiddleware

import store
from config import ALLOWED_DOMAINS

# 本機開發專用旁路：Firebase 未設定（匿名模式）時，允許把自己當管理員
# 測試管理員面板（例如驗證地端模型路由設定）。正式環境一定有 Firebase，
# 此旁路不會生效。
_DEV_FORCE_ADMIN = os.environ.get("DEV_FORCE_ADMIN", "").lower() in ("1", "true", "yes")


# ------------------------------------------------------------------
# Token 驗證
# ------------------------------------------------------------------
async def decode_token(authorization: Optional[str]) -> dict:
    if not store.firebase_ready:
        # uid 維持 anonymous，讓所有「匿名就跳過 Firestore」的既有保護照常生效；
        # DEV_FORCE_ADMIN 只額外授予 admin，用來在本機測試管理員面板
        return {"uid": "anonymous", "email": "", "admin": _DEV_FORCE_ADMIN}
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing auth token")
    try:
        decoded = await asyncio.to_thread(fb_auth.verify_id_token, authorization[7:])
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid auth token")

    # 網域限制套用到所有非管理員，不分登入方式（google / email-link 一視同仁），
    # 避免有人自行用任意信箱繞過「僅限 NTPU 帳號」的限制。
    # 例外：Firebase 裡「本來就已經存在、且先前登入過」的帳號（例如既有的 Google
    # 管理員帳號）不受網域限制，允許繼續登入；只有「第一次登入＝新註冊」才卡網域。
    if not decoded.get("admin"):
        email = decoded.get("email", "")
        domain = email.split("@")[-1] if "@" in email else ""
        if domain not in ALLOWED_DOMAINS and not await _is_pre_existing_account(decoded.get("uid", "")):
            raise HTTPException(
                status_code=403,
                detail="僅限 NTPU 師生帳號（@gm.ntpu.edu.tw / @ms.ntpu.edu.tw）註冊登入",
            )

    return decoded


async def _is_pre_existing_account(uid: str) -> bool:
    """判斷這個 uid 是否「先前就登入過」（非本次才第一次建立的新帳號）。

    Firebase 帳號在建立當下 creation_timestamp == last_sign_in_timestamp；
    之後每次登入 last_sign_in_timestamp 都會更新，兩者不同即代表帳號並非
    這次登入才剛註冊，屬於既有帳號，不受 NTPU 網域限制。
    """
    if not uid:
        return False
    try:
        user_record = await asyncio.to_thread(fb_auth.get_user, uid)
    except Exception:
        return False
    meta = user_record.user_metadata
    return bool(
        meta.creation_timestamp
        and meta.last_sign_in_timestamp
        and meta.last_sign_in_timestamp > meta.creation_timestamp
    )


async def verify_token(authorization: Optional[str] = Header(None)) -> str:
    return (await decode_token(authorization)).get("uid", "anonymous")


async def require_admin(authorization: Optional[str] = Header(None)) -> str:
    decoded = await decode_token(authorization)
    if not decoded.get("admin"):
        raise HTTPException(status_code=403, detail="Admin only")
    return decoded["uid"]


# ------------------------------------------------------------------
# 速率限制（單機記憶體版滑動視窗，足夠擋住單人狂刷；
# 若之後跑多個 instance，需改用 Redis 等集中式儲存）
# ------------------------------------------------------------------
_PRUNE_THRESHOLD = 5000  # key 數超過此值時清掉閒置者，避免長期執行記憶體無上限


class RateLimiter:
    def __init__(self, limit_per_minute: int):
        self.limit = limit_per_minute
        self._hits: dict[str, deque] = defaultdict(deque)

    def check(self, key: str):
        """超過限制時丟出 429。limit 為 0 表示停用。"""
        if self.limit <= 0:
            return
        now = time.time()
        hits = self._hits[key]
        while hits and now - hits[0] > 60:
            hits.popleft()
        if len(hits) >= self.limit:
            raise HTTPException(status_code=429, detail="請求過於頻繁，請稍後再試")
        hits.append(now)
        if len(self._hits) > _PRUNE_THRESHOLD:
            self._prune(now)

    def _prune(self, now: float):
        stale = [k for k, v in self._hits.items() if not v or now - v[-1] > 60]
        for k in stale:
            del self._hits[k]


# ------------------------------------------------------------------
# 安全回應標頭
# ------------------------------------------------------------------
# CSP 白名單對應前端實際使用的外部資源：
#   script  — cdn.jsdelivr.net（marked）、www.gstatic.com（Firebase SDK）、apis.google.com（Google 登入）
#   connect — Firebase Auth（identitytoolkit / securetoken 等 googleapis）
#   frame   — Google 登入彈窗（*.firebaseapp.com、accounts.google.com）
# 前端為單一 HTML 檔且大量使用 inline script/style，故保留 'unsafe-inline'。
_DEFAULT_CSP = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://www.gstatic.com https://apis.google.com; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: blob: https:; "
    "media-src 'self' data: blob:; "
    "connect-src 'self' https://*.googleapis.com https://*.firebaseapp.com https://www.gstatic.com; "
    "frame-src https://*.firebaseapp.com https://accounts.google.com; "
    "object-src 'none'; base-uri 'self'; form-action 'self'"
)
CSP_POLICY = os.environ.get("CONTENT_SECURITY_POLICY", _DEFAULT_CSP)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(self), geolocation=()")
        response.headers.setdefault("Content-Security-Policy", CSP_POLICY)
        # 瀏覽器只在 HTTPS 回應中採用 HSTS，本機 HTTP 開發不受影響
        response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        return response
