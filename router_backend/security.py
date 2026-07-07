"""安全相關：Token 驗證、網域限制、速率限制、安全回應標頭。"""
import asyncio
import time
from collections import defaultdict, deque
from typing import Optional

from fastapi import Header, HTTPException
from firebase_admin import auth as fb_auth
from starlette.middleware.base import BaseHTTPMiddleware

import store
from config import ALLOWED_DOMAINS


# ------------------------------------------------------------------
# Token 驗證
# ------------------------------------------------------------------
async def decode_token(authorization: Optional[str]) -> dict:
    if not store.firebase_ready:
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
# 速率限制（單機記憶體版滑動視窗，足夠擋住單人狂刷；
# 若之後跑多個 instance，需改用 Redis 等集中式儲存）
# ------------------------------------------------------------------
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


# ------------------------------------------------------------------
# 安全回應標頭
# ------------------------------------------------------------------
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault("Permissions-Policy", "camera=(), geolocation=()")
        return response
