"""跨對話長期記憶：每隔固定時間，用目前 session 內容壓縮更新一次使用者摘要。

跟 store.log_usage_background 一樣是背景 fire-and-forget，不影響回應速度。
平常每則訊息完全不寫入 memory 相關欄位，只有距離上次壓縮超過
MEMORY_COMPRESS_INTERVAL_HOURS 才會觸發一次讀取現有摘要 + 壓縮 + 寫回，
避免像逐輪累積 buffer 那樣每則訊息都要讀改寫 Firestore。
"""
import asyncio
import datetime
import logging

import httpx

import store
from config import (
    MEMORY_COMPRESS_INTERVAL_HOURS,
    MEMORY_MAX_CHARS,
    MEMORY_MIN_MESSAGES_FOR_FIRST_COMPRESS,
    MEMORY_MODEL_ALIAS,
)
from llm import call_litellm
from prompts import MEMORY_COMPRESS_PROMPT

logger = logging.getLogger("router.memory")

# 壓縮時最多帶入目前 session 最近幾則訊息，避免極端的超長對話讓單次壓縮成本失控
_MAX_INPUT_MESSAGES = 40

# 追蹤背景任務，避免還在跑的 task 被 GC 提早回收（這個任務要跑一次 LLM 往返，
# 活得比一般 fire-and-forget 任務久，比較容易被 GC 中斷）
_running_tasks: set[asyncio.Task] = set()


def maybe_compress_background(uid: str, profile: dict, session_history: list) -> None:
    """未達壓縮條件時直接跳過；達到才背景觸發，不阻塞回應。"""
    if uid == "anonymous":
        return
    if not _should_compress(profile, session_history):
        return
    task = asyncio.create_task(_compress(uid, profile, session_history))
    _running_tasks.add(task)
    task.add_done_callback(_running_tasks.discard)


def _should_compress(profile: dict, session_history: list) -> bool:
    updated_at = profile.get("memory_updated_at")
    if updated_at is None:
        # 從沒壓縮過：至少要有一定內容才值得壓縮，避免帳號剛建立、
        # 只講一兩句話就急著壓縮出空洞的摘要
        return len(session_history) >= MEMORY_MIN_MESSAGES_FOR_FIRST_COMPRESS
    age = datetime.datetime.now(tz=datetime.timezone.utc) - updated_at
    return age >= datetime.timedelta(hours=MEMORY_COMPRESS_INTERVAL_HOURS)


async def _compress(uid: str, profile: dict, session_history: list) -> None:
    try:
        recent = session_history[-_MAX_INPUT_MESSAGES:]
        turns_text = "\n\n".join(
            f"{'使用者' if m['role'] == 'user' else 'AI'}：{store.content_str(m['content'])[:1000]}"
            for m in recent
        )
        existing_memory = profile.get("memory", "")
        messages = [
            {"role": "system", "content": MEMORY_COMPRESS_PROMPT.format(max_chars=MEMORY_MAX_CHARS)},
            {"role": "user", "content": f"既有長期記憶（可能為空）：\n{existing_memory or '（無）'}\n\n"
                                        f"這位使用者最近一段對話：\n{turns_text}"},
        ]
        # 不可借用 chat.py 那個 request-scoped httpx client——這個背景任務
        # 真正執行時，該次 request 很可能已經結束、client 已經被關閉。
        async with httpx.AsyncClient() as client:
            content, _usage = await call_litellm(client, MEMORY_MODEL_ALIAS, messages, max_tokens=800)
        await store.set_user_memory(uid, content.strip()[:MEMORY_MAX_CHARS])
    except Exception:
        # 失敗就跳過這一輪，memory_updated_at 維持原樣，等下次自然機會再重試；
        # 不讓例外往上炸，也不要寫入半殘的結果
        logger.exception("長期記憶壓縮失敗 uid=%s", uid)
