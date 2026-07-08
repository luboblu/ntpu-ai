"""跨對話長期記憶：每個 session 各自追蹤「有新內容待壓縮」的狀態與時間，
獨立判斷是否該把自己的內容折進使用者的長期記憶摘要，不會因為使用者在別的
session 講話、搶走觸發機會而被跳過。

狀態機（存在對應 session 文件的 memory_pending_since 欄位）：
  沒有這個欄位（None）── 目前沒有待處理的新內容（乾淨狀態）。
  有這個欄位（timestamp）── 從這個時間點起開始累積新內容，等
    MEMORY_COMPRESS_INTERVAL_HOURS 到期後才會被壓縮進使用者的長期記憶，
    壓縮完這個欄位會被清掉，回到乾淨狀態。

跟 store.log_usage_background 一樣是背景 fire-and-forget，不影響回應速度。
狀態 0→1 只是記一個時間戳記，代價很小；真正花錢的壓縮呼叫只在狀態 1→0
（也就是真的到期）時才會執行。
"""
import asyncio
import datetime
import logging

import httpx

import store
from config import MEMORY_COMPRESS_INTERVAL_HOURS, MEMORY_MAX_CHARS, MEMORY_MODEL_ALIAS
from llm import call_litellm
from prompts import MEMORY_COMPRESS_PROMPT

logger = logging.getLogger("router.memory")

# 壓縮時最多帶入這個 session 最近幾則訊息，避免極端的超長對話讓單次壓縮成本失控
_MAX_INPUT_MESSAGES = 40

# 追蹤背景任務，避免還在跑的 task 被 GC 提早回收（這個任務要跑一次 LLM 往返，
# 活得比一般 fire-and-forget 任務久，比較容易被 GC 中斷）
_running_tasks: set[asyncio.Task] = set()


def maybe_compress_background(uid: str, session_id: str, session_history: list) -> None:
    if uid == "anonymous":
        return
    task = asyncio.create_task(_check_and_compress(uid, session_id, session_history))
    _running_tasks.add(task)
    task.add_done_callback(_running_tasks.discard)


async def _check_and_compress(uid: str, session_id: str, session_history: list) -> None:
    try:
        pending_since = await store.get_session_memory_pending_since(uid, session_id)
        now = datetime.datetime.now(tz=datetime.timezone.utc)

        if pending_since is None:
            # 狀態 0 → 1：這個 session 剛開始累積新內容，記下起點，這輪先不壓縮
            await store.set_session_memory_pending(uid, session_id, now)
            return

        if now - pending_since < datetime.timedelta(hours=MEMORY_COMPRESS_INTERVAL_HOURS):
            return  # 還在等待期限內，什麼都不做

        profile = await store.get_user_profile(uid)
        await _compress(uid, profile, session_history)
        # 狀態 1 → 0：這個 session 待處理的內容已經折進長期記憶
        await store.set_session_memory_pending(uid, session_id, None)
    except Exception:
        # 失敗就跳過這一輪，memory_pending_since 維持原樣（還是「待處理」），
        # 下次這個 session 有新訊息時會立刻再試一次；不讓例外往上炸
        logger.exception("長期記憶狀態檢查/壓縮失敗 uid=%s session=%s", uid, session_id)


async def _compress(uid: str, profile: dict, session_history: list) -> None:
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
