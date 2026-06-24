"""
路由後端：負責「判斷難度→決定路由→呼叫 LiteLLM」這一段。
目前先全部走雲端、不碰地端：用一個雲端模型專門判斷難度，
另外用一個雲端小模型、一個雲端大模型負責實際回答。

啟動方式（本機測試）：
    pip install -r requirements.txt
    export LITELLM_BASE_URL=http://localhost:4000
    export LITELLM_MASTER_KEY=sk-1234
    uvicorn app:app --reload --port 8000
"""

import os
import json
import time
from typing import Optional, Dict

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ------------------------------------------------------------------
# 設定：全部用環境變數帶入，方便在 docker-compose 裡覆寫
# ------------------------------------------------------------------
LITELLM_BASE_URL = os.environ.get("LITELLM_BASE_URL", "http://localhost:4000")  # 原生啟動時 LiteLLM 跑在本機，所以是 localhost
LITELLM_API_KEY = os.environ.get("LITELLM_MASTER_KEY", "sk-1234")

SMALL_MODEL_ALIAS = os.environ.get("SMALL_MODEL_ALIAS", "cloud-small")   # 對應 litellm_config.yaml 裡的 model_name
LARGE_MODEL_ALIAS = os.environ.get("LARGE_MODEL_ALIAS", "cloud-large")   # 同上
JUDGE_MODEL_ALIAS = os.environ.get("JUDGE_MODEL_ALIAS", "judge-model")  # 專門判斷難度的模型，跟負責回答的兩個模型是分開的

MODEL_THRESHOLD = 6.0       # AI 判斷式採 0-10 分制
DECAY_PER_TURN = 0.45       # 任務累積分數每輪最多衰減多少（升級容易、降級要熬幾輪）

app = FastAPI(title="AI Router Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 內部 demo 用；正式上線請改成白名單，例如資訊中心內網網域
    allow_methods=["*"],
    allow_headers=["*"],
)

# 任務層級的黏性分數，用 session_id 區分不同對話。
# demo 用記憶體字典即可；正式環境多台後端水平擴展時，建議換成 Redis 共享狀態。
_session_scores: Dict[str, float] = {}
_session_context: Dict[str, Dict[str, str]] = {}


class ChatRequest(BaseModel):
    session_id: str
    message: str


# ------------------------------------------------------------------
# 呼叫 LiteLLM（OpenAI 相容的 /v1/chat/completions）
# ------------------------------------------------------------------
async def call_litellm(client: httpx.AsyncClient, model_alias: str, messages: list, max_tokens: int = 4096) -> str:
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
    return content


# ------------------------------------------------------------------
# AI 判斷式：額外打一次模型評估難度，並把上一輪對話脈絡也帶進去
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


async def model_classify(client: httpx.AsyncClient, text: str, context_summary: Optional[str]) -> dict:
    context_block = (
        f'對話脈絡：\n"""\n{context_summary}\n"""\n\n'
        if context_summary else ""
    )
    user_message = f"{context_block}請評估以下訊息的難度：\n```\n{text}\n```"
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]
    raw = await call_litellm(client, JUDGE_MODEL_ALIAS, messages)
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(cleaned)
        score = max(0.0, min(10.0, float(parsed.get("score", 5))))
        return {"score": score, "reason": parsed.get("reason", ""), "normalized": score / MODEL_THRESHOLD}
    except Exception:
        return {
            "score": 5.0,
            "reason": "模型回應無法解析為 JSON，採用預設中等難度",
            "normalized": 5.0 / MODEL_THRESHOLD,
            "parse_failed": True,
        }


# ------------------------------------------------------------------
# API
# ------------------------------------------------------------------
@app.get("/")
async def index():
    return FileResponse("index.html")


@app.post("/chat")
async def chat(req: ChatRequest):
    async with httpx.AsyncClient() as client:
        t0 = time.time()
        ctx = _session_context.get(req.session_id)
        context_summary = (
            f"上一輪使用者：{ctx['user']}\n上一輪回答摘要：{ctx['assistant'][:200]}" if ctx else None
        )
        judge = await model_classify(client, req.message, context_summary)
        judge_elapsed_ms = int((time.time() - t0) * 1000)

        # 任務層級的黏性分數：本次正規化分數 vs. 上一輪衰減後的分數，取較大值
        prev_score = _session_scores.get(req.session_id, 0.0)
        decayed = max(0.0, prev_score - DECAY_PER_TURN)
        session_score = max(judge["normalized"], decayed)
        route = "large" if session_score >= 1.0 else "small"
        model_alias = LARGE_MODEL_ALIAS if route == "large" else SMALL_MODEL_ALIAS

        t1 = time.time()
        answer = await call_litellm(client, model_alias, [{"role": "user", "content": req.message}])
        answer_elapsed_ms = int((time.time() - t1) * 1000)

    _session_scores[req.session_id] = session_score
    _session_context[req.session_id] = {"user": req.message, "assistant": answer}

    return {
        "route": route,
        "model": model_alias,
        "judge": judge,
        "session_score": session_score,
        "judge_elapsed_ms": judge_elapsed_ms,
        "answer_elapsed_ms": answer_elapsed_ms,
        "answer": answer,
    }


@app.post("/reset")
async def reset(session_id: str):
    """開新任務：清空這個 session 的累積難度記憶。"""
    _session_scores.pop(session_id, None)
    _session_context.pop(session_id, None)
    return {"ok": True}


@app.get("/health")
async def health():
    return {"status": "ok"}
