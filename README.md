# AI 路由系統 — 前端 + 路由後端 + LiteLLM（目前走雲端，地端通道已預留）

```
瀏覽器（frontend/index.html）
        │
        ▼
路由後端（router_backend，FastAPI）── 負責「難度判斷 + 任務記憶」
        │
        ▼
LiteLLM Proxy（OpenAI 相容介面）
   ├── judge-model  → 雲端，專門判斷難度
   ├── cloud-small-*  → 雲端小模型候選，負責簡單任務的回答
   ├── cloud-medium-* → 雲端中模型候選，負責中等任務的回答
   ├── cloud-large-*  → 雲端大模型候選，負責困難任務的回答
   └── local-*        → 地端模型（Ollama / vLLM），通道已預留、暫未啟用
```

## 1. 後端程式架構（router_backend/）

單一巨大 `app.py` 已拆成職責清楚的模組，`app.py` 只留 FastAPI 組裝與 API 端點：

| 模組 | 職責 |
|---|---|
| `config.py` | 所有環境變數與模型 alias 設定的唯一入口 |
| `schemas.py` | API 請求的 Pydantic 資料模型（含長度驗證） |
| `security.py` | Firebase token 驗證、NTPU 網域限制、速率限制、安全回應標頭 |
| `store.py` | Firebase Auth / Firestore / GCS 存取（同步呼叫都包成 async） |
| `prompts.py` | 系統提示與 judge 提示 |
| `routing.py` | 難度判斷（judge）、級距與模型選擇、地端優先邏輯 |
| `llm.py` | LiteLLM 呼叫封裝（一般／串流／tool-calling 迴圈） |
| `chat.py` | Chat 串流編排（judge → 選模型 → 回答 → 儲存歷史與用量） |
| `memory.py` | 跨對話長期記憶（時間週期壓縮，見第 7 節） |
| `tools.py` | 搜尋工具（Tavily 網路／校內搜尋、北大官網抓取） |
| `attachments.py` | 附件解析（文字、Office、圖片/PDF 轉 base64） |
| `app.py` | FastAPI app 組裝與所有 API 端點 |

模型清單由 `GET /models` 端點對外提供（來源是 `config.py`），前端啟動時會自動同步，
新增模型（含地端 `local-*`）不需要改前端。

## 2. 地端模型通道（已預留）

alias 以 `local-` 開頭即被視為地端模型。之後接 Ollama / vLLM 時：

1. 在 `litellm_config.yaml` 取消「地端模型通道」區塊的註解（已附 Ollama 與 vLLM 範例），
   設定 `OLLAMA_BASE_URL` 或 `VLLM_BASE_URL`。
2. 在 router 後端環境變數填入對應級距的 alias，例如
   `LOCAL_SMALL_MODEL_ALIASES=local-small-llama`。
3. 管理員面板（或 `POST /admin/config`）開啟 `prefer_local`，
   該級距有地端候選時就會優先走地端，資料不出機房。

前端與其餘路由邏輯完全不用動。

## 3. 啟動方式（原生安裝，不使用 Docker）

```bash
# 1) LiteLLM Proxy
pip install 'litellm[proxy]'
export LITELLM_MASTER_KEY=$(python3 -c "import secrets; print('sk-'+secrets.token_urlsafe(24))")
export ANTHROPIC_API_KEY=sk-ant-xxxxx      # 換成你自己的金鑰
export GEMINI_API_KEY=xxxxx                # Gemini 回答模型需要
export OPENROUTER_API_KEY=sk-or-xxxxx      # judge-model（Mistral Small 4）與 OpenRouter 候選模型需要
litellm --config litellm_config.yaml --port 4000 &

# 2) 路由後端
cd router_backend
pip install -r requirements.txt
export LITELLM_BASE_URL=http://localhost:4000
# LITELLM_MASTER_KEY 沿用上面 export 的值
uvicorn app:app --host 0.0.0.0 --port 8000 &
```

啟動後：

- LiteLLM Proxy：`http://localhost:4000`
- 路由後端：`http://localhost:8000`

確認後端活著：`curl http://localhost:8000/health`，回 `{"status":"ok"}` 就代表正常。

> 也可以用 Docker：先在 repo 根目錄建立 `.env`（至少要有 `LITELLM_MASTER_KEY`），
> 再 `docker compose up -d --build`。compose 已改成金鑰一律從 `.env` 帶入，
> 沒設定會直接啟動失敗，避免用預設金鑰上線。

## 4. 開啟前端

`frontend/index.html` 是純 HTML/JS，不需要任何打包工具：

- 本機測試：直接用瀏覽器打開這個檔案即可。
- 正式給同事使用：把這個檔案放到任何靜態網頁伺服器（nginx、或最簡單用 `python3 -m http.server 8080`）。
- 記得把 `index.html` 裡的 `BASE_URL` 改成路由後端實際的網址。

## 5. 設定檔對照

| 檔案 | 負責什麼 | 何時要改 |
|---|---|---|
| `litellm_config.yaml` | 定義 `judge-model`、各級距候選與地端模型 alias | 換供應商、換模型版本、接地端模型時 |
| `router_backend/config.py` | 所有環境變數的預設值與模型清單 | 加新模型 alias、調限制參數時 |
| `router_backend/routing.py` | 難度判斷與級距選擇邏輯 | 要調整路由準不準、想改判斷邏輯時 |
| `docker-compose.yml` | 服務怎麼啟動、port、環境變數 | 想用 Docker 部署時 |
| `frontend/index.html` | 使用者介面 | 想改介面、改 BASE_URL 時 |

## 6. 模型角色的設計理由

- **judge-model 跟負責回答的模型分開**：判斷難度用一個便宜模型專門做這件事，跟實際回答的
  `cloud-small-*` / `cloud-medium-*` / `cloud-large-*` 候選模型互相獨立，之後想單獨換掉判斷邏輯（例如換成自己訓練的分類器）
  不會影響回答品質。
- **每個級距有多模型候選**：judge 會同時輸出 `route`（small / medium / large）與實際 `model`
  alias。若 judge 選到不存在或不屬於該級距的模型，後端會自動退回該級距第一個候選模型。
- **管理員可強制級距或單一模型**：管理員面板的「強制模型」可以選 small / medium / large
  候選池，也可以直接指定 `cloud-small-gemini`、`cloud-medium-claude` 等單一模型（只影響管理員自己的對話）。
- **每次訊息獨立評分**：目前採用純 judge 評分，每則訊息由 judge 依 `threshold_medium` /
  `threshold_large`（可選 `threshold_tiny`）決定級距，沒有跨輪的黏性/衰減邏輯。
- **AI 判斷式會帶上下文**：`model_classify()` 會把最近幾輪對話內容（`HISTORY_LIMIT`）一起餵給判斷模型。

## 7. 跨對話長期記憶

`memory.py` 讓 AI 能記得使用者跨對話的背景與偏好，而不是每次都從零開始，
同時避免無限制地把完整歷史塞進 prompt 造成 token 浪費：

- **每個 session 各自狀態機，不會互相搶觸發機會**：每個 session 文件
  （`users/{uid}/sessions/{session_id}`）有自己的 `memory_pending_since`
  欄位，代表「這個 session 從什麼時候開始累積了新內容、還沒被折進長期記憶」。
  空值＝乾淨狀態；有值＝正在等待。使用者同時開好幾個對話分頭聊時，每一個都
  會各自被顧到，不會因為在別的 session 講話就被跳過。
- **時間週期觸發，不是每則訊息都算**：某個 session 收到新訊息時：
  - 若目前是乾淨狀態（`memory_pending_since` 為空）：只記下「現在開始算」的
    時間戳記，這一輪不壓縮（狀態 0 → 1，代價很小，只是一次小欄位寫入）。
  - 若已經在等待中、且還沒超過 `MEMORY_COMPRESS_INTERVAL_HOURS`（預設 3
    小時，效仿 Claude 的節奏）：什麼都不做。
  - 若已經在等待中、且超過時間門檻：讀現有摘要 → 用這個 session 最近的內容
    壓縮合併 → 寫回 `memory` 欄位 → 清空 `memory_pending_since`（狀態
    1 → 0），這一步才是真正花錢呼叫 LLM 的地方。
- **背景執行、不卡回應**：跟 `store.log_usage_background()` 一樣是
  fire-and-forget，在 `chat.py` 的 `_persist()` 存完歷史後才觸發。
- **壓縮出來的是摘要，不是逐字紀錄**：只保留跨對話仍然有用的事實與偏好
  （科系、身份、常見需求、回答風格偏好等），長度上限 `MEMORY_MAX_CHARS`
  （預設 2000 字），是所有 session 共用的同一份摘要，每次新對話都會注入
  system prompt。
- **壓縮模型**：正式環境設定 `MEMORY_MODEL_ALIAS=memory-model`（Gemini 2.5
  Flash，繁中摘要品質較穩）；跟高頻的 judge（Mistral Small 4，省錢重點）刻意
  拆開——記憶壓縮每人每 3 小時才跑一次，用量極小，品質優先。未設定時預設
  沿用 `JUDGE_MODEL_ALIAS`。
- **使用者可查看、可清除**：設定頁的「AI 記得關於你的事」對應
  `GET`/`DELETE /user/memory`，NTPU 學生用校園帳號登入，透明度是刻意的設計。

## 8. 安全設計

- **金鑰不落地**：LiteLLM `master_key` 改由 `LITELLM_MASTER_KEY` 環境變數帶入，
  設定檔與 repo 裡不再有金鑰；docker-compose 未設定金鑰會直接啟動失敗。
- **登入與網域限制**：Firebase ID token 驗證，非管理員一律限制 `@gm.ntpu.edu.tw` / `@ms.ntpu.edu.tw`；
  首位管理員以 Firestore 交易防止搶權（TOCTOU 防護）。
- **速率限制**：`/chat/stream` 與 `/upload`、`/transcribe` 有每人每分鐘的滑動視窗限制
  （`CHAT_RATE_LIMIT_PER_MINUTE`、`UPLOAD_RATE_LIMIT_PER_MINUTE`，設 0 停用）。
  目前是單機記憶體版，之後跑多個 instance 要改集中式（例如 Redis）。
- **錯誤訊息不外洩**：串流錯誤與檔案讀取失敗只回一般化訊息，詳細堆疊留在 server log。
- **輸入驗證**：訊息長度上限 10 萬字、上傳 20MB／音檔 25MB、上傳副檔名只允許英數字、
  檔案預覽只能讀自己 `uploads/{uid}/` 底下的檔案、`fetch_ntpu_page` 只接受 ntpu.edu.tw 網域，
  且跟隨轉址後會再驗證一次網域（SSRF 防護）。
- **檔案預覽防 XSS**：`/file-preview` 只以原始 MIME 回傳圖片／音訊／影片／PDF；
  HTML、SVG 與各種文字格式一律改用 `text/plain` 回應，其他未知格式回 `application/octet-stream` 附件下載，
  避免使用者上傳的內容在 API 網域上執行腳本。
- **安全回應標頭**：`X-Content-Type-Options`、`X-Frame-Options`、`Referrer-Policy`、`Permissions-Policy`、
  `Content-Security-Policy`（白名單對應前端實際使用的 CDN 與 Firebase，可用 `CONTENT_SECURITY_POLICY` 環境變數覆寫）、
  `Strict-Transport-Security`（HSTS）。
- **容器強化**：後端映像升級為 `python:3.12-slim`，並以非 root 使用者執行。
- **CORS**：預設 `*` 會在啟動時警告；正式環境請設定 `ALLOWED_ORIGINS`（逗號分隔白名單）。

## 9. 後續可以調整的參數

- `threshold_medium` / `threshold_large`：路由門檻（0-10 分制），預設 4 分以上走中模型、7 分以上走大模型，建議上線後用真實 log 重新校準。
- `threshold_tiny`：選填，低於此分數改走開源小模型（`TINY_MODEL_ALIAS`，未設定時停用）。
- `prefer_local`：地端優先開關（需先設定 `LOCAL_*_MODEL_ALIASES`）。
- Claude 回答模型目前對應為：`cloud-small-claude` = Haiku、`cloud-medium-claude` = Sonnet、`cloud-large-claude` = Opus。
- Gemini 回答模型目前對應為：`cloud-small-gemini` = Flash-Lite、`cloud-medium-gemini` = Flash、`cloud-large-gemini` = Pro。
- 各級距候選由 `SMALL_MODEL_ALIASES`、`MEDIUM_MODEL_ALIASES`、`LARGE_MODEL_ALIASES`
  （地端為 `LOCAL_*_MODEL_ALIASES`）控制；
  例如要加 OpenAI，只要在 `litellm_config.yaml` 新增 `cloud-small-openai`，再把它加入 `SMALL_MODEL_ALIASES`。
- `MEMORY_COMPRESS_INTERVAL_HOURS`：長期記憶壓縮週期（預設 3 小時）。
- `MEMORY_MAX_CHARS`：長期記憶摘要長度上限（預設 2000 字）。
- `MEMORY_MODEL_ALIAS`：長期記憶壓縮用的模型（正式環境設為 `memory-model` = Gemini 2.5 Flash；未設定時沿用 `JUDGE_MODEL_ALIAS`）。

## 10. 已知的簡化（正式上線前建議處理）

- 速率限制是單機記憶體版，多 instance 部署時需換成 Redis 等集中式方案。
- LiteLLM 只用 master key 一把金鑰；若要細緻的存取控制，可改用 LiteLLM 虛擬金鑰。
- 長期記憶壓縮觸發時只把「觸發那一刻」的 session 內容折進去，不同 session 各
  自獨立計時，理論上兩個 session 幾乎同時觸發壓縮時，讀改寫 `memory` 欄位不是
  交易（transaction），較晚寫入的那個可能覆蓋掉較早的結果——實務上發生機率低，
  且下一輪壓縮會自然帶入新內容，影響有限。
- **前端只有一份正本 `frontend/index.html`**：Docker build context 已改為 repo 根目錄，
  Dockerfile 會把 `frontend/index.html` 打包成映像內的 `index.html`，後端以 `FileResponse` serve。
