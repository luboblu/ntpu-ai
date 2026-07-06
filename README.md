# AI 路由系統 — 前端 + 路由後端 + LiteLLM（目前先全部走雲端）

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
   └── cloud-large-*  → 雲端大模型候選，負責困難任務的回答
```

地端開源模型（Ollama／vLLM）先不接，之後資料隱私需求確定後再加進來，
到時只要在 `litellm_config.yaml` 多加一個 model_name、在 `app.py` 多一個路由判斷分支即可；
目前雲端回答模型已經有 small / medium / large 三層，
不需要動前端跟其他邏輯。

## 1. 啟動方式（原生安裝，不使用 Docker）

```bash
# 1) LiteLLM Proxy
pip install 'litellm[proxy]'
export ANTHROPIC_API_KEY=sk-ant-xxxxx      # 換成你自己的金鑰
export GEMINI_API_KEY=xxxxx                # judge-model 仍使用 Gemini 時需要
litellm --config litellm_config.yaml --port 4000 &

# 2) 路由後端
cd router_backend
pip install -r requirements.txt
export LITELLM_BASE_URL=http://localhost:4000
export LITELLM_MASTER_KEY=sk-1234
uvicorn app:app --host 0.0.0.0 --port 8000 &
```

啟動後：

- LiteLLM Proxy：`http://localhost:4000`
- 路由後端：`http://localhost:8000`

確認後端活著：`curl http://localhost:8000/health`，回 `{"status":"ok"}` 就代表正常。

> 也可以用 Docker：`docker-compose.yml` 跟 `router_backend/Dockerfile` 都還在，
> 直接 `docker compose up -d --build` 即可，不需要額外改設定
> （這個版本已經不需要 Ollama，所以 compose 檔裡也拿掉了那個服務）。

## 2. 開啟前端

`frontend/index.html` 是純 HTML/JS，不需要任何打包工具：

- 本機測試：直接用瀏覽器打開這個檔案即可。
- 正式給同事使用：把這個檔案放到任何靜態網頁伺服器（nginx、或最簡單用 `python3 -m http.server 8080`）。
- 記得把 `index.html` 裡的 `BASE_URL` 改成路由後端實際的網址。

## 3. 設定檔對照

| 檔案 | 負責什麼 | 何時要改 |
|---|---|---|
| `litellm_config.yaml` | 定義 `judge-model` 與各級距候選模型 alias，例如 `cloud-small-claude` / `cloud-small-gemini` | 換供應商、換模型版本時 |
| `router_backend/app.py` | 難度判斷邏輯（judge 的 AI 判斷式 prompt）、級距門檻與工具（搜尋）呼叫 | 要調整路由準不準、想改判斷邏輯時 |
| `docker-compose.yml` | 服務怎麼啟動、port、環境變數 | 想用 Docker 部署時 |
| `frontend/index.html` | 使用者介面 | 想改介面、改 BASE_URL 時 |

## 4. 模型角色的設計理由

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

## 5. 後續可以調整的參數

- `threshold_medium` / `threshold_large`：路由門檻（0-10 分制），預設 4 分以上走中模型、7 分以上走大模型，建議上線後用真實 log 重新校準。
- `threshold_tiny`：選填，低於此分數改走開源小模型（`TINY_MODEL_ALIAS`，未設定時停用）。
- Claude 回答模型目前對應為：`cloud-small-claude` = Haiku、`cloud-medium-claude` = Sonnet、`cloud-large-claude` = Opus。
- Gemini 回答模型目前對應為：`cloud-small-gemini` = Flash-Lite、`cloud-medium-gemini` = Flash、`cloud-large-gemini` = Pro。
- 各級距候選由 `SMALL_MODEL_ALIASES`、`MEDIUM_MODEL_ALIASES`、`LARGE_MODEL_ALIASES` 控制；
  例如要加 OpenAI，只要在 `litellm_config.yaml` 新增 `cloud-small-openai`，再把它加入 `SMALL_MODEL_ALIASES`。

## 6. 已知的簡化（正式上線前建議處理）

- LiteLLM 的 `master_key` 寫在 `litellm_config.yaml` 裡只是 demo 用，正式環境請用環境變數帶入、加上虛擬金鑰做存取控制。
- CORS 預設仍是 `*`，正式環境請設定 `ALLOWED_ORIGINS` 環境變數（逗號分隔的白名單網域）收斂。
- **前端只有一份正本 `frontend/index.html`**：Docker build context 已改為 repo 根目錄，
  Dockerfile 會把 `frontend/index.html` 打包成映像內的 `index.html`，後端以 `FileResponse` serve。
  （之前曾有 `router_backend/index.html` 副本容易忘記同步，現已移除。）
