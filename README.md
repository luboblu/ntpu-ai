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
   ├── cloud-small  → 雲端小模型，負責簡單任務的回答
   └── cloud-large  → 雲端大模型，負責困難任務的回答
```

地端開源模型（Ollama／vLLM）先不接，之後資料隱私需求確定後再加進來，
到時只要在 `litellm_config.yaml` 多加一個 model_name、在 `app.py` 多一個路由判斷分支即可，
不需要動前端跟其他邏輯。

## 1. 啟動方式（原生安裝，不使用 Docker）

```bash
# 1) LiteLLM Proxy
pip install 'litellm[proxy]'
export ANTHROPIC_API_KEY=sk-ant-xxxxx      # 換成你自己的金鑰
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
| `litellm_config.yaml` | 定義「judge-model」「cloud-small」「cloud-large」三個別名實際對應哪個模型 | 換供應商、換模型版本時 |
| `router_backend/app.py` | 難度判斷邏輯（AI 判斷式 prompt）、任務累積分數的衰減速度 | 要調整路由準不準、想改判斷邏輯時 |
| `docker-compose.yml` | 服務怎麼啟動、port、環境變數 | 想用 Docker 部署時 |
| `frontend/index.html` | 使用者介面 | 想改介面、改 BASE_URL 時 |

## 4. 三個模型角色的設計理由

- **judge-model 跟負責回答的模型分開**：判斷難度用一個便宜模型專門做這件事，跟實際回答的
  `cloud-small` / `cloud-large` 互相獨立，之後想單獨換掉判斷邏輯（例如換成自己訓練的分類器）
  不會影響回答品質。
- **任務累積難度（黏性路由）**：`app.py` 裡 `DECAY_PER_TURN = 0.45`，分數用
  `max(本次正規化分數, 上一輪分數 - 衰減值)` 計算，不會因為單句話變簡單就立刻降級。
- **AI 判斷式會帶上下文**：`model_classify()` 會把上一輪對話內容一起餵給判斷模型。

## 5. 後續可以調整的參數

- `MODEL_THRESHOLD`：路由門檻（0-10 分制），建議上線後用真實 log 重新校準。
- `DECAY_PER_TURN`：數字越小，黏性越強（越不容易降回小模型）。
- 三個模型角色都可以換成不同供應商（例如 `cloud-small` 換成 OpenAI 的 `gpt-4o-mini`），
  只要改 `litellm_config.yaml`，`app.py` 完全不用動——這就是 LiteLLM 統一介面的好處。

## 6. 已知的簡化（正式上線前建議處理）

- `_session_scores` 用記憶體字典存放，重啟後端會清空；多台後端水平擴展時請換成 Redis。
- LiteLLM 的 `master_key` 寫在 `litellm_config.yaml` 裡只是 demo 用，正式環境請用環境變數帶入、加上虛擬金鑰做存取控制。
- CORS 目前開放 `*`，正式環境請改成資訊中心內網的白名單網域。
