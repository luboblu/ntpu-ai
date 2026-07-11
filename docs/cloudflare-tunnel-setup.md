# 地端模型對外連線：Cloudflare Tunnel 設定

目的：讓部署在 GCP Cloud Run 上的 `router_backend` 能連到跑在地端機器
（目前是 `100.103.48.80`，Tailscale 內網位址）的 LiteLLM 服務，
且**不需要**在 GCP 端常駐任何背景程式（Cloud Run 對這種需求不友善）。

作法：在地端機器上跑 `cloudflared`，建立一條「僅出站」的加密隧道到
Cloudflare，並綁定一個子網域。Cloud Run 之後只要對這個公開網域發送
HTTPS 請求即可，完全相容無伺服器架構。

## 為什麼選這個而不是 Tailscale

Cloud Run 沒有常駐 process 的能力，沒辦法讓 `tailscaled` 一直保持連線
（除非用 sidecar + min-instances，會有冷啟動風險與固定費用）。
Cloudflare Tunnel 把「維持連線」這件事完全放在地端機器，GCP 端維持
純無伺服器、零額外設定。

## 前置需求

- 一個已加入 Cloudflare 的網域（免費版即可）
- Cloudflare 帳號登入權限

## 地端機器設定步驟

```bash
# 1. 安裝 cloudflared
winget install --id Cloudflare.cloudflared

# 2. 登入（會開瀏覽器走 OAuth 授權）
cloudflared tunnel login

# 3. 建立 tunnel（僅需一次）
cloudflared tunnel create ntpu-local-llm

# 4. 把子網域路由指向這個 tunnel
cloudflared tunnel route dns ntpu-local-llm llm.<你的網域>.com
```

建立設定檔 `~/.cloudflared/config.yml`：

```yaml
tunnel: ntpu-local-llm
credentials-file: C:\Users\<user>\.cloudflared\<tunnel-id>.json

ingress:
  - hostname: llm.<你的網域>.com
    service: http://localhost:4000
  - service: http_status:404
```

啟動（測試用前景執行，正式用建議註冊成 Windows 服務常駐）：

```bash
cloudflared tunnel run ntpu-local-llm

# 正式部署建議改用服務常駐，開機自動啟動：
cloudflared service install
```

## 安全性：務必加 Cloudflare Access

裸的 `sk-local-1234` 這種弱 key 一旦子網域對外公開，很容易被掃描到。
建議在 Cloudflare Zero Trust 後台（Access → Applications）替
`llm.<你的網域>.com` 加一層 Service Token 驗證，只有帶正確 Service
Token 的請求才能通過 Cloudflare 邊緣，LiteLLM 自己的 API key 驗證
變成第二層防護（縱深防禦）。

## router_backend 端要改的設定

不需要改程式碼，只要把 litellm_config.yaml 或環境變數裡原本指向
`http://100.103.48.80:4000` 的位址，換成 Cloudflare Tunnel 給的
公開網址 `https://llm.<你的網域>.com`，並在 headers 帶上
Cloudflare Access 的 Service Token（如果有啟用）。

## 驗證清單

- [ ] 地端機器 `cloudflared tunnel run` 正常且無報錯
- [ ] 從任意外部網路 `curl https://llm.<你的網域>.com/v1/models` 能拿到 200
- [ ] 若啟用 Cloudflare Access，未帶 Service Token 時應該被拒絕（驗證有生效）
- [ ] `router_backend` 從 GCP 環境對這個網址發請求能正常拿到回應
