"""本機測試用啟動腳本：載入 repo 根目錄的 .env 後啟動 uvicorn。
不是正式部署的一部分，正式環境的環境變數由 Cloud Run 直接注入。
"""
import os
import pathlib

_ENV_PATH = pathlib.Path(__file__).resolve().parent.parent / ".env"

with open(_ENV_PATH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
