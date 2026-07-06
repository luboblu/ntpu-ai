# Build context 是 repo 根目錄（見 cloudbuild.router.yaml 與 docker-compose.yml），
# 讓後端與唯一的前端正本 frontend/index.html 一起打包，避免重複維護兩份 index.html。
FROM python:3.11-slim

WORKDIR /app
COPY router_backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY router_backend/app.py .
COPY frontend/index.html ./index.html

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
