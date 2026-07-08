"""集中管理所有環境變數與模型設定。

其他模組一律從這裡取設定，不直接讀 os.environ，
方便部署時一眼看出有哪些可調參數。
"""
import logging
import os

logger = logging.getLogger("router")


def _alias_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


# ------------------------------------------------------------------
# LiteLLM 連線
# ------------------------------------------------------------------
LITELLM_BASE_URL = os.environ.get("LITELLM_BASE_URL", "http://localhost:4000")
LITELLM_API_KEY  = os.environ.get("LITELLM_MASTER_KEY", "sk-1234")

if LITELLM_API_KEY == "sk-1234":
    logger.warning("LITELLM_MASTER_KEY 仍是預設值 sk-1234，正式環境請務必更換")

# ------------------------------------------------------------------
# 模型 alias 與級距
# ------------------------------------------------------------------
SMALL_MODEL_ALIAS  = os.environ.get("SMALL_MODEL_ALIAS", "cloud-small-claude")
MEDIUM_MODEL_ALIAS = os.environ.get("MEDIUM_MODEL_ALIAS", "cloud-medium-claude")
LARGE_MODEL_ALIAS  = os.environ.get("LARGE_MODEL_ALIAS", "cloud-large-claude")
JUDGE_MODEL_ALIAS  = os.environ.get("JUDGE_MODEL_ALIAS", "judge-model")
TINY_MODEL_ALIAS   = os.environ.get("TINY_MODEL_ALIAS", "")  # 開源小模型，選填

SMALL_MODEL_ALIASES  = os.environ.get("SMALL_MODEL_ALIASES", f"{SMALL_MODEL_ALIAS},cloud-small-gemini")
MEDIUM_MODEL_ALIASES = os.environ.get("MEDIUM_MODEL_ALIASES", f"{MEDIUM_MODEL_ALIAS},cloud-medium-gemini")
LARGE_MODEL_ALIASES  = os.environ.get("LARGE_MODEL_ALIASES", f"{LARGE_MODEL_ALIAS},cloud-large-gemini")

# 地端模型通道：alias 以 local- 開頭即視為地端模型。
# 之後接 Ollama / vLLM 時，在 litellm_config.yaml 加上對應 model_name，
# 再把 alias 填進下面對應級距的環境變數即可，程式不用改。
# 例：LOCAL_SMALL_MODEL_ALIASES=local-small-llama
LOCAL_MODEL_PREFIX = "local-"
LOCAL_SMALL_MODEL_ALIASES  = os.environ.get("LOCAL_SMALL_MODEL_ALIASES", "")
LOCAL_MEDIUM_MODEL_ALIASES = os.environ.get("LOCAL_MEDIUM_MODEL_ALIASES", "")
LOCAL_LARGE_MODEL_ALIASES  = os.environ.get("LOCAL_LARGE_MODEL_ALIASES", "")


def is_local_alias(alias: str) -> bool:
    return alias.startswith(LOCAL_MODEL_PREFIX)


MODEL_CANDIDATES: dict[str, list[str]] = {
    "small":  _alias_list(SMALL_MODEL_ALIASES)  + _alias_list(LOCAL_SMALL_MODEL_ALIASES),
    "medium": _alias_list(MEDIUM_MODEL_ALIASES) + _alias_list(LOCAL_MEDIUM_MODEL_ALIASES),
    "large":  _alias_list(LARGE_MODEL_ALIASES)  + _alias_list(LOCAL_LARGE_MODEL_ALIASES),
}
if TINY_MODEL_ALIAS:
    MODEL_CANDIDATES["tiny"] = [TINY_MODEL_ALIAS]

MODEL_NOTES = {
    "cloud-small-claude": "Claude Haiku 4.5：快速、省成本，適合簡單問答與短任務",
    "cloud-small-gemini": "Gemini 3.1 Flash-Lite：最快速、成本低，適合大量輕量任務",
    "cloud-small-gpt": "GPT-4o mini（OpenRouter）：便宜快速，適合簡單問答與短任務",
    "cloud-medium-claude": "Claude Sonnet 5：品質穩定，適合一般推理、寫作與程式任務",
    "cloud-medium-gemini": "Gemini 3.5 Flash：低延遲且能力均衡，適合中等複雜任務",
    "cloud-medium-gemma": "Gemma 4 27B（OpenRouter）：Google 開源模型，高 CP 值，通用推理與程式能力均衡，適合中等任務",
    "cloud-medium-llama": "Llama 3.3 70B（OpenRouter）：開源模型、成本低，適合中等難度的一般任務",
    "cloud-medium-nemotron": "Llama 3.3 Nemotron Super 49B（NVIDIA）：NVIDIA 調校，推理與指令遵循佳，支援工具呼叫，適合中等任務",
    "cloud-large-claude": "Claude Opus 4.8：高品質深度推理、長任務與複雜 coding",
    "cloud-large-gemini": "Gemini 2.5 Pro：深度推理與 coding，適合複雜任務",
}

MODEL_TO_ROUTE = {
    alias: route
    for route, aliases in MODEL_CANDIDATES.items()
    for alias in aliases
}

# 不支援 function calling 的模型；啟用搜尋工具時要避開，否則工具呼叫會失敗。
# 目前候選模型清單中沒有已知不支援的模型，之後加入新模型若不支援 function
# calling，直接用環境變數列出其 alias（逗號分隔）即可。
NO_TOOL_MODELS = set(_alias_list(os.environ.get("NO_TOOL_MODELS", "")))

# ------------------------------------------------------------------
# 其他服務金鑰 / 參數
# ------------------------------------------------------------------
GCS_BUCKET     = os.environ.get("GCS_BUCKET", "ntpu-ai-uploads")
TAVILY_KEY     = os.environ.get("TAVILY_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
FIREBASE_SERVICE_ACCOUNT_B64 = os.environ.get("FIREBASE_SERVICE_ACCOUNT_B64", "")

HISTORY_LIMIT        = 10
UPLOAD_MAX_BYTES     = 20 * 1024 * 1024   # 20 MB
TRANSCRIBE_MAX_BYTES = 25 * 1024 * 1024   # 25 MB（Whisper 上限）
MAX_ANSWER_TOKENS    = 64000

# ------------------------------------------------------------------
# 安全設定
# ------------------------------------------------------------------
ALLOWED_DOMAINS = {"gm.ntpu.edu.tw", "ms.ntpu.edu.tw"}

CORS_ORIGINS = _alias_list(os.environ.get("ALLOWED_ORIGINS", "")) or ["*"]
if CORS_ORIGINS == ["*"]:
    logger.warning("CORS 未設定 ALLOWED_ORIGINS，目前允許所有來源；正式環境請設定白名單")

# 每位使用者的速率限制（滑動視窗，每分鐘請求數）；設 0 表示停用
CHAT_RATE_LIMIT_PER_MINUTE   = int(os.environ.get("CHAT_RATE_LIMIT_PER_MINUTE", "20"))
UPLOAD_RATE_LIMIT_PER_MINUTE = int(os.environ.get("UPLOAD_RATE_LIMIT_PER_MINUTE", "10"))
