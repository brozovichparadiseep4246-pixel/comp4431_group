"""集中配置：环境变量、模型名、路径。"""
from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
KB_DOCS_DIR = DATA_DIR / "kb_docs"
COURSES_JSON = DATA_DIR / "courses.json"
INDEX_DIR = BASE_DIR / "vector_index"

load_dotenv(BASE_DIR / ".env")

# ---- 在线大模型 (OpenAI 兼容接口) ----
# 配置优先级:
#   1. .env 显式配置 (LLM_API_KEY + LLM_BASE_URL + LLM_MODEL, 任意 OpenAI 兼容服务,
#      如 Groq 免费API / OpenRouter / Gemini / Moonshot 等)
#   2. 本机 Kimi CLI 的 OAuth token 自动兜底 (开发机自用)
#   3. 都没有 -> 界面切换为"本地微调模型"引擎, 完全离线可用

def _load_kimi_cli_token() -> str | None:
    cred = Path.home() / ".kimi" / "credentials" / "kimi-code.json"
    try:
        token = json.loads(cred.read_text(encoding="utf-8")).get("access_token")
        return token or None
    except Exception:
        return None


LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("MOONSHOT_API_KEY")

if LLM_API_KEY:
    # .env 显式配置 (默认 Groq 免费 API, 见 .env.example)
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
else:
    LLM_API_KEY = _load_kimi_cli_token()
    if LLM_API_KEY:  # Kimi CLI OAuth token 只能用于 coding 端点
        LLM_BASE_URL = "https://api.kimi.com/coding/v1"
        LLM_MODEL = "kimi-for-coding"
    else:  # 无在线模型可用, 仅本地微调引擎
        LLM_BASE_URL = ""
        LLM_MODEL = ""

MOONSHOT_API_KEY = LLM_API_KEY  # 兼容旧引用

# temperature：留空表示不设置（部分服务/模型不允许自定义 temperature）
_temp = os.getenv("LLM_TEMPERATURE")
if _temp:
    LLM_TEMPERATURE = float(_temp)
else:
    LLM_TEMPERATURE = 0.3 if (LLM_API_KEY and "kimi.com" not in LLM_BASE_URL) else None

# ---- 检索 ----
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
TOP_K = int(os.getenv("TOP_K", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.25"))
# 检索最高置信度低于该值时, 视为知识库未命中 -> 网络搜索兜底
WEB_FALLBACK_THRESHOLD = float(os.getenv("WEB_FALLBACK_THRESHOLD", "0.45"))

# ---- 语音 ----
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny")
TTS_VOICE_ZH = os.getenv("TTS_VOICE_ZH", "zh-CN-XiaoxiaoNeural")
TTS_VOICE_EN = os.getenv("TTS_VOICE_EN", "en-US-AriaNeural")

# ---- 多轮记忆 ----
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "10"))
