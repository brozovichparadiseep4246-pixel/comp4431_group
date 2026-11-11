"""语音识别 (faster-whisper, 本地离线) 与语音合成 (edge-tts, 中英双语)。"""
from __future__ import annotations

import asyncio
import re
import tempfile
import uuid
from pathlib import Path

from config import TTS_VOICE_EN, TTS_VOICE_ZH, WHISPER_MODEL

_whisper = None  # 延迟加载


def transcribe(audio_path: str | Path) -> str:
    """把麦克风录制的音频转写成文字（自动检测语言）。"""
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel

        _whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    segments, _info = _whisper.transcribe(str(audio_path), beam_size=5)
    return "".join(seg.text for seg in segments).strip()


def detect_language(text: str) -> str:
    """简单中英检测：中文字符占比高则返回 'zh'，否则 'en'。"""
    chinese = len(re.findall(r"[\u4e00-\u9fff]", text))
    return "zh" if chinese > len(text) * 0.2 else "en"


async def _tts_to_file(text: str, voice: str, out_path: str) -> None:
    import edge_tts

    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(out_path)


def synthesize(text: str) -> str | None:
    """把回复合成语音，返回音频文件路径；失败时返回 None（不影响文本回复）。"""
    plain = re.sub(r"[*#>`\-\[\]()|]", "", text).strip()
    if not plain:
        return None
    plain = plain[:600]  # 避免过长音频
    voice = TTS_VOICE_ZH if detect_language(plain) == "zh" else TTS_VOICE_EN
    out_path = str(Path(tempfile.gettempdir()) / f"coursemate_tts_{uuid.uuid4().hex}.mp3")
    try:
        asyncio.run(_tts_to_file(plain, voice, out_path))
        return out_path
    except Exception as e:  # 网络问题等，降级为纯文本
        Path(out_path).unlink(missing_ok=True)
        print(f"[TTS] 语音合成失败（不影响聊天）: {e}")
        return None
