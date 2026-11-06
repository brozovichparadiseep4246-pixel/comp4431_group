"""CourseMate 选课咨询聊天机器人 — Gradio 网页界面（DeepSeek 风格）。

功能: 在线大模型(支持 Groq 等免费 OpenAI 兼容 API) + RAG 知识库
      / 本地 LoRA 微调模型 双引擎,
      语音问答, 2D 卡通吉祥物, 中英双语, 多轮记忆。

运行:  python src/app.py  (或双击 run.bat)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr

from avatar import AVATAR_BODY, AVATAR_CSS, AVATAR_SCRIPT
from config import LLM_MODEL
from knowledge_base import KnowledgeBase
from llm import ChatBot
from voice import synthesize, transcribe

kb = KnowledgeBase()
kb.load()
bot = ChatBot(kb)

EXAMPLES = [
    "COMP4431 人工智能这门课主要学什么？考核方式是什么？",
    "我想下学期学机器学习，需要先修哪些课？",
    "Which database course should I take and what are its prerequisites?",
    "加退选是什么时候？如果没满足先修课要求怎么办？",
    "帮我规划一条学习人工智能方向的选课路线",
    "Is COMP4911 Capstone Project offered in both semesters?",
]

MODE_KIMI = "🧠 在线大模型（RAG 检索）"
MODE_LOCAL = "🤖 本地微调（离线）"


def _mode(v: str) -> str:
    return "local" if v == MODE_LOCAL else "kimi"


def respond(message: str, history: list, speak: bool, mode: str):
    """文本/示例提问 → LLM 流式回复 →（可选）语音播报 + 吉祥物状态。"""
    message = (message or "").strip()
    if not message:
        yield history, None, "", "idle"
        return
    history = history + [{"role": "user", "content": message}]
    m = _mode(mode)
    if m == "local":
        yield history, None, "🤖 本地微调模型生成中（CPU 推理，回答将逐句输出，前几个词可能稍慢）…", "thinking"
    else:
        yield history, None, "🔎 正在检索课程知识库…", "thinking"
    try:
        gen, sources, web_used = bot.chat(message, stream=True, model_mode=m)
        partial = ""
        for delta in gen:
            partial += delta
            yield history + [{"role": "assistant", "content": partial}], None, "", "speaking"
    except Exception as e:
        yield history, None, f"❌ 出错了: {e}", "idle"
        return
    history = history + [{"role": "assistant", "content": partial}]
    audio = synthesize(partial) if speak else None
    if sources is None:
        # 固定话术回复: 根据内容区分状态提示
        if '已记录你的背景' in partial:
            status = f"👤 {bot.profile_text()}"
        elif '请先告诉我' in partial:
            status = "❓ 已请用户补充专业/年级（学校默认 PolyU），再给出个性化建议"
        elif '抱歉，我是 CourseMate' in partial:
            status = "🙅 非选课领域问题，已礼貌拒绝（本助手只提供选课咨询）"
        else:
            status = "💬 寒暄回复"
    elif m == "local":
        status = "🤖 由本地 LoRA 微调模型（Qwen2.5-0.5B）生成，无需联网。"
    elif web_used and sources:
        srcs = ", ".join(sorted({c.source for c in sources}))
        status = f"📚+🌐 知识库低置信命中({srcs})，已结合网络搜索回答，请核实。"
    elif web_used:
        status = "🌐 知识库未命中，已从网络搜索获取信息（内容仅供参考，请核实）。"
    elif sources:
        srcs = ", ".join(sorted({c.source for c in sources}))
        status = f"📚 知识来源: {srcs}"
    else:
        status = "⚠️ 知识库和网络都未找到相关内容，请以学校官方信息为准。"
    yield history, audio, status, "happy"


def voice_ask(audio_path, history: list, speak: bool, mode: str):
    """语音提问：转写 → 走正常文本流程。"""
    if not audio_path:
        yield history, None, "请先点击麦克风录制语音，再按“语音提问”。", "idle"
        return
    yield history, None, "🎤 正在识别语音…", "thinking"
    try:
        text = transcribe(audio_path)
    except Exception as e:
        yield history, None, f"语音识别失败: {e}", "idle"
        return
    if not text:
        yield history, None, "未能识别到语音内容，请再试一次。", "idle"
        return
    yield from respond(text, history, speak, mode)


def clear_all():
    bot.reset()
    return [], None, "", "idle"


theme = gr.themes.Soft(
    primary_hue="stone",
    secondary_hue="neutral",
    neutral_hue="stone",
).set(
    body_background_fill="#faf9f7",
    block_background_fill="#ffffff",
    block_border_width="0px",
    button_primary_background_fill="#1f1f1f",
    button_primary_background_fill_hover="#000000",
    button_primary_text_color="#ffffff",
    button_border_width="0px",
    block_radius="16px",
)

# 回车后立即清空输入框（兼容中文输入法: 组词确认的回车不清空）
CLEAR_SCRIPT = """
<script>
(function () {
  if (window.__composer_clear) return;
  window.__composer_clear = true;
  function bind() {
    var ta = document.querySelector('.composer textarea');
    if (!ta) { setTimeout(bind, 800); return; }
    if (!ta.__clearBound) {
      ta.__clearBound = true;
      ta.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
          setTimeout(function () {
            ta.value = '';
            ta.dispatchEvent(new Event('input', { bubbles: true }));
          }, 50);
        }
      });
    }
    setTimeout(bind, 2000);  // Gradio 可能重建 textarea，持续重绑
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', bind);
  else bind();
})();
</script>
"""

CSS = """
/* ===== Kimi 风格：暖米白极简 ===== */
body, .gradio-container { background: #faf9f7 !important; }
.gradio-container { max-width: 800px !important; margin: 0 auto; font-family: -apple-system, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif; }
/* ===== 顶部导航栏 ===== */
.topbar { display: flex; align-items: center; justify-content: space-between; padding: 12px 4px 14px; border-bottom: 1px solid #efece5; margin-bottom: 12px; }
.topbar-left { display: flex; align-items: center; gap: 12px; }
.topbar-logo { width: 34px; height: 34px; border-radius: 10px; background: #1f1f1f; display: flex; align-items: center; justify-content: center; font-size: 18px; }
.topbar-name { font-size: 17px; font-weight: 600; color: #1a1a18; letter-spacing: 0.2px; }
.topbar-sub { font-size: 12px; color: #a29b8e; margin-top: 1px; }
.topbar-right { display: flex; align-items: center; gap: 8px; }
.tb-badge { font-size: 11px; color: #8f897c; background: #f1eee7; border-radius: 999px; padding: 3px 11px; }
/* ===== 聊天区：机器人纯文本无气泡，用户暖灰气泡 ===== */
.chat-panel { border: none !important; background: transparent !important; }
.chat-panel .message { border-radius: 18px !important; box-shadow: none !important; line-height: 1.75; }
.chat-panel .message.bot { background: transparent !important; color: #26241f !important; border: none !important; padding: 8px 4px !important; }
.chat-panel .message.bot * { color: #26241f; }
.chat-panel .message.user { background: #e9e5de !important; color: #26241f !important; border: none !important; padding: 10px 16px !important; }
.chat-panel .message.user * { color: #26241f !important; }
/* ===== 状态行 ===== */
.status-line { min-height: 20px; text-align: center; color: #a89f91; font-size: 12px; }
/* ===== 底部输入框（Kimi 风格大圆角白卡） ===== */
.composer { border: 1px solid #e7e2d9; border-radius: 26px; background: #ffffff; box-shadow: 0 4px 24px rgba(90,80,60,0.07); padding: 4px 8px 4px 4px; align-items: center; }
.composer > .textbox, .composer .textbox { border: none !important; box-shadow: none !important; background: transparent !important; }
.composer textarea { border: none !important; box-shadow: none !important; background: transparent !important; resize: none; font-size: 15px; padding: 12px 8px 4px 16px !important; }
.composer textarea:focus { outline: none !important; }
.composer-send { min-width: 42px !important; max-width: 42px !important; height: 42px !important; border-radius: 999px !important; padding: 0 !important; margin-top: 2px; }
.composer-send button { border-radius: 999px !important; background: #1f1f1f !important; color: #fff !important; }
.composer-send button:hover { background: #000000 !important; }
/* ===== 工具行 ===== */
.tools-row { background: #f4f1ea; border: 1px solid #ece7de; border-radius: 18px; padding: 10px 14px; margin-top: 10px; }
.tools-row label { font-size: 12px !important; color: #8f897c !important; }
/* ===== 示例问题卡片 ===== */
.examples .examples-table button { border: 1px solid #e7e2d9 !important; border-radius: 14px !important; background: #ffffff !important; color: #6f6a5e !important; font-size: 13px !important; padding: 9px 14px !important; text-align: left; box-shadow: none !important; }
.examples .examples-table button:hover { border-color: #d9d2c4 !important; color: #1f1f1f !important; background: #f7f4ee !important; }
#bot_state_box { display: none !important; }
footer { display: none !important; }
""" + AVATAR_CSS


with gr.Blocks(title="CourseMate 选课咨询机器人") as demo:
    # 注意: Gradio 6 对 visible=False 的组件不渲染 DOM, 所以用 CSS 隐藏而不是 visible=False
    bot_state = gr.Textbox(value="idle", elem_id="bot_state_box", show_label=False, label="")
    gr.HTML(
        '<div class="topbar">'
        '<div class="topbar-left">'
        '<div class="topbar-logo">🎓</div>'
        '<div><div class="topbar-name">CourseMate 选课咨询助手</div>'
        '<div class="topbar-sub">大模型 + RAG 知识库 · 中英双语 · 语音对话</div></div>'
        '</div>'
        '<div class="topbar-right">'
        '<span class="tb-badge">🧠 RAG</span><span class="tb-badge">🤖 LoRA 微调</span>'
        + AVATAR_BODY +
        '</div></div>',
        head=AVATAR_SCRIPT,
    )
    gr.HTML("", head=CLEAR_SCRIPT)  # 注入回车清空脚本

    with gr.Column(elem_classes="chat-panel"):
        chatbot = gr.Chatbot(
            height=500,
            show_label=False,
            placeholder="向我咨询课程信息、先修要求、选课规划…",
        )
    status = gr.Markdown("", elem_classes="status-line")

    with gr.Row(elem_classes="composer"):
        msg = gr.Textbox(
            placeholder="输入你的选课问题，例如：COMP4432 的先修课是什么？",
            scale=9, lines=1, max_lines=4,
            show_label=False,
        )
        send = gr.Button("➤", variant="primary", scale=0, elem_classes="composer-send")

    with gr.Row(elem_classes="tools-row"):
        mode = gr.Radio([MODE_KIMI, MODE_LOCAL], value=MODE_KIMI,
                        label="模型引擎", scale=4)
        speak = gr.Checkbox(value=True, label="🔊 语音播报", scale=1)
        mic = gr.Audio(sources=["microphone", "upload"], type="filepath", scale=4,
                       label="🎤 语音输入（录音或直接上传音频文件）")
        voice_btn = gr.Button("🎤 提问", scale=1)
        tts_audio = gr.Audio(label="语音回复播放", interactive=False, scale=3)
        clear = gr.Button("🗑️ 清空", scale=1)

    ex = gr.Examples(examples=EXAMPLES, inputs=msg,
                     label="💡 试试这些问题")
    ex.elem_classes = ["examples"]

    inputs = [msg, chatbot, speak, mode]
    outputs = [chatbot, tts_audio, status, bot_state]
    msg.submit(respond, inputs, outputs).then(lambda: "", None, msg)
    send.click(respond, inputs, outputs).then(lambda: "", None, msg)
    voice_btn.click(voice_ask, [mic, chatbot, speak, mode], outputs)
    clear.click(clear_all, None, outputs)

if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=theme, css=CSS)
