"""本地微调模型（Qwen2.5-0.5B + LoRA adapter）推理封装。"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent.parent
ADAPTER_DIR = BASE_DIR / "models" / "qwen-lora"
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# 本地引擎的 system 提示词模板（训练与推理必须保持一致，见 build_dataset.py）

LOCAL_SYS_TEMPLATE = (
    
    "你是选课咨询助手。你需要根据下面的课程资料，回答且只能回答学术问题，要求：\n"
    """
    0. 首先判断用户的问题是否与选课相关。如果不是，则拒绝回答，如果是，则进一步询问情况；
    1. 记录用户的相关信息，并在对话开始时复述。如用户输入大三，计算机，ai需要输出：好的，让我记录下来。您是大三计算机系学生；
    2. 语法自然，完整包含主谓宾三要素；
    3. 拒绝回答资料中没有的信息（如任课老师、教材、具体考试日期），如实说明并建议咨询相关机构；
    4. 资料目前只有 COMP 计算机课程；其他专业如实说明并建议咨询对应学院；
    5. 同时给出中英文回答。"""
    "错误案例如：用户要求推荐几个音乐选秀，ai回复《中国好声音》 - 每周一至周五...\n"
    "正确案例如：抱歉，我无法提供与学术问题无关的建议。Sorry, I am unable to provide advice unrelated to academic issues.\n"
    "错误案例如：用户要求推荐一些选修课，ai回复抱歉，当前资料中没有列出...\。\n"
    "正确案例如：好的，为了给您推荐，我需要进一步了解情况。请问您的专业和学习阶段是？Okay, in order to recommend it to you, I need to further understand the situation. May I ask about your major and stage of study?\n"
    "课程资料：\n{context}"
)

_model = None
_tokenizer = None


def available() -> bool:
    return ADAPTER_DIR.exists() and any(ADAPTER_DIR.iterdir())


def _load():
    global _model, _tokenizer
    if _model is None:
        if not available():
            raise RuntimeError(
                f"未找到微调模型: {ADAPTER_DIR}\n请先运行: python build_dataset.py && python finetune.py")
        print("[LocalLLM] 正在加载本地微调模型（首次需下载基座模型约 1GB）…")
        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
        _model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
        _model.eval()
        print("[LocalLLM] 模型加载完成")
    return _model, _tokenizer


class _SentenceDeduper:
    """逐句去重：0.5B 小模型容易重复整句，丢弃滑动窗口内出现过的句子。"""

    def __init__(self) -> None:
        self._seen: list[str] = []

    @staticmethod
    def _norm(s: str) -> str:
        import re
        return re.sub(r"[\s，,、]+", "", s)

    def feed(self, sentence: str) -> str | None:
        """返回应输出的句子；若为重复句或自我纠错噪音返回 None。"""
        s = sentence.strip()
        if not s:
            return None
        # 小模型在答完后容易追加“抱歉，我之前的解答存在误…”这类噪音，丢弃
        if any(p in s for p in ("之前的解答存在误", "我之前", "更正", "修正", "重新回答")):
            return None
        n = self._norm(s)
        if n in self._seen:
            return None
        self._seen.append(n)
        self._seen = self._seen[-3:]
        return s


def chat_stream(user_msg: str, history: list[dict], max_new_tokens: int = 1024,
                context: str = ""):
    """真实流式生成：边生成边按句子输出（去重后），首句等待时间大幅缩短。
    history 为 [{"role","content"}...]；context 为检索结果，用于兜底保证事实准确。"""
    import re
    import threading
    from transformers import TextIteratorStreamer

    model, tokenizer = _load()
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))  # 20 逻辑核 -> 10 线程，避免超卖变慢

    messages = history[-6:] + [{"role": "user", "content": user_msg}]
    if context:
        sys_msg = LOCAL_SYS_TEMPLATE.format(context=context)
        messages = [{"role": "system", "content": sys_msg}] + messages
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(**inputs, max_new_tokens=max_new_tokens,
                      pad_token_id=tokenizer.eos_token_id,
                      streamer=streamer, do_sample=True, temperature=0.7,
                      top_k=20, top_p=0.8, repetition_penalty=1.1)
    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    deduper = _SentenceDeduper()
    buf = ""
    for piece in streamer:          # 逐 token 接收
        buf += piece
        m = re.search(r"^(.*[。！？!?\n])(.*)$", buf, re.DOTALL)
        if m and m.group(1).strip():
            out = deduper.feed(m.group(1))
            buf = m.group(2)
            if out:
                yield out
    thread.join()
    if buf.strip():                 # 末尾没有标点的残留
        out = deduper.feed(buf)
        if out:
            yield out
