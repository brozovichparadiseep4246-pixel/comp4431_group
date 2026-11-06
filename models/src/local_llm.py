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


def chat_stream(user_msg: str, history: list[dict], max_new_tokens: int = 240,
                context: str = ""):
    """真实流式生成：边生成边按句子输出（去重后），首句等待时间大幅缩短。
    history 为 [{"role","content"}...]；context 为检索结果，用于兜底保证事实准确。"""
    import re
    import threading
    from transformers import TextIteratorStreamer

    model, tokenizer = _load()
    torch.set_num_threads(max(1, os.cpu_count() or 4))  # 用满 CPU 线程

    messages = history[-6:] + [{"role": "user", "content": user_msg}]
    if context:
        sys_msg = ("你是选课咨询助手。学生问选课问题时，你要：1. 从下面的课程资料中挑 1-3 门"
                   "最匹配的课程；2. 用「课程代码：课程名称 + 推荐理由」的列表格式回答；"
                   "3. 只推荐资料中出现的课程，绝对不要编造资料里没有的课程代码或课程名称；"
                   "4. 涉及先修课时，只列出资料中'先修要求'字段写明的内容；"
                   "5. 资料中没有的信息（例如任课老师姓名、教材、具体考试日期、成绩分布等），"
                   "必须如实回答'知识库中没有收录该信息'，绝对不要编造任何名字、数字、日期；"
                   "6. 课程资料目前只有 COMP 计算机课程；如果学生不是计算机相关专业，请如实说明"
                   "知识库暂无该专业课程，建议咨询对应学院。\n课程资料：\n" + context)
        messages = [{"role": "system", "content": sys_msg}] + messages
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                      repetition_penalty=1.2, pad_token_id=tokenizer.eos_token_id,
                      streamer=streamer)
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
