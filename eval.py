"""评测脚本：用固定问答对测试知识库检索与机器人回答（非流式）。

用法:  python eval.py            # 检索 + LLM 回答评测（消耗少量 API 额度）
        python eval.py --retrieval-only   # 只测检索，不调用 LLM
"""
import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from knowledge_base import KnowledgeBase

# (问题, 期望命中的知识来源关键词)
QA_PAIRS = [
    ("COMP4431 人工智能主要学什么？", "COMP4431"),
    ("我想学机器学习，需要先修什么课？", "COMP4432"),
    ("What are the prerequisites of COMP2411 Database Systems?", "COMP2411"),
    ("加退选是什么时候？", "general_info"),
    ("Is COMP4911 Capstone Project a full-year course?", "COMP4911"),
    ("课程名额满了怎么办？", "kb_docs"),
    ("Which course covers dynamic programming and greedy algorithms?", "COMP3011"),
    ("没满足先修课要求可以豁免吗？", "general_info"),
    ("操作系统课有 Linux 实验吗？", "COMP2432"),
    ("帮我规划 AI 方向的选课路线", "COMP4431"),
]


def eval_retrieval(kb: KnowledgeBase) -> float:
    print("=" * 60)
    print("检索评测 (Top-k RAG Retrieval)")
    print("=" * 60)
    hit = 0
    for q, expect in QA_PAIRS:
        results = kb.retrieve(q)
        top_sources = [c.source for c, _ in results]
        ok = any(expect in s for s in top_sources)
        hit += ok
        mark = "✅" if ok else "❌"
        print(f"{mark} Q: {q}\n   期望: {expect} | 命中: {top_sources[:3]}")
    acc = hit / len(QA_PAIRS)
    print("-" * 60)
    print(f"检索命中率 Retrieval Accuracy: {hit}/{len(QA_PAIRS)} = {acc:.0%}")
    return acc


def eval_llm(kb) -> None:
    from llm import ChatBot

    print("\n" + "=" * 60)
    print("回答评测 (LLM Answers, RAG)")
    print("=" * 60)
    bot = ChatBot(kb)
    for q, _ in QA_PAIRS[:5]:
        reply, sources, web_used = bot.chat(q, stream=False)
        srcs = ", ".join(sorted({c.source for c in sources}))
        tag = "🌐网络" if web_used else f"📚{srcs}"
        print(f"\nQ: {q}\nA: {reply}\n来源: {tag}\n{'-' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-only", action="store_true",
                        help="只评测检索，不调用 LLM（不消耗 API 额度）")
    args = parser.parse_args()

    kb = KnowledgeBase()
    kb.load()
    acc = eval_retrieval(kb)
    if not args.retrieval_only:
        eval_llm(kb)
    print(f"\n完成。检索准确率: {acc:.0%}")
