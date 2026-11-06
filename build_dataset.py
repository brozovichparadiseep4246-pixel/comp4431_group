"""从课程知识库自动生成微调数据集（JSONL，Qwen2.5 chat 格式）。

用法:  python build_dataset.py
输出:  data/finetune_data.jsonl  （每条 {"prompt": ..., "completion": ...}）
"""
import json
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
DATA = json.loads((BASE_DIR / "data" / "courses.json").read_text(encoding="utf-8"))
OUT = BASE_DIR / "data" / "finetune_data.jsonl"

rng = random.Random(42)


def qa(questions: list[str], answer: str) -> list[dict]:
    return [{"prompt": q, "completion": a} for q, questions in [[q, questions] for q in questions] for a in [answer]]


def course_qas(c: dict) -> list[dict]:
    code = c["code"]
    title = f"{c['title_zh']}（{c['title_en']}）"
    pairs = [
        ([f"{code} 是什么课？", f"介绍一下 {code}", f"什么是 {code} {c['title_zh']}？",
          f"What is {code}?", f"Tell me about {code} {c['title_en']}."],
         f"{code}《{title}》{c['description_zh']}"),
        ([f"{code} 的先修课是什么？", f"修 {code} 之前需要上过什么课？",
          f"What are the prerequisites for {code}?",
          f"Can I take {code} without prerequisites?"],
         f"{code} 的先修要求是：{c['prerequisites']}。"),
        ([f"{code} 多少学分？", f"{code} 是几学分的课？", f"How many credits is {code}?"],
         f"{code} 是 {c['credits']} 学分的课程。"),
        ([f"{code} 什么时候开课？", f"{code} 哪个学期有开？",
          f"When is {code} offered?", f"Which semester does {code} run in?"],
         f"{code} 的开课学期为：{c['semester']}。"),
        ([f"{code} 怎么考核？", f"{code} 的考核方式是什么？", f"How is {code} assessed?"],
         f"{code} 的考核方式为：{c['assessment']}。"),
    ]
    out = []
    for qs, a in pairs:
        out += qa(qs, a)
    return out


def general_qas(g: dict) -> list[dict]:
    return qa(
        [f"{g['topic_zh']}的规定是什么？", f"请介绍一下{g['topic_zh']}的相关政策。",
         f"Tell me about {g['topic_en']}.", f"What should I know about {g['topic_en']}?"],
        g["content_zh"])


PLANNING = [
    (["我想往人工智能方向发展，应该怎么选课？", "帮我规划一条 AI 方向的学习路线",
      "I want to specialise in AI. What courses should I take?"],
     "推荐路线：先修 COMP1011 程序设计基础 → COMP2011 数据结构 → COMP4431 人工智能 "
     "→ COMP4432 机器学习 → COMP4441 自然语言处理导论。注意 COMP4432 要求先修 COMP4431，"
     "COMP4441 要求先修 COMP4432，请按顺序修读。"),
    (["我明年想上机器学习，今年要修什么？", "为了下学期修机器学习，这学期该选什么课？"],
     "机器学习 COMP4432 的先修课是 COMP4431 人工智能（建议具备概率统计基础）。"
     "这学期请优先修读 COMP4431；如果还没修过 COMP2011 数据结构，也需先完成它。"),
    (["零基础新生第一学期应该选什么课？", "我刚入学，推荐选哪些课？",
      "I'm a freshman with no programming background. What should I take first?"],
     "建议第一学期修读 COMP1011 程序设计基础（Python 教学，适合零基础），"
     "之后再依次修读 COMP2011 数据结构，为后续高阶课程打好基础。"),
    (["数据库课和操作系统课应该先修哪门？", "COMP2411 和 COMP2432 有先后顺序吗？"],
     "两门课的先修要求都是 COMP2011 数据结构，彼此之间没有先后强制要求，"
     "可按开课学期（COMP2411 在 Sem 1，COMP2432 在 Sem 2）灵活安排。"),
]


def unknown_qas(courses: list[dict]) -> list[dict]:
    """为每门课生成"知识库未收录"类样本，教模型如实说不知道，防止编造。"""
    unknown_fields = [
        ("的任课老师是谁？", "任课教师姓名"),
        ("的教材是什么？", "教材信息"),
        ("的期末考试具体是几号？", "具体考试日期"),
        ("在哪个教室上课？", "上课教室"),
        ("有多少个学生选这门课？", "选课人数"),
        ("往届平均成绩是多少？", "成绩分布"),
        (" who teaches it?", "任课教师姓名"),
    ]
    out = []
    for c in courses:
        code = c["code"]
        title = f"{c['title_zh']}（{c['title_en']}）"
        for suffix, field in unknown_fields[:4]:  # 每门课 4 条，避免样本比例失衡
            q = f"{code} {suffix}"
            a = (f"抱歉，知识库中没有收录 {code}《{title}》的{field}。"
                 f"建议你登录学校选课系统或联系学系办公室查询。")
            out.append({"prompt": q, "completion": a})
    return out


def main() -> None:
    examples: list[dict] = []
    for c in DATA["courses"]:
        examples += course_qas(c)
    for g in DATA["general_info"]:
        examples += general_qas(g)
    for qs, a in PLANNING:
        examples += qa(qs, a)
    examples += unknown_qas(DATA["courses"])

    rng.shuffle(examples)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"已生成 {len(examples)} 条微调样本 -> {OUT}")
    print("样例:", json.dumps(examples[0], ensure_ascii=False)[:120])


if __name__ == "__main__":
    main()
