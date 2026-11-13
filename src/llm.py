"""在线大模型封装：RAG 提示词、领域守门、多轮历史管理、流式输出。"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict, deque
from pathlib import Path

from openai import OpenAI

from config import (COURSES_JSON, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE,
                    MAX_HISTORY_TURNS, TOP_K)
from knowledge_base import Chunk, KnowledgeBase

SYSTEM_PROMPT = """你是选课咨询助手。你需要根据下面的课程资料，回答且只能回答学术问题，要求：
0. 首先判断用户的问题是否与选课相关。如果不是，则拒绝回答，如果是，则进一步询问情况；
1. 记录用户的相关信息，并在对话开始时复述。如用户输入大三，计算机，ai需要输出：好的，让我记录下来。您是大三计算机系学生；
2. 语法自然，完整包含主谓宾三要素；
3. 拒绝回答资料中没有的信息（如任课老师、教材、具体考试日期），如实说明并建议咨询相关机构；
4. 资料目前只有 COMP 计算机课程；其他专业如实说明并建议咨询对应学院；
5. 同时给出中英文回答。
错误案例如：用户要求推荐几个音乐选秀，ai回复《中国好声音》 - 每周一至周五...
正确案例如：抱歉，我无法提供与学术问题无关的建议。Sorry, I am unable to provide advice unrelated to academic issues.
错误案例如：用户要求推荐一些选修课，ai回复抱歉，当前资料中没有列出...
正确案例如：好的，为了给您推荐，我需要进一步了解情况。请问您的专业和学习阶段是？Okay, in order to recommend it to you, I need to further understand the situation. May I ask about your major and stage of study
课程资料：
{context}
"""


# 选课领域关键词（中英）
DOMAIN_KEYWORDS = [
    # 中文
    "课程", "选课", "学分", "先修", "必修", "选修", "考试", "考核", "课表", "校历",
    "学期", "毕业", "学位", "绩点", "gpa", "成绩", "重修", "补考", "挂科",
    "转专业", "专业", "学习", "修读", "旁听", "暑课", "暑期", "交换", "留学",
    "研究生", "硕士", "博士", "导师", "学费", "奖学金", "退课", "加课", "抢课",
    "免修", "豁免", "双学位", "辅修", "新生", "大一", "大二", "大三", "大四",
    "时间表", "开班", "名额", "waitlist", "退选", "加选", "补选", "预选",
    "实验", "上机", "作业", "期末", "期中", "挂", "缓考", "实验室",
    "选什么", "什么课", "门课", "学计算机", "想学生物", "想选", "开课", "数据库",
    # 英文
    "course", "credit", "prerequisite", "enroll", "enrol", "semester", "timetable",
    "curriculum", "degree", "gpa", "grade", "major", "minor", "elective",
    "compulsory", "freshman", "sophomore", "junior", "senior", "tuition",
    "scholarship", "graduate", "postgraduate", "assessment", "module",
    "lecture", "tutorial", "examination", "study plan", "academic",
    "capstone", "thesis", "dissertation", "毕业设计", "毕设", "论文",
]

# 明显的非选课话题关键词, 直接拒绝（不再交给大模型判断）
OFFDOMAIN_KEYWORDS = [
    "社团", "理财", "脱单", "恋爱", "电影", "电视剧", "天气", "美食", "游戏",
    "穿搭", "化妆", "减肥", "健身", "旅游攻略", "房价", "股票", "基金",
    "crypto", "bitcoin", "movie", "weather", "game", "dating", "recipe",
    "fitness", "stock", "invest",
]

# 寒暄语（整条消息只是打招呼/道谢/告别时, 友好回应并引导提问, 不算越界）
_GREET_RE = re.compile(
    r"^(你好|您好|嗨|哈喽|喂|早|早上好|下午好|晚上好|好呀|好啊|hi|hello|hey|yo"
    r"|good\s?(morning|afternoon|evening|day))[!！。\.~～\s]*$", re.IGNORECASE)
_THANKS_RE = re.compile(r"^(谢谢|感谢|辛苦|thanks|thank you|thx|ty)[!！。\.~～\s]*$", re.IGNORECASE)
_BYE_RE = re.compile(r"^(再见|拜拜|bye|goodbye|see you)[!！。\.~～\s]*$", re.IGNORECASE)

GREETING_REPLY = (
    "你好呀！👋 我是 CourseMate **选课咨询助手**，可以帮你：\n\n"
    "- 查询课程内容、学分、开课学期与考核方式\n"
    "- 确认先修要求、规划选课路线\n"
    "- 了解加退选、学分政策等选课问题\n\n"
    "想咨询哪门课或哪方面的选课问题呢？\n\n"
    "Hello! Ask me about course info, prerequisites, credits, or study planning 😊"
)
THANKS_REPLY = "不客气！😊 还有想咨询的选课问题吗？Feel free to ask me more about course selection."
BYE_REPLY = "再见！👋 祝选课顺利～ Good luck with your course selection!"

# 规划/推荐类问题关键词（这类问题需要学生背景才能答准，缺档案时先反问）
ADVISORY_KEYWORDS = [
    "规划", "推荐", "建议", "路线", "怎么选", "如何选", "选什么", "选哪", "选哪些",
    "帮我选", "安排", "计划", "值得", "适合我", "定位", "发展方向",
    "plan", "recommend", "advice", "roadmap", "which course", "what course",
    "should i take", "schedule",
]

ASK_PROFILE_REPLY = (
    "好的！为了给你更准确的选课建议，请先告诉我：\n\n"
    "1️⃣ 你的**专业**（如：计算机科学 / Computer Science）\n"
    "2️⃣ 你的**年级**（大一 / 大二 / 大三 / 大四）\n\n"
    "学校默认按 **香港理工大学（PolyU）** 处理；如不是，请注明学校。\n\n"
    "Sure! To give you accurate advice, please tell me your **major** and "
    "**year of study** (school defaults to PolyU). 😊"
)

_PROFILE_YEAR_RE = re.compile(r"大([一二三四1-4])")
# 两种模式: "X专业"（如 计算机专业/软件工程专业）或 "专业是X"
_PROFILE_MAJOR_A_RE = re.compile(r"([一-龥A-Za-z&]{2,12}?)\s*专业")
_PROFILE_MAJOR_B_RE = re.compile(r"专业(?:是|为|读的是|：|:)?\s*([一-龥A-Za-z&（）()]{2,15}?)(?:[，。,.；;]|$)")
_PROFILE_MAJOR_C_RE = re.compile(r"学([一-龥A-Za-z]{2,10}?)的")  # "我学计算机的"
_PROFILE_MAJOR_D_RE = re.compile(r"我是([一-龥A-Za-z]{2,10}?)(?:专业)?的")  # "我是计算机的"
_PROFILE_MAJOR_E_RE = re.compile(r"我是([一-龥A-Za-z&]{2,12}?)(?:专业)?学生")  # "我是计算机学生"


def _clean_major(text: str) -> str | None:
    t = text.strip(" 的一二三四了")
    if len(t) < 2 or re.match(r"^(大[一二三四1-4]|\d)", t):
        return None
    # 排除 "学习很认真的" 这类误匹配
    if t.startswith(("学习", "学会", "学到", "学好")) or t in {"学院", "学历"}:
        return None
    return t
_PROFILE_SCHOOL_RE = re.compile(
    r"(香港理工大学|香港大学|香港中文大学|香港科技大学|香港城市大学|香港浸会大学|"
    r"PolyU|HKU|CUHK|HKUST|CityU|HKBU|[一-龥]{2,8}大学)", re.IGNORECASE)

DOMAIN_REFUSAL = (
    "抱歉，我是 CourseMate **选课咨询助手**，只回答课程选择、课程信息、学分与先修要求、"
    "选课政策与学习规划相关的 questions 哦～\n\n"
    "Sorry, I can only help with **course selection & academic advising** questions. "
    "For other topics, please try a general-purpose AI assistant. 😊"
)


def build_local_context(kb: KnowledgeBase, user_msg: str, profile: str) -> tuple[str, list[Chunk]]:
    """本地引擎的上下文构造（训练数据生成与线上推理共用，保证分布一致）。
    返回 (context, sources)。"""
    hits = kb.retrieve(user_msg, top_k=TOP_K)
    # 查询中明确提到的课程代码对应的资料块排在最前，减少小模型张冠李戴
    sources = sorted([c for c, _ in hits], key=lambda c: 0 if c.source in user_msg else 1)
    # 小模型 prefill 慢: 只保留最相关的 3 条并截断, 缩短首句等待
    context = "\n\n".join(f"[{c.source}]\n{c.text}" for c in sources[:3])[:700]
    context = (f"学生背景：{profile}（未知表示未提供，请结合背景回答）\n\n" + context)
    return context, sources


# ---------- 本地模式：课程事实确定性回答（离线准确优先于表达多样） ----------

_COURSES_DB: dict | None = None


def _course_db() -> dict:
    """courses.json 按课程代码索引（懒加载缓存）。"""
    global _COURSES_DB
    if _COURSES_DB is None:
        data = json.loads(Path(COURSES_JSON).read_text(encoding="utf-8"))
        _COURSES_DB = {c["code"]: c for c in data.get("courses", [])}
    return _COURSES_DB


def _no_prereq(p: str) -> bool:
    return not p or p.strip() in ("无 / None", "无", "None", "none")


def _followups(db: dict, code: str) -> list[str]:
    """以本课程为先修课的后续课程（用于简单建议）。"""
    return [f"{x['code']}《{x['title_zh']}》" for x in db.values()
            if x["code"] != code and code in (x.get("prerequisites") or "")]


def _local_course_reply(user_msg: str) -> str | None:
    """本地模式：课程事实类问题直接从知识库组装确定性回答 + 简单建议，
    不让 0.5B 小模型自由生成（防止编造）。返回 None 表示交给微调模型。"""
    db = _course_db()
    zh = bool(re.search(r"[\u4e00-\u9fff]", user_msg))
    upper = user_msg.upper()

    # 非计算机专业选课咨询 → 如实说明知识库范围
    if re.search(r"护理|医学|商科|金融|会计|经济|法律|心理|教育|nursing|business|medic"
                 r"|finance|accounting|psycholog|\blaw\b|education", user_msg, re.IGNORECASE):
        return ("抱歉，目前知识库只收录 COMP 计算机专业课程，暂无其他专业的课程信息。"
                "建议咨询你所在学院的办公室或学术导师（Academic Advisor）获取选课建议。" if zh else
                "Sorry, the knowledge base currently only covers COMP (Computer Science) "
                "courses. Please consult your home school/department or academic advisor.")

    # 非 COMP 课程代码（如 MATH1011）→ 如实说明只收录 COMP
    for m in re.findall(r"\b[A-Z]{2,6}\s?\d{3,4}\b", upper):
        code = m.replace(" ", "")
        if not code.startswith("COMP"):
            return (f"抱歉，目前知识库只收录 COMP 计算机课程，暂未收录 {code} 这类课程，"
                    "建议咨询对应学院或学系办公室。" if zh else
                    f"Sorry, the knowledge base currently only covers COMP courses; {code} "
                    "is not included. Please check with the corresponding department.")

    codes = [m.replace(" ", "") for m in re.findall(r"COMP\s?\d{4}", upper)]
    if not codes:
        # 规划/推荐/比较类问题优先交给微调模型（别名匹配只服务单课事实查询）
        if re.search(r"规划|路线|怎么选|如何选|选什么|选哪些|推荐|建议.{0,6}课|"
                     r"roadmap|plan|recommend|which course|what course|should i take",
                     user_msg, re.IGNORECASE):
            return None
        # 没写课程代码但提到课程名（如“机器学习”“数据库课”）→ 映射到课程代码
        # 别名：完整课名 + 前 3 字（如 数据库系统→数据库）
        hits = []
        for x in db.values():
            t_zh, t_en = x.get("title_zh", ""), x.get("title_en", "")
            aliases = {t_zh, t_zh[:3] if len(t_zh) > 3 else "", t_en.lower()}
            if any(a and (a in user_msg or a in user_msg.lower()) for a in aliases):
                hits.append(x["code"])
        if len(hits) == 1:
            codes = hits
        elif len(hits) > 1:
            return None  # 提到多门课 → 交给微调模型比较/推理
        else:
            return None  # 政策/规划类问题交给微调模型
    missing = [c for c in codes if c not in db]
    if missing:
        return (f"知识库中没有收录 {missing[0]} 这门课（目前共收录 {len(db)} 门 COMP 课程）。"
                "请核对课程代码，或登录学校选课系统 / 咨询学系办公室查询。" if zh else
                f"{missing[0]} is not in the knowledge base (it currently covers {len(db)} "
                "COMP courses). Please double-check the course code or consult the department office.")
    if len(set(codes)) > 1:
        return None  # 多课程比较/推理交给微调模型

    c = db[codes[0]]
    code = c["code"]
    title = f"{c['title_zh']} / {c['title_en']}" if zh else f"{c['title_en']} ({c['title_zh']})"
    name = f"{code}《{c['title_zh']}》" if zh else f"{code} {c['title_en']}"
    prereq, sem, assess = c.get("prerequisites", ""), c.get("semester", ""), c.get("assessment", "")
    prereq_disp = ("无（任何同学都可直接选修）" if zh else "None (open to all students)") \
        if _no_prereq(prereq) else prereq

    # 简单建议：先修提醒 + 后续衔接 + 开课学期提醒
    def tips() -> str:
        t = []
        if not _no_prereq(prereq):
            t.append(f"选课前需先完成先修课：{prereq}" if zh else
                     f"complete the prerequisite first: {prereq}")
        fol = _followups(db, code)
        if fol:
            t.append(("修完后可衔接：" + "、".join(fol)) if zh else
                     "it leads to: " + ", ".join(fol))
        if "&" not in sem:
            t.append(f"仅 {sem} 开课，注意提前规划" if zh else
                     f"only offered in {sem}, plan ahead")
        return ("\n💡 建议：" + "；".join(t) + "。") if t and zh else \
               ("\n💡 Tip: " + "; ".join(t) + ".") if t else ""

    if re.search(r"先修|pre.?req", user_msg, re.IGNORECASE):
        return (f"{name}的先修要求是：{prereq_disp}。{tips()}" if zh else
                f"The prerequisite for {name} is: {prereq_disp}.{tips()}")
    if re.search(r"学分|credit", user_msg, re.IGNORECASE):
        return (f"{name}是 {c['credits']} 学分的课程，{sem} 开课。{tips()}" if zh else
                f"{name} is a {c['credits']}-credit course, offered in {sem}.{tips()}")
    if re.search(r"开课|哪个学期|学期|semester|offered|when", user_msg, re.IGNORECASE):
        return (f"{name}的开课学期是：{sem}。{tips()}" if zh else
                f"{name} is offered in: {sem}.{tips()}")
    if re.search(r"考核|怎么评|考试|assess|exam", user_msg, re.IGNORECASE):
        return (f"{name}的考核方式是：{assess}。{tips()}" if zh else
                f"{name} is assessed by: {assess}.{tips()}")
    # 默认：课程总体介绍（是什么/学什么/怎么样/值得选吗 等）
    desc = c.get("description_zh" if zh else "description_en", "")
    remark = c.get("remarks", "")
    lines = [f"{code}《{title}》，{c['credits']} 学分，{sem} 开课。" if zh else
             f"{code} {title}, {c['credits']} credits, offered in {sem}.",
             (f"- 内容：{desc}" if zh else f"- Content: {desc}"),
             (f"- 先修：{prereq}\n- 考核：{assess}" if zh else
              f"- Prerequisite: {prereq}\n- Assessment: {assess}")]
    if remark:
        lines.append(f"- 备注：{remark}" if zh else f"- Note: {remark}")
    return "\n".join(lines) + tips()


class ChatBot:
    def __init__(self, kb: KnowledgeBase):
        # 无 API Key 时报错并停止在线服务
        self.client = (OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
                       if LLM_API_KEY else None)
        self.kb = kb
        self._histories: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=MAX_HISTORY_TURNS * 2))
        # 学生档案: {session_id: {"school": str, "major": str|None, "year": str|None}}
        self._profiles: dict[str, dict] = defaultdict(
            lambda: {"school": "香港理工大学 (PolyU)", "major": None, "year": None})

    def reset(self, session_id: str = "default") -> None:
        self._histories.pop(session_id, None)
        self._profiles.pop(session_id, None)

    # ---------- 学生档案 ----------

    def _extract_profile(self, msg: str, session_id: str) -> None:
        """从用户消息中自动提取专业/年级/学校信息。"""
        prof = self._profiles[session_id]
        m = _PROFILE_YEAR_RE.search(msg)
        if m:
            _year_map = {"一": "一", "二": "二", "三": "三", "四": "四",
                         "1": "一", "2": "二", "3": "三", "4": "四"}
            prof["year"] = "大" + _year_map[m.group(1)]
        major = None
        for rx in (_PROFILE_MAJOR_A_RE, _PROFILE_MAJOR_B_RE, _PROFILE_MAJOR_C_RE, _PROFILE_MAJOR_D_RE, _PROFILE_MAJOR_E_RE):
            m = rx.search(msg)
            if m:
                major = _clean_major(m.group(1))
                if major:
                    break
        if major:
            prof["major"] = major
        m = _PROFILE_SCHOOL_RE.search(msg)
        if m:
            prof["school"] = m.group(1)

    def profile_text(self, session_id: str = "default") -> str:
        prof = self._profiles[session_id]
        return (f"学校: {prof['school']}；专业: {prof['major'] or '未知'}；"
                f"年级: {prof['year'] or '未知'}")

    def profile_complete(self, session_id: str = "default") -> bool:
        prof = self._profiles[session_id]
        return bool(prof["major"] and prof["year"])

    def _is_advisory(self, msg: str) -> bool:
        return False           # 恒非规划类

    @staticmethod
    def _contains_profile_info(msg: str) -> bool:
        return False           # 恒不认为消息包含档案信息

    # ---------- RAG ----------

    def retrieve_context(self, query: str) -> tuple[str, list[Chunk], bool]:
        """检索知识库; 低置信度(知识库未命中)时自动联网搜索兜底。
        返回 (context, sources, web_used)。"""
        from config import WEB_FALLBACK_THRESHOLD
        from web_search import search_web

        hits = self.kb.retrieve(query, top_k=TOP_K)
        best = hits[0][1] if hits else 0.0
        sources = [c for c, _ in hits]
        kb_context = "\n\n".join(f"[{c.source}]\n{c.text}" for c, _ in hits)
        if best >= WEB_FALLBACK_THRESHOLD:
            return kb_context, sources, False
        # 知识库未命中(无结果或置信度低) -> 网络搜索兜底；若仍有弱相关 KB 结果则合并提供
        web_text = search_web(query)
        if web_text:
            web_block = ("WEB SEARCH RESULTS / 网络搜索结果"
                         "（知识库未命中，内容来自互联网，仅供参考，请提醒用户核实）:\n" + web_text)
            context = (kb_context + "\n\n" + web_block) if kb_context else web_block
            return context, sources, True
        if hits:
            return kb_context, sources, False
        return "", [], False

    # ---------- 对话 ----------

    def _in_domain(self, msg: str) -> bool:
        return True            # 恒放行

    def chat(self, user_msg: str, session_id: str = "default",
             stream: bool = True, model_mode: str = "kimi"):
        """返回 (assistant_reply, sources, web_used)。stream=True 时 assistant_reply 为生成器。
        model_mode: "kimi" = 在线大模型 + RAG; "local" = 本地微调模型。
        非选课领域问题直接礼貌拒绝（sources 返回 None）；知识库未命中时自动联网搜索兜底。"""
        # 配置错误优先级最高：在线模式下缺 Key 直接报错，
        # 不进入寒暄/档案确认/领域守门等固定话术分支。
        if model_mode != "local" and self.client is None:
            raise RuntimeError(
                "未配置在线大模型 API Key。请编辑 .env 填入 LLM_API_KEY（推荐免费的 Groq: "
                "https://console.groq.com/keys，详见 .env.example），或切换为本地微调模型引擎。")
        small_talk = None
        if _GREET_RE.match(user_msg.strip()):
            small_talk = GREETING_REPLY
        elif _THANKS_RE.match(user_msg.strip()):
            small_talk = THANKS_REPLY
        elif _BYE_RE.match(user_msg.strip()):
            small_talk = BYE_REPLY
        if small_talk:
            self._push(session_id, user_msg, small_talk)
            if not stream:
                return small_talk, None, False
            def _greet():
                yield small_talk
            return _greet(), None, False
        # 学生档案：先提取（用户回答档案问题的消息不应被守门拦截）
        self._extract_profile(user_msg, session_id)
        has_profile_info = self._contains_profile_info(user_msg)
        advisory = self._is_advisory(user_msg)
        # 档案补充类短消息（如“我是计算机的”“大二”）→ 直接确认记录，不走守门
        if has_profile_info and not advisory and (
                len(user_msg) <= 25 or not self.profile_complete(session_id)):
            reply = (f"已记录你的背景：{self.profile_text(session_id)} ✅\n\n"
                     "还有想补充或修改的吗？没有的话，直接告诉我你的需求"
                     "（如：帮我规划 AI 方向的选课路线），我会结合你的背景给出建议～")
            self._push(session_id, user_msg, reply)
            if not stream:
                return reply, None, False
            def _record():
                yield reply
            return _record(), None, False

        if not self._in_domain(user_msg):
            self._push(session_id, user_msg, DOMAIN_REFUSAL)
            if not stream:
                return DOMAIN_REFUSAL, None, False
            def _refuse():
                yield DOMAIN_REFUSAL
            return _refuse(), None, False

        # 规划/推荐类问题缺档案时先反问
        if advisory and not self.profile_complete(session_id):
            reply = ASK_PROFILE_REPLY
            self._push(session_id, user_msg, reply)
            if not stream:
                return reply, None, False
            def _ask():
                yield reply
            return _ask(), None, False
        if model_mode == "local":
            gen, sources, web_used = self._chat_local(user_msg, session_id)
            if not stream:
                reply = "".join(gen)
                return reply, sources, web_used
            return gen, sources, web_used
        # 到这里 self.client 一定存在（已在 chat() 开头校验）
        context, sources, web_used = self.retrieve_context(user_msg)
        history = self._histories[session_id]
        system = (SYSTEM_PROMPT.format(context=context)
                  + "\n\nSTUDENT PROFILE / 学生背景:\n" + self.profile_text(session_id)
                  + "\n请结合学生背景给出个性化建议；背景中的“未知”表示学生尚未提供。")
        messages = [{"role": "system", "content": system}]
        messages += list(history)
        messages.append({"role": "user", "content": user_msg})

        kwargs = {"model": LLM_MODEL, "messages": messages, "stream": stream}
        if LLM_TEMPERATURE is not None:
            kwargs["temperature"] = LLM_TEMPERATURE
        resp = self.client.chat.completions.create(**kwargs)
        if not stream:
            reply = resp.choices[0].message.content
            self._push(session_id, user_msg, reply)
            return reply, sources, web_used

        def gen():
            parts: list[str] = []
            for event in resp:
                delta = event.choices[0].delta.content or ""
                parts.append(delta)
                yield delta
            self._push(session_id, user_msg, "".join(parts))

        return gen(), sources, web_used

    # ---------- 本地模型的确定性保护 ----------

    # 知识库没有收录的字段: 问这些直接如实回答, 不交给小模型（它会编造）
    _NONKB_FIELDS = [
        (r"老师|教师|教授|教授|professor|teacher|lecturer|teach", "任课教师姓名"),
        (r"教材|课本|textbook|book", "教材信息"),
        (r"考试日期|几号考|什么时候考试|何时考试|考试在第几|exam date|when is the (final|exam)", "具体考试日期"),
        (r"教室|课室|在哪个教室|classroom|room", "上课教室"),
        (r"多少人|人数|选课人数|多少学生|students|enroll", "选课人数"),
        (r"平均成绩|成绩分布|平均分|grade", "成绩分布"),
    ]

    # 知识库收录的字段: 从资料块中提取事实写入提示词，强制小模型照搬
    _FIELD_FACT_HINTS = [
        (r"先修|pre.?requisite", "先修要求 Prerequisites"),
        (r"学分|credit", "学分 Credits"),
        (r"什么时候开课|开课|哪个学期|开课学期|semester|when is .* offered|offered", "开课学期 Semester"),
        (r"考核|怎么评|评估方式|assessment", "考核方式 Assessment"),
    ]

    def _local_field_guard(self, user_msg: str, sources: list[Chunk]):
        """返回 (canned_reply 或 None, fact_hint)。
        问知识库未收录字段 -> 固定如实回复；问收录字段 -> 从资料提取事实提示词。"""
        top = sources[0] if sources else None
        code_m = re.search(r"课程代码 Course Code: (\S+)", top.text) if top else None
        name_m = re.search(r"课程名称 Course Title: ([^\n]+)", top.text) if top else None
        code = code_m.group(1) if code_m else "该课程"
        name = name_m.group(1) if name_m else ""

        for pat, label in self._NONKB_FIELDS:
            if re.search(pat, user_msg, re.IGNORECASE):
                reply = (f"抱歉，知识库中没有收录 {code}《{name}》的{label}。"
                         f"建议你登录学校选课系统或联系学系办公室查询。")
                return reply, ""
        for pat, prefix in self._FIELD_FACT_HINTS:
            if re.search(pat, user_msg, re.IGNORECASE) and top:
                line_m = re.search(rf"^{re.escape(prefix)}: (.+)$", top.text, re.MULTILINE)
                if line_m:
                    return None, f"事实参考（必须照搬）：{code} 的{prefix.split()[0]}是：{line_m.group(1)}。"
        return None, ""

    def _chat_local(self, user_msg: str, session_id: str):
        """本地微调模型：微调赋予领域问答风格与知识，RAG 检索结果作为上下文兜底保证事实准确。"""
        import local_llm

        # 本地引擎完全离线：不联网搜索，只用知识库检索（web 文本会干扰 0.5B 小模型）
        context, sources = build_local_context(self.kb, user_msg,
                                               self.profile_text(session_id))
        web_used = False

        # 未收录字段（老师/教材/考试日期等）-> 固定如实回复，不交给小模型编造
        canned, fact_hint = self._local_field_guard(user_msg, sources)
        # 课程事实类问题 -> 直接从知识库组装确定性回答（离线准确优先）
        if not canned:
            canned = _local_course_reply(user_msg)
        if canned:
            def _canned():
                yield canned
            self._push(session_id, user_msg, canned)
            return _canned(), sources, web_used

        # 小模型 prefill 慢: 事实提示词排在上下文最前，强制小模型照搬
        if fact_hint:
            context = fact_hint + "\n" + context
        # 0.5B 小模型处理不了多轮历史（会被带偏），本地模式每题独立回答；
        # 学生背景通过上方 context 注入，个性化不受影响
        history: list[dict] = []

        def gen():
            parts: list[str] = []
            for delta in local_llm.chat_stream(user_msg, history, context=context):
                parts.append(delta)
                yield delta
            self._push(session_id, user_msg, "".join(parts))

        return gen(), sources, web_used

    def _push(self, session_id: str, user_msg: str, reply: str) -> None:
        h = self._histories[session_id]
        h.append({"role": "user", "content": user_msg})
        h.append({"role": "assistant", "content": reply})
