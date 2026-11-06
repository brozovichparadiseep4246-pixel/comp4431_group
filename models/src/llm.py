"""在线大模型封装：RAG 提示词、领域守门、多轮历史管理、流式输出。"""
from __future__ import annotations

import re
from collections import defaultdict, deque

from openai import OpenAI

from config import (LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE,
                    MAX_HISTORY_TURNS, TOP_K)
from knowledge_base import Chunk, KnowledgeBase

SYSTEM_PROMPT = """You are "CourseMate", an AI academic advising chatbot for university course selection.
你是选课咨询机器人 "CourseMate"，为大学生提供选课建议。

Rules:
1. Answer based on the KNOWLEDGE BASE provided below. Do not invent course codes, credits, prerequisites or policies.
2. If the knowledge base contains a "WEB SEARCH RESULTS" section (知识库未命中时的网络搜索补充), you may use it to answer, but MUST tell the user the information comes from the internet and may need verification.
3. If neither the knowledge base nor web results contain the answer, say so honestly and suggest the student check with the department office.
4. Always reply in the SAME language as the user's latest message (Chinese or English).
5. Be concise, friendly and helpful. Use bullet points when listing courses.
6. For study-plan questions, consider prerequisites before recommending a course.
7. The knowledge base currently ONLY covers COMP (Computer Science) courses. If the student's major/question is outside this scope (e.g. nursing, business, engineering), clearly state that the knowledge base does not cover that major and suggest they consult the corresponding school/department. NEVER invent course codes or course names.
8. If the knowledge base (and web results, if any) do not contain a specific piece of information — such as the professor's name, textbook, exact exam dates, class size, grade distribution — you MUST honestly say the information is not available ("知识库中没有收录该信息"). NEVER make up names, numbers, dates or any details.

The KNOWLEDGE BASE below was retrieved for the user's question:
---
{context}
---
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


class ChatBot:
    def __init__(self, kb: KnowledgeBase):
        # 无 API Key 时不报错, 仅 Kimi 引擎不可用（可用本地微调模型引擎）
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

    @staticmethod
    def _is_advisory(msg: str) -> bool:
        m = msg.lower()
        return any(k in m for k in ADVISORY_KEYWORDS)

    @staticmethod
    def _contains_profile_info(msg: str) -> bool:
        if _PROFILE_YEAR_RE.search(msg) or _PROFILE_SCHOOL_RE.search(msg):
            return True
        for rx in (_PROFILE_MAJOR_A_RE, _PROFILE_MAJOR_B_RE, _PROFILE_MAJOR_C_RE, _PROFILE_MAJOR_D_RE, _PROFILE_MAJOR_E_RE):
            m = rx.search(msg)
            if m and _clean_major(m.group(1)):
                return True
        return False

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
        """判断问题是否与选课/学业咨询相关。关键词初筛 + 在线大模型二次判定。"""
        m = msg.lower()
        if re.search(r"[a-z]{2,8}\s?\d{3,4}", m):  # 含课程代码
            return True
        if any(k in m for k in OFFDOMAIN_KEYWORDS):
            return False
        if any(k in m for k in DOMAIN_KEYWORDS):
            return True
        # 关键词未命中时, 用在线大模型做一次分类（本地模型太小, 不启用）
        if self.client is None:
            return False
        try:
            judge_kwargs = {"model": LLM_MODEL, "max_tokens": 4}
            if LLM_TEMPERATURE is not None:   # 部分模型(如 kimi-for-coding)不允许自定义 temperature
                judge_kwargs["temperature"] = 0
            r = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content":
                     "你是严格的问题分类器。只有问题涉及大学选课、课程信息、学分、先修要求、"
                     "考试/考核安排、课表、毕业设计/capstone、学业规划时才输出 YES；"
                     "校园生活、社团、娱乐、理财、新闻、编程、情感等一切其他话题一律输出 NO。"
                     "只输出 YES 或 NO，不要输出其他内容。\n"
                     "例：'加退选时间' YES；'机器学习先修课' YES；'怎么理财' NO；'推荐电影' NO；"
                     "'天气如何' NO；'帮我写代码' NO。"},
                    {"role": "user", "content": msg},
                ],
                **judge_kwargs)
            return "YES" in (r.choices[0].message.content or "").upper()
        except Exception as e:
            print(f"[Guard] 分类器调用失败，按不相关处理: {e}")
            return False

    def chat(self, user_msg: str, session_id: str = "default",
             stream: bool = True, model_mode: str = "kimi"):
        """返回 (assistant_reply, sources, web_used)。stream=True 时 assistant_reply 为生成器。
        model_mode: "kimi" = 在线大模型 + RAG; "local" = 本地微调模型。
        非选课领域问题直接礼貌拒绝（sources 返回 None）；知识库未命中时自动联网搜索兜底。"""
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
        if self.client is None:
            raise RuntimeError(
                "未配置在线大模型 API Key。请编辑 .env 填入 LLM_API_KEY（推荐免费的 Groq: "
                "https://console.groq.com/keys，详见 .env.example），或切换为本地微调模型引擎。")
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
        hits = self.kb.retrieve(user_msg, top_k=TOP_K)
        web_used = False
        # 查询中明确提到的课程代码对应的资料块排在最前，减少小模型张冠李戴
        sources = sorted([c for c, _ in hits], key=lambda c: 0 if c.source in user_msg else 1)

        # 未收录字段（老师/教材/考试日期等）-> 固定如实回复，不交给小模型编造
        canned, fact_hint = self._local_field_guard(user_msg, sources)
        if canned:
            def _canned():
                yield canned
            self._push(session_id, user_msg, canned)
            return _canned(), sources, web_used

        # 小模型 prefill 慢: 只保留最相关的 3 条并截断, 缩短首句等待
        sources = sources[:3]
        context = "\n\n".join(f"[{c.source}]\n{c.text}" for c in sources)[:700]
        if fact_hint:
            context = fact_hint + "\n" + context
        context = (f"学生背景：{self.profile_text(session_id)}（未知表示未提供，请结合背景回答）\n\n"
                   + context)
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
