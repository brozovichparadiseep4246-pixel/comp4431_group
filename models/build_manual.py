# -*- coding: utf-8 -*-
"""生成《CourseMate 使用说明书.docx》（详细版：使用 + 配置 + 维护 + 技术实现）
运行:  python build_manual.py
"""
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

BLUE = RGBColor(0x2F, 0x6F, 0xED)
DARK = RGBColor(0x1F, 0x29, 0x37)
GRAY = RGBColor(0x6B, 0x72, 0x80)
MONO = RGBColor(0x0B, 0x5C, 0xAB)

doc = Document()
style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(11)
style.element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")


def set_cn(run, size=11, bold=False, color=None, mono=False):
    run.font.name = "Consolas" if color == MONO else "Calibri"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")
    run.font.size = Pt(size)
    run.font.bold = bold
    if color:
        run.font.color.rgb = color


def title(text, sub=None):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_cn(p.add_run(text), 22, True, DARK)
    if sub:
        p2 = doc.add_paragraph(); p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_cn(p2.add_run(sub), 12, False, GRAY)


def h1(text):
    p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(16)
    set_cn(p.add_run(text), 15, True, BLUE)


def h2(text):
    p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(8)
    set_cn(p.add_run(text), 12.5, True, DARK)


def para(text, size=11, bold=False, color=None, indent=0):
    p = doc.add_paragraph()
    if indent:
        p.paragraph_format.left_indent = Pt(indent)
    set_cn(p.add_run(text), size, bold, color)
    return p


def bullet(text, prefix=None):
    p = doc.add_paragraph(style="List Bullet")
    if prefix:
        set_cn(p.add_run(prefix), 11, True)
    set_cn(p.add_run(text), 11)


def numbered(text, prefix=None):
    p = doc.add_paragraph(style="List Number")
    if prefix:
        set_cn(p.add_run(prefix), 11, True)
    set_cn(p.add_run(text), 11)


def code(text):
    p = doc.add_paragraph(); p.paragraph_format.left_indent = Pt(18)
    set_cn(p.add_run(text), 10.5, False, MONO, mono=True)


def kv(label, text):
    """粗体标签 + 内容 的项目符号"""
    bullet(text, prefix=label + "：")


# ================= 封面 =================
title("CourseMate 选课咨询聊天机器人",
      "COMP4431 Artificial Intelligence · Group Project · 使用说明书（详细版）")
para("")

# ================= 一、项目简介 =================
h1("一、项目简介")
para("CourseMate 是一个面向大学生选课咨询场景的 AI 聊天机器人。用户可以用文字或语音询问课程信息、"
     "先修要求、学分与考核方式、选课政策与学习路线规划等问题，机器人基于课程知识库（RAG 向量检索）"
     "给出准确回答，知识库未命中时自动联网搜索补充；界面右上角的 2D 卡通吉祥物会随对话实时变化"
     "表情动作（思考、眨眼、对口型说话等）。")
para("本助手只回答选课 / 学业规划相关问题，其他话题会礼貌拒绝（领域守门功能）。"
     "回答规划/推荐类问题前会先了解学生的专业与年级（学校默认香港理工大学 PolyU），"
     "再给出个性化建议。", bold=True)

# ================= 二、功能特性 =================
h1("二、功能特性")
h2("基本功能")
bullet("选课领域问答：课程内容、学分、开课学期、先修要求、考核方式、选课政策、学习路线规划")
bullet("中英双语：自动检测用户语言并用同语言回复")
bullet("多轮对话记忆：支持追问（如“这门课有先修要求吗？→ 那它什么时候开课？”）")
bullet("学生档案：规划/推荐类问题会先了解专业、年级（默认 PolyU），自动提取记忆后个性化建议")
bullet("寒暄与领域守门：打招呼友好回应；非选课问题礼貌拒绝（关键词 + AI 语义判定混合）")
h2("附加功能")
bullet("RAG 知识库（向量检索）：课程知识向量化存储，余弦相似度 Top-K 检索；检索置信度低时"
       "自动调用 DuckDuckGo 网络搜索兜底，并明确标注信息来自网络", "1. ")
bullet("语音识别与合成：faster-whisper 本地离线语音识别 + edge-tts 神经网络语音合成，"
       "支持语音提问与语音播报", "2. ")
bullet("2D 虚拟形象：纯 SVG+JS 卡通吉祥物，待机小动作（张望/弹跳/绿叶摇摆/眨眼）、"
       "思考/说话/微笑状态联动，语音播报时嘴巴按音频振幅对口型", "3. ")
bullet("微调大语言模型：使用 LoRA 自行微调 Qwen2.5-0.5B-Instruct（CPU 可训练），"
       "界面可一键切换到本地微调模型，完全离线运行", "4. ")
bullet("友好的用户界面：DeepSeek 风格清爽聊天界面，气泡对话、流式输出、示例问题卡片", "5. ")

# ================= 三、运行环境 =================
h1("三、运行环境与准备工作")
bullet("操作系统：Windows 10 / 11（Linux / macOS 亦可，用命令行方式运行）")
bullet("Python 3.11 或更高版本：https://www.python.org/downloads/ ，"
       "安装时务必勾选 “Add python.exe to PATH”")
bullet("（可选）安装 uv 可加快环境搭建：https://docs.astral.sh/uv/ ，不装也不影响")
bullet("网络：首次运行需下载依赖与模型文件；之后使用“本地微调”引擎对话时无需联网")

# ================= 四、启动步骤 =================
h1("四、启动步骤")
numbered("解压压缩包到任意文件夹（路径不要包含特殊字符）")
numbered("双击 run.bat")
para("首次运行会自动创建虚拟环境并安装依赖（约 3–10 分钟），随后自动构建课程知识库索引，"
     "最后启动服务并自动打开浏览器访问 http://127.0.0.1:7860 。")
para("注意：运行期间请保持黑色命令行窗口开启，关闭该窗口即停止服务。", bold=True)

# ================= 五、使用指南 =================
h1("五、使用指南")
h2("5.1 文字提问")
numbered("在底部输入框输入选课问题（支持中文 / 英文），按回车或点击右侧蓝色按钮发送")
numbered("输入框发送后自动清空；机器人流式输出回答，下方状态栏显示知识来源")
numbered("点“清空”可清空对话并重置多轮记忆与学生档案")
h2("5.2 语音对话")
numbered("点击中部“🎤 语音输入”的圆形按钮开始录音，说完后再次点击停止")
numbered("点击“🎤 提问”，机器人将识别语音并回答")
numbered("勾选“🔊 语音播报”后，回答会自动朗读；右上角吉祥物会同步对口型")
h2("5.3 切换模型引擎")
para("界面右下角“模型引擎”可在两种模式间切换：")
bullet("在线大模型（RAG 检索）：回答质量高，需配置免费 API Key（见第六节）")
bullet("本地微调（离线）：使用本组自行 LoRA 微调的 Qwen2.5-0.5B 模型，无需 Key 和联网，"
       "首次切换需自动下载基座模型（约 1GB），回答约 2–5 秒/条")
h2("5.4 示例问题")
bullet("COMP4431 人工智能这门课主要学什么？考核方式是什么？")
bullet("我想下学期学机器学习，需要先修哪些课？")
bullet("Which database course should I take and what are its prerequisites?")
bullet("加退选是什么时候？如果没满足先修课要求怎么办？")
bullet("帮我规划一条学习人工智能方向的选课路线")

# ================= 六、API Key 配置 =================
h1("六、在线大模型 API Key 配置（可选，免费方案）")
para("不配 Key 不影响演示——直接使用“本地微调（离线）”引擎即可。如需“在线大模型”引擎，"
     "任选下列免费服务（约 2 分钟注册完成，无需绑定信用卡）：", bold=True)
h2("★ 推荐：Groq（开源模型，免费额度充足）")
numbered("打开 https://console.groq.com/keys ，注册并创建一个 API Key（以 gsk_ 开头的一串字符）")
numbered("打开项目文件夹（即 run.bat 所在的文件夹），找到 .env.example 文件")
para("     按住 Ctrl 拖动复制一份，把副本重命名为 .env", indent=18)
para("     （注意：全名就是 .env，不是 .env.txt；若系统隐藏扩展名，请在资源管理器"
         "“查看 → 文件扩展名”处确认）", indent=18, color=GRAY)
numbered("用记事本打开 .env，找到“方案一”标题下面这一行：")
code("# LLM_API_KEY=gsk_xxxxxxxxxxxxxxxx")
para("     把行首的 # 删掉，并把 gsk_xxxxxxxxxxxxxxxx 替换成你的 Key，改完后应为：", indent=18)
code("LLM_API_KEY=gsk_你的真实key")
numbered("保存文件，重新双击 run.bat 启动即可")
para("说明：LLM_BASE_URL 与 LLM_MODEL 两行的默认值即为 Groq，保持被注释状态即可，无需改动；"
     "如默认模型不可用，可参照 https://console.groq.com/docs/models 更换为其他开源模型名。",
     10.5, False, GRAY)
h2("其他可选免费服务（配置方法详见 .env.example 内注释）")
bullet("OpenRouter 免费模型（模型名带 :free 后缀）")
bullet("Google Gemini 免费额度")
bullet("Moonshot Kimi（国内服务，赠送少量额度）")

# ================= 七、项目目录结构与文件职责 =================
h1("七、项目目录结构与文件职责")
para("解压后的文件夹结构及各文件用途如下：")
code(
    "chatbot/\n"
    "├── run.bat                  一键启动脚本（Windows 双击运行）\n"
    "├── README.md                项目说明（含组员名单、实现原理）\n"
    "├── 使用说明书.docx / 使用说明.txt   本文档\n"
    "├── requirements.txt         Python 依赖清单\n"
    "├── .env.example             在线大模型 API 配置模板（复制改名为 .env 使用）\n"
    "├── build_kb.py              构建/重建知识库向量索引\n"
    "├── build_dataset.py         从课程知识库自动生成微调训练数据\n"
    "├── finetune.py              LoRA 微调脚本（CPU 约 6 分钟）\n"
    "├── eval.py                  评测脚本（检索命中率 + 回答质量）\n"
    "│\n"
    "├── src/                     核心源代码目录\n"
    "│   ├── app.py               网页界面（Gradio）：界面布局、事件、头像/语音联动\n"
    "│   ├── llm.py               ★ 回答主逻辑：寒暄→领域守门→学生档案→RAG→联网兜底→生成\n"
    "│   ├── knowledge_base.py    RAG 向量检索（嵌入模型、相似度计算、课程代码置顶）\n"
    "│   ├── local_llm.py         本地微调模型推理（流式输出、句子去重）\n"
    "│   ├── voice.py             语音识别(faster-whisper)与语音合成(edge-tts)\n"
    "│   ├── web_search.py        网络搜索兜底（DuckDuckGo，免费无需 Key）\n"
    "│   ├── avatar.py            2D 卡通吉祥物（SVG+JS：表情、口型、小动作）\n"
    "│   └── config.py            全局配置（API、模型名、阈值、线程数等）\n"
    "│\n"
    "├── data/\n"
    "│   ├── courses.json         ★ 课程知识库（课程代码/学分/先修/考核方式等，可自由修改）\n"
    "│   ├── finetune_data.jsonl  微调训练数据（build_dataset.py 生成）\n"
    "│   └── kb_docs/             自由格式知识文档（.txt/.md，放入即纳入检索）\n"
    "│\n"
    "├── models/qwen-lora/        微调好的 LoRA 模型参数（16MB，可直接使用）\n"
    "└── vector_index/            知识库向量索引（build_kb.py 生成，无需改动）"
)

# ================= 八、技术实现说明（想改哪里看哪里） =================
h1("八、技术实现说明（想改哪里，看哪里）")

h2("8.1 回答逻辑（一段提问是如何被处理的）")
para("完整的回答处理链路在 src/llm.py 的 ChatBot.chat() 方法中，流程如下：")
numbered("寒暄识别：整条消息是打招呼/道谢/告别时，直接回复固定话术（llm.py 顶部 _GREET_RE 等）")
numbered("领域守门：明显选课问题（含课程代码或关键词）放行；明显无关话题拒绝；"
         "拿不准的交给大模型做语义分类（llm.py 的 _in_domain()，关键词表在 DOMAIN_KEYWORDS / "
         "OFFDOMAIN_KEYWORDS，想增删识别词直接改这两个列表）")
numbered("学生档案：自动从消息提取专业/年级/学校（_PROFILE_*_RE 正则）；规划/推荐类问题"
         "且档案不全时先反问（ADVISORY_KEYWORDS 控制哪些问法算“规划类”）")
numbered("RAG 检索：知识库向量相似度检索 Top-5；含课程代码（如 COMP4431）时对应资料置顶")
numbered("联网兜底：检索最高置信度低于阈值时，调用 src/web_search.py 做 DuckDuckGo 搜索"
         "（阈值在 config.py 的 WEB_FALLBACK_THRESHOLD，默认 0.45）")
numbered("生成回答：在线引擎调用配置的 API（模型名/地址在 .env 或 config.py）；"
         "本地引擎调用 models/qwen-lora 微调模型（src/local_llm.py）")
numbered("界面联动：流式输出 + 语音播报 + 吉祥物状态（src/app.py 的 respond()）")

h2("8.2 知识库在哪里、怎么改")
para("知识库有两个来源，修改后需重新运行 python build_kb.py 重建索引：")
kv("data/courses.json", "结构化课程数据。每门课一个 JSON 条目，含代码、中英文名、学分、"
   "开课学期、先修要求、简介、考核方式。增删课程直接编辑此文件即可")
kv("data/kb_docs/", "自由格式的知识文档（.txt / .md），例如选课 FAQ。把任何文本文件放进这个"
   "文件夹，机器人就能检索到")
para("注：当前 courses.json 为演示用示例数据，正式使用请替换为学校官方课程信息。",
     10.5, False, GRAY)

h2("8.3 微调模型如何重新训练")
para("若修改了课程知识库并想让本地微调模型也掌握新知识：")
numbered("运行 python build_dataset.py（自动从知识库生成约 250 条中英问答训练数据）")
numbered("运行 python finetune.py（LoRA 微调，CPU 约 6 分钟，输出到 models/qwen-lora/）")
para("训练参数（LoRA rank、学习率、轮数）在 finetune.py 顶部可调。")

h2("8.4 其他可调参数（config.py）")
kv("TOP_K", "检索返回的资料条数，默认 5")
kv("WEB_FALLBACK_THRESHOLD", "检索置信度低于该值时联网搜索，默认 0.45（调大更容易联网）")
kv("WHISPER_MODEL", "语音识别模型，默认 tiny（可改 base/small 提升准确率、变慢）")
kv("TTS_VOICE_ZH / TTS_VOICE_EN", "语音播报发音人，默认晓晓（中）/ Aria（英）")
kv("MAX_HISTORY_TURNS", "多轮记忆轮数，默认 10")

# ================= 九、常见问题 =================
h1("九、常见问题（FAQ）")
faq = [
    ("浏览器没有自动打开？",
     "手动在浏览器地址栏输入 http://127.0.0.1:7860 ，回车即可。"),
    ("双击 run.bat 后闪退，或提示“未找到 Python”？",
     "说明未安装 Python 或未勾选 Add to PATH。请先按第三节安装 Python 3.11+ 后重试。"),
    ("首次使用语音识别 / 本地模型时等待很久？",
     "正常现象——首次使用需从 HuggingFace 下载模型（语音识别约 75MB，本地大模型基座约 1GB），"
     "只下载一次，之后永久缓存。"),
    ("提示端口 7860 被占用？",
     "先关闭其他占用该端口的程序；或设置环境变量 GRADIO_SERVER_PORT=其他端口 后重新运行。"),
    ("机器人拒绝回答我的问题？",
     "本助手限定于选课 / 学业规划领域，其他话题会被礼貌拒绝，属于正常的领域守门行为。"),
    ("回答中出现知识不准确？",
     "当前课程数据为演示示例，请替换 data/courses.json 后重建索引；"
     "知识库未命中时回答来自网络搜索，界面会有明确提示，请以学校官方信息为准。"),
    ("打包提交压缩包时要注意什么？",
     "请勿打包 .venv 文件夹（依赖环境，约 1.3GB，老师运行时会自动重新安装）；"
     "其余文件（含 models/qwen-lora 与 vector_index）都应包含。"),
]
for q, a in faq:
    para("Q：" + q, bold=True)
    para("A：" + a)

# ================= 十、组员名单 =================
h1("十、组员名单")
t = doc.add_table(rows=3, cols=2)
t.style = "Light Grid Accent 1"
t.rows[0].cells[0].paragraphs[0].add_run("姓名 Name")
t.rows[0].cells[1].paragraphs[0].add_run("学号 Student ID")
t.rows[1].cells[0].paragraphs[0].add_run("Dingziang")
t.rows[1].cells[1].paragraphs[0].add_run("23111092D")
t.rows[2].cells[0].paragraphs[0].add_run("（请补充）")
t.rows[2].cells[1].paragraphs[0].add_run("（请补充）")
for row in t.rows:
    for c in row.cells:
        for p in c.paragraphs:
            for r in p.runs:
                set_cn(r, 11)

para("")
para("建议演示流程（约 7 分钟）：文字问答（展示 RAG 知识来源）→ 语音对话（展示对口型）→ "
     "规划类问题（展示学生档案反问与个性化建议）→ 切换本地微调引擎（展示微调成果）→ "
     "询问非选课问题（展示领域守门）。", 10.5, False, GRAY)

doc.save("CourseMate 使用说明书.docx")
print("已生成: CourseMate 使用说明书.docx")
