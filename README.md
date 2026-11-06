# CourseMate — 选课咨询聊天机器人（COMP4431 Group Project）

一个面向大学生选课咨询场景的 AI 聊天机器人：基于 **Kimi (Moonshot) 大模型 + RAG 课程知识库**，
支持**中英双语**、**语音输入与语音播报**、**多轮对话记忆**和 **Gradio 网页界面**。

## 组员名单 Group Members

| 姓名 Name | 学号 Student ID |
|-----------|-----------------|
| Dingziang | 23111092D       |
| （请补充） |                 |
| （请补充） |                 |

> 提交前请填写完整组员名单（2-4 人）。

## 快速开始 Quick Start

### 方式一：一键启动（Windows，推荐）

双击 `run.bat`：首次运行自动安装依赖并构建知识库索引（约 3-10 分钟），
之后浏览器自动打开 `http://127.0.0.1:7860`。**保持命令行窗口开启**，关闭即停止服务。

> 在**另一台电脑**（如演示教室机器）上运行：只需安装 Python 3.11+（勾选 Add to PATH）
> 或 uv，然后双击 `run.bat` 即可。详细图文步骤见随附的 **`使用说明.txt`**。

### 方式二：命令行

```bash
# 1. 创建环境（需要 uv，或自行用 python -m venv 替代）
uv venv .venv
uv pip install -r requirements.txt

# 2. （可选）配置在线大模型 API Key（推荐免费方案 Groq，2 分钟注册）
cp .env.example .env   # 填入 LLM_API_KEY（详见 .env.example 内四种免费方案）
#    不配置时自动尝试复用本机 Kimi CLI 登录凭证；都没有也可用
#    界面右下角切换为“本地微调”引擎，完全离线运行

# 3. 构建知识库向量索引（修改 data/ 后需重新执行）
python build_kb.py

# 4. 启动
python src/app.py
```

## 功能与实现方式 Features

### 基本功能（Basic Features）

- **双引擎问答**：默认使用**在线大模型（OpenAI 兼容 API，支持 Groq 等免费服务）+ RAG 检索**在线回答；也可切换到**本地 LoRA 微调模型**完全离线回答。两者均可处理课程内容、学分、先修要求、考核方式、选课政策、学习路线规划等选课问题。
- **中英双语**：自动检测用户语言并用同语言回复。
- **领域守门（Domain Guard）**：只回答选课/学业相关问题。三层判定：
  明显选课问题（课程代码/关键词）直接放行；明显无关话题（电影/理财/游戏等）直接拒绝；
  **拿不准的模糊问题交给大模型用 AI 语义分类判定**（带正反例提示词），如"我绩点低想选水课"
  会被正确识别为选课问题，"写个冒泡排序"会被拒绝。非选课问题不会触发检索或联网搜索。

### 附加功能（Additional Features）

1. **RAG 知识库检索（向量数据库）**
   - 课程数据结构化存储于 `data/courses.json`，自由文档放入 `data/kb_docs/`。
   - 使用本地开源嵌入模型 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` 将知识块编码为向量，
     以余弦相似度做 Top-K 检索（向量缓存于 `vector_index/`）。
   - 检索结果作为上下文注入 LLM 提示词，机器人只依据知识库回答，降低幻觉；
     界面同时展示"知识来源"。
   - **网络搜索兜底**：检索最高置信度低于阈值（默认 0.45，可在 .env 调）时，
     自动调用 DuckDuckGo 免费搜索（`src/web_search.py`），把搜索结果与弱相关 KB 结果
     合并提供给 LLM，并明确提示用户"内容来自网络、需要核实"；
     查询中出现课程代码（如 COMP4431）时，对应课程资料块强制置顶，避免张冠李戴。

2. **语音识别与合成（Voice Recognition & Synthesis）**
   - **ASR**：`faster-whisper`（tiny 模型，CPU 本地离线运行）转写麦克风语音输入。
   - **TTS**：`edge-tts`（微软 Edge 神经网络语音）合成回复语音，按回复语言自动选择
     中文/英文发音人；TTS 失败时自动降级为纯文本回复。

3. **2D 虚拟形象（2D Animated Avatar）**
   - 聊天界面右侧内置纯 SVG + JS 动画机器人头像，**无需任何外部资源、离线可用**。
   - 表情状态与对话实时联动：待机时随机眨眼；机器人思考时眼珠上翻、天线摆动、
     出现思考气泡；回答/播报时嘴巴开合，**口型由 TTS 音频振幅驱动**（Web Audio API 分析，
     失败时降级为随机节奏）；回答完毕呈现微笑表情。
   - Python 端通过隐藏状态组件下发表情指令，前端 MutationObserver 监听驱动动画。

4. **微调大语言模型（Fine-tuned LLM, LoRA）**
   - **不是只调 API**：本项目自行对 **Qwen2.5-0.5B-Instruct** 进行 LoRA 微调。
   - 数据：`build_dataset.py` 从课程知识库自动生成 250 条中英双语问答对
     （`data/finetune_data.jsonl`，Qwen chat 模板，仅对回答部分计算损失）。
   - 训练：`finetune.py` 使用 peft + transformers，CPU 即可训练
     （LoRA rank=8, alpha=16, lr=5e-5, 2 epochs, completion-only loss，约 6 分钟），产出 adapter 至 `models/qwen-lora/`。
   - 推理：`src/local_llm.py` 加载基座 + adapter。**微调模型 + RAG 检索增强**的架构：
     微调赋予模型选课领域的问答风格与知识，检索结果同时作为上下文注入进一步保证事实准确
     （并对小模型的重复输出做句子去重后处理）；界面"模型引擎"可一键切换到
     本地微调模型，**断网也能准确回答课程问题**。

5. **用户友好界面（User-friendly Interface）**
   - Gradio 网页聊天界面：渐变 Banner、聊天气泡、流式打字效果、示例问题一键填入、
     麦克风录音、语音回复播放、知识来源提示、一键清空对话、双引擎切换。

6. **多轮对话上下文记忆（Multi-turn Memory）**
   - 每个会话保存最近的多轮对话历史（滑动窗口），支持追问，
     如"这门课有先修要求吗 → 那它什么时候开课？"

7. **学生档案与个性化建议（Student Profile）**
   - 询问规划/推荐类问题时，若尚未了解用户背景，会先反问补充
     （专业/年级，学校默认香港理工大学 PolyU；用户声明其他学校则覆盖）。
   - 用户消息中提到的专业/年级/学校会被正则自动提取并记忆，
     之后结合背景给出个性化选课建议；点"清空对话"时档案一并重置。

## 训练本地模型（可选，约 10-30 分钟 CPU）

```bash
python build_dataset.py   # 从知识库生成 250 条微调数据
python finetune.py        # LoRA 微调 Qwen2.5-0.5B，输出 models/qwen-lora/
```

训练完成后，在界面右下角"模型引擎"选择"本地微调模型"即可离线使用。

## 评测 Evaluation

```bash
python eval.py                  # 检索准确率 + LLM 回答示例（消耗少量 API 额度）
python eval.py --retrieval-only # 只测检索准确率
```

`eval.py` 内置 10 个中英问答对，输出检索命中率（Top-K 中是否包含期望课程）
以及 RAG 回答样例，可用于演示中的 evaluation results。

## 目录结构

```
chatbot/
├── src/
│   ├── config.py          # 配置：API Key、模型名、检索参数
│   ├── knowledge_base.py  # RAG：向量化、索引构建、余弦检索
│   ├── llm.py             # Kimi LLM 封装：提示词、历史记忆、流式输出
│   ├── voice.py           # 语音识别 (faster-whisper) / 语音合成 (edge-tts)
│   ├── avatar.py          # 2D 动画虚拟形象 (SVG + JS)
│   ├── local_llm.py       # 本地 LoRA 微调模型推理
│   └── app.py             # Gradio 网页界面
├── data/
│   ├── courses.json       # 结构化课程知识库（⚠️ 示例数据，请核对/替换）
│   ├── finetune_data.jsonl # 微调数据集（build_dataset.py 生成）
│   └── kb_docs/           # 自由格式知识文档（.txt/.md）
├── models/qwen-lora/      # LoRA adapter（finetune.py 生成）
├── vector_index/          # 向量索引缓存（build_kb.py 生成）
├── build_kb.py            # 构建/重建向量索引
├── build_dataset.py       # 从知识库生成微调数据集
├── finetune.py            # LoRA 微调 Qwen2.5-0.5B（CPU 可训练）
├── eval.py                # 评测脚本
├── requirements.txt
├── run.bat                # Windows 一键启动
└── .env.example           # API Key 配置模板
```

## 自定义知识库

- 修改 `data/courses.json`（增删课程、修改先修要求等），然后重新运行 `python build_kb.py`。
- 把任何 `.txt` / `.md` 文档放进 `data/kb_docs/` 即可纳入检索，同样需重建索引。

## 已知局限与说明 Limitations

- `data/courses.json` 中课程信息为**演示用示例数据**，正式演示前请替换为学校官方课程信息。
- 语音识别使用 tiny 模型以保证 CPU 上快速运行，嘈杂环境下准确率有限。
- 语音合成依赖网络（edge-tts 微软服务），离线时自动降级为纯文本。
- 本地微调模型为 0.5B 小模型，知识以训练数据为界、回答风格较简单，适合作为
  "微调"技术展示与离线兜底；日常高质量回答请使用 Kimi 引擎。
- 本地模型在 CPU 上推理较慢（约 10-30 秒/条），属正常现象。
- 对话历史保存在内存中，服务重启后清空；不同浏览器标签页共享一个默认会话。
- 嵌入模型本地免费运行；Kimi 引擎调用消耗 API 额度，本地微调模型完全免费离线。
