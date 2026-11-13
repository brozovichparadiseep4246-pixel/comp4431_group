"""RAG 知识库：本地 sentence-transformers 向量化 + 余弦相似度检索（向量缓存到磁盘）。

索引由 build_kb.py 构建，包含两类知识：
1. data/courses.json 中的结构化课程记录与通用选课信息；
2. data/kb_docs/ 下的自由文档（.txt/.md），按段落切分。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from config import (
    COURSES_JSON,
    EMBED_MODEL,
    INDEX_DIR,
    KB_DOCS_DIR,
    SIMILARITY_THRESHOLD,
    TOP_K,
)

VECTOR_FILE = INDEX_DIR / "vectors.npy"
META_FILE = INDEX_DIR / "meta.json"


@dataclass
class Chunk:
    text: str
    source: str  # 例如 "COMP4431" 或 "kb_docs/xxx.txt"


class KnowledgeBase:
    """加载向量索引并提供 top-k 检索。"""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None
        self._model = None  # 延迟加载 embedding 模型

    # ---------- 构建索引 ----------

    def build(self) -> None:
        chunks = load_courses_chunks() + load_doc_chunks(KB_DOCS_DIR)
        if not chunks:
            raise RuntimeError("知识库为空：请检查 data/courses.json 和 data/kb_docs/")
        model = self._embed_model()
        texts = [c.text for c in chunks]
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        np.save(VECTOR_FILE, np.asarray(embeddings, dtype=np.float32))
        META_FILE.write_text(
            json.dumps([{"text": c.text, "source": c.source} for c in chunks],
                       ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        self._chunks, self._vectors = chunks, np.asarray(embeddings, dtype=np.float32)
        print(f"[KB] 索引已构建：{len(chunks)} 个文本块 -> {INDEX_DIR}")

    # ---------- 加载 / 检索 ----------

    def load(self) -> None:
        if not VECTOR_FILE.exists() or not META_FILE.exists():
            raise RuntimeError(
                "未找到向量索引，请先运行: python build_kb.py")
        self._vectors = np.load(VECTOR_FILE)
        raw = json.loads(META_FILE.read_text(encoding="utf-8"))
        self._chunks = [Chunk(text=r["text"], source=r["source"]) for r in raw]
        print(f"[KB] 已加载 {len(self._chunks)} 个文本块")

    def is_ready(self) -> bool:
        return self._vectors is not None and len(self._chunks) > 0

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[tuple[Chunk, float]]:
        """返回 [(chunk, score), ...]，过滤低相似度结果。
        若查询中包含课程代码（如 COMP4431），该课程的资料块会被置顶。"""
        if not self.is_ready():
            raise RuntimeError("知识库未加载，请先运行 build_kb.py")
        q = self._embed_model().encode([query], normalize_embeddings=True)
        scores = (self._vectors @ q[0]).astype(np.float64)
        idx = np.argsort(-scores)[:top_k]
        results = [(self._chunks[i], float(scores[i])) for i in idx]
        results = [(c, s) for c, s in results if s >= SIMILARITY_THRESHOLD]
        # 关键词置顶: 查询中出现 COMPxxxx 时, 对应课程块排最前
        codes = re.findall(r"[A-Z]{2,6}\s?\d{3,4}", query.upper())
        if codes:
            pinned = []
            for code in codes:
                code = code.replace(" ", "")
                for c in self._chunks:
                    if c.source.upper() == code and all(c.source != p.source for p in pinned):
                        pinned.append(c)
            if pinned:
                rest = [(c, s) for c, s in results if all(c.source != p.source for p in pinned)]
                results = [(p, 1.0) for p in pinned] + rest
        return results

    # ---------- 内部 ----------

    def _embed_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(EMBED_MODEL)
        return self._model


# ---------- 文本块构造 ----------


def _course_to_text(c: dict) -> str:
    return (
        f"课程代码 Course Code: {c.get('code', '')}\n"
        f"课程名称 Course Title: {c.get('title_en', '')} / {c.get('title_zh', '')}\n"
        f"学分 Credits: {c.get('credits', '')}\n"
        f"开课学期 Semester: {c.get('semester', '')}\n"
        f"先修要求 Prerequisites: {c.get('prerequisites', '')}\n"
        f"考核方式 Assessment: {c.get('assessment', '')}\n"
        f"课程简介 Description: {c.get('description_en', '')} {c.get('description_zh', '')}\n"
        f"备注 Remarks: {c.get('remarks', '')}"
    )


def _general_to_text(g: dict) -> str:
    return (
        f"主题 Topic: {g.get('topic_en', '')} / {g.get('topic_zh', '')}\n"
        f"内容 Content: {g.get('content_en', '')} {g.get('content_zh', '')}"
    )


def load_courses_chunks() -> list[Chunk]:
    data = json.loads(COURSES_JSON.read_text(encoding="utf-8"))
    chunks = [Chunk(text=_course_to_text(c), source=c.get("code", "course"))
              for c in data.get("courses", [])]
    chunks += [Chunk(text=_general_to_text(g), source="general_info")
               for g in data.get("general_info", [])]
    return chunks


def load_doc_chunks(docs_dir: Path, max_chars: int = 800) -> list[Chunk]:
    """把 kb_docs 下的 .txt/.md 按空行/标题切成文本块。"""
    if not docs_dir.exists():
        return []
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.rglob("*")):
        if path.suffix.lower() not in {".txt", ".md"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        parts = re.split(r"\n\s*\n", text)
        buf = ""
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(buf) + len(part) + 2 > max_chars and buf:
                chunks.append(Chunk(text=buf.strip(), source=f"kb_docs/{path.name}"))
                buf = part
            else:
                buf = f"{buf}\n\n{part}" if buf else part
        if buf:
            chunks.append(Chunk(text=buf.strip(), source=f"kb_docs/{path.name}"))
    return chunks


if __name__ == "__main__":
    kb = KnowledgeBase()
    kb.build()
