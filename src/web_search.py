"""网络搜索兜底：知识库未命中时，用 DuckDuckGo 免费搜索补充信息。"""
from __future__ import annotations


def search_web(query: str, max_results: int = 5, snippet_chars: int = 240) -> str:
    """返回拼接好的搜索结果文本；失败或无结果返回空字符串。"""
    try:
        try:
            from ddgs import DDGS
        except ImportError:  # 兼容旧包名
            from duckduckgo_search import DDGS

        with DDGS(timeout=15) as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return ""
        blocks = []
        for i, r in enumerate(results, 1):
            body = (r.get("body") or "")[:snippet_chars]
            blocks.append(f"[网络结果 {i}] {r.get('title', '')}\n{body}\n来源: {r.get('href', '')}")
        return "\n\n".join(blocks)
    except Exception as e:
        print(f"[WebSearch] 搜索失败（将按无结果处理）: {e}")
        return ""
