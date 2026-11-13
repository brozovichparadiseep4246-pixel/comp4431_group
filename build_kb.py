"""构建/重建向量知识库索引。修改 data/ 下数据后请重新运行本脚本。"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from knowledge_base import KnowledgeBase

if __name__ == "__main__":
    kb = KnowledgeBase()
    kb.build()
    print("完成。现在可以运行: python src/app.py")
