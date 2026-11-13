"""Check native dependencies before starting the web application."""
import importlib
import re
import sys


TORCH_BLOCKED = 2


def is_policy_block(exc: Exception) -> bool:
    """Recognize Windows application-control errors, including wrapped errors."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if getattr(exc, "winerror", None) == 4551:
            return True
        if re.search(r"\bWinError\s+4551\b", str(exc), re.IGNORECASE):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def main() -> int:
    failed = False
    for name in ("numpy", "pandas", "torch", "jiter", "av", "ctranslate2",
                 "sentence_transformers", "peft"):
        try:
            importlib.import_module(name)
            print(f"[OK] {name}", flush=True)
        except Exception as exc:
            print(f"[ERROR] {name}: {exc}", flush=True)
            if name == "torch" and is_policy_block(exc):
                print("[ERROR] Windows application control blocked Torch. "
                      "The launcher may try the CPU compatibility build.", flush=True)
                return TORCH_BLOCKED
            failed = True
    if failed:
        print("[ERROR] Runtime dependencies cannot load. Package metadata alone "
              "does not detect blocked DLLs. Do not disable Windows security.")
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
