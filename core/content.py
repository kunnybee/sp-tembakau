import json
from pathlib import Path

def load_disease_info(json_path: str) -> dict:
    p = Path(json_path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
