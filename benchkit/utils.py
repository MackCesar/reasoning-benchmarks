import json, time
from pathlib import Path
from typing import Any, Dict, Iterable, List

def now_ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def save_jsonl(path:str | Path, rows: Iterable[Dict[str, Any]]):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w",encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r,ensure_ascii=False)+"\n")

def load_json(path: str | Path) -> List[Dict[str, Any]]:
    with Path(path).open("r",encoding="utf-8") as f:
        return json.load(f)