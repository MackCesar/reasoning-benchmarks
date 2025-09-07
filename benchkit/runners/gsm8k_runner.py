import argparse
import re
import yaml
from pathlib import Path
from typing import Dict, List

from benchkit.datasets import load_gsm8k
from benchkit.prompts import SYSTEM, cot, sc_base, tot_root, tot_refine
from benchkit.utils import save_jsonl, now_ts
from benchkit.engines.base import Engine
from benchkit.engines.openai_engine import OpenAIEngine, OpenAIConfig
from benchkit.engines.ollama_engine import OllamaEngine, OllamaConfig
from benchkit.engines.hf_engine import HFEngine, HFConfig
from benchkit.metrics import mc_accuracy

def load_engine(name:str, cfg_path:str) -> Engine:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if name == "openai":
        return OpenAIEngine(OpenAIConfig(**cfg))
    elif name == "ollama":
        return OllamaEngine(OllamaConfig(**cfg))
    elif name == "hf":
        return HFEngine(HFConfig(**cfg))
    raise ValueError(f"Unknown engine: {name}")

def extract_final(text:str) -> str:
    m = re.search(r"FINAL:\s*(.*)", text.strip(),flags=re.IGNORECASE)
    return m.group(1).strip() if m else text.strip().splitlines()[-1].strip()[-1]

def run_cot(engine:Engine, q:str) -> str:
    return extract_final(engine.complete(cot(q), system=SYSTEM))

def run_sc(engine:Engine, q:str, k:int = 5) -> str:
    final: List[str] = []
    for _ in range(k):
        final.append(run_cot(engine, q))
    from collections import Counter
    return Counter(final).most_common(1)[0][0]

def parse_branches(text: str, max_branches: int) -> List[str]:
    lines = [l.strip("-• \t") for l in text.splitlines() if l.strip()]
    lines = [l for l in lines if 3 <= len(l.split()) <= 40]
    return lines[:max_branches] or (lines[:1] if lines else [text.strip()[:140]])

def run_tot(engine: Engine, q: str, partial: str, breadth: int=3, depth: int = 2) -> str:
    frontier = [""]
    for d in range(depth):
        new_frontier: List[str] = []
        for partial in frontier:
            prompt = tot_root(q) if d == 0 else tot_refine(q, partial)
            raw = engine.complete(prompt,system=SYSTEM)
            branches = parse_branches(raw, max_branches=breadth)
            for b in branches:
                new_frontier.append((partial + ("\n" if partial else "") + b).strip())
            new_frontier.sort(key=len)
            frontier = new_frontier[:breadth]
    return run_cot(engine, q)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["openai","hf","ollama"])
    ap.add_argument("--engine-config", required=True, help="YAML config for engine")
    ap.add_argument("--prompt-style",default="cot",choices=["cot","sc","tot"])
    ap.add_argument("--k",type=int,default=5,help="Number of times to run")
    ap.add_argument("--breadth",type=int,default=3,help="Number of branches to consider")
    ap.add_argument("--depth",type=int,default=2,help="Depth of tree to consider")
    ap.add_argument("--split",default="test")
    ap.add_argument("--max-samples",type=int,default=50)
    ap.add_argument("--out",default=None)
    args = ap.parse_args()

    engine = load_engine(args.engine, args.engine_config)
    examples: List[Dict] =  load_gsm8k(split=args.split, max_samples=args.max_samples)

    rows = []
    for i, r in enumerate(examples):
        q, gold = r["q"], r["a"]
        if args.prompt_style == "cot":
            final = run_cot(engine, q)
        elif args.prompt_style == "sc":
            final = run_sc(engine, q, k=args.k)
        else:
            final = run_tot(engine, q, partial="", breadth=args.breadth, depth=args.depth)
        rows.append({"idx":i, "question":q, "gold":gold, "final":final})
    out = args.Zout or f"results/gs8k_{args.engine}_{args.prompt_style}_{now_ts()}.jsonl"
    save_jsonl(out, rows)
    print(f"Saved to {len(rows)} rows to {out}")

    acc = mc_accuracy(rows)
    summary = {"_summary": True, "metric": "mc", "value": float(acc), "n": len(rows)}
    with Path(out).open("a", encoding="utf-8") as f:
        import json
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print(f"[metrics] mc_accuracy={acc:.4f} n={len(rows)} → appended to {out}")

if __name__ == "__main__":
    main()