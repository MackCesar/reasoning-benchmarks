import argparse
import re
import yaml
from pathlib import Path
from typing import List, Dict

from benchkit.datasets import load_mmlu
from benchkit.prompts import SYSTEM, cot, sc_base, tot_root, tot_refine
from benchkit.utils import save_jsonl, now_ts
from benchkit.engines.base import Engine
from benchkit.engines.openai_engine import OpenAIEngine, OpenAIConfig
from benchkit.engines.hf_engine import HFEngine, HFConfig
from benchkit.engines.ollama_engine import OllamaEngine, OllamaConfig


def load_engine(name: str, cfg_path: str) -> Engine:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if name == "openai":
        return OpenAIEngine(OpenAIConfig(**cfg))
    if name == "hf":
        return HFEngine(HFConfig(**cfg))
    if name == "ollama":
        return OllamaEngine(OllamaConfig(**cfg))
    raise ValueError(f"Unknown engine: {name}")


def extract_final(text: str) -> str:
    m = re.search(r"FINAL:\s*(.*)", text.strip(), flags=re.IGNORECASE)
    return m.group(1).strip() if m else text.strip().splitlines()[-1]


def run_cot(engine: Engine, q: str) -> str:
    return extract_final(engine.complete(cot(q), system=SYSTEM))


def run_sc(engine: Engine, q: str, k: int = 5) -> str:
    finals: List[str] = []
    for _ in range(k):
        finals.append(run_cot(engine, q))
    from collections import Counter
    return Counter(finals).most_common(1)[0][0]


def parse_branches(text: str, max_branches: int) -> List[str]:
    lines = [l.strip("-• \t") for l in text.splitlines() if l.strip()]
    lines = [l for l in lines if 3 <= len(l.split()) <= 40]
    return lines[:max_branches] or (lines[:1] if lines else [text.strip()[:140]])


def run_tot(engine: Engine, q: str, breadth: int = 3, depth: int = 2) -> str:
    frontier = [""]
    for d in range(depth):
        new_frontier: List[str] = []
        for partial in frontier:
            prompt = tot_root(q) if d == 0 else tot_refine(q, partial)
            raw = engine.complete(prompt, system=SYSTEM)
            branches = parse_branches(raw, max_branches=breadth)
            for b in branches:
                new_frontier.append((partial + ("\n" if partial else "") + b).strip())
        new_frontier.sort(key=len)
        frontier = new_frontier[:breadth]
    return run_cot(engine, q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["openai", "hf", "ollama"])
    ap.add_argument("--engine-config", required=True)
    ap.add_argument("--prompt-style", default="cot", choices=["cot", "sc", "tot"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--breadth", type=int, default=3)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--subjects", required=True, help="comma-separated subject list, e.g. math,physics")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-samples", type=int, default=50)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    engine = load_engine(args.engine, args.engine_config)
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    examples: List[Dict] = load_mmlu(subjects, split=args.split, max_samples=args.max_samples)

    rows = []
    for i, r in enumerate(examples):
        q, gold = r["q"], r["a"]
        if args.prompt_style == "cot":
            final = run_cot(engine, q)
        elif args.prompt_style == "sc":
            final = run_sc(engine, q, k=args.k)
        else:
            final = run_tot(engine, q, breadth=args.breadth, depth=args.depth)
        row = {"idx": i, "question": q, "gold": gold, "final": final}
        if "subject" in r:
            row["subject"] = r["subject"]
        rows.append(row)

    out = args.out or f"results/mmlu_{args.engine}_{args.prompt_style}_{now_ts()}.jsonl"
    save_jsonl(out, rows)
    print(f"Saved {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()