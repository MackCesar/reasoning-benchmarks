from typing import List, Dict, Any
from datasets import load_dataset


def load_gsm8k(split: str = "test", max_samples: int | None = None) -> List[Dict[str, Any]]:
    ds = load_dataset("openai/gsm8k", "main")[split]
    rows = ds.select(range(min(len(ds), max_samples))) if max_samples else ds
    out = []
    import re
    for r in rows:
        m = re.findall(r"[-+]?\d*\.?\d+", r["answer"])
        gold = m[-1] if m else r["answer"].strip()
        out.append({"q": r["question"], "a": gold})
    return out


def load_arc(split: str = "test", max_samples: int | None = None) -> List[Dict[str, Any]]:
    ds = load_dataset("ai2_arc", "ARC-Challenge")[split]
    rows = ds.select(range(min(len(ds), max_samples))) if max_samples else ds
    out = []
    for r in rows:
        ch = r.get("choices", [])
        choices: List[str] = []
        if isinstance(ch, dict):
            texts, labels = ch.get("text", []), ch.get("label", [])
            if labels and len(labels) == len(texts):
                choices = [f"{lab}) {txt}" for lab, txt in zip(labels, texts)]
            else:
                choices = list(texts)
        elif isinstance(ch, list):
            for c in ch:
                lab, txt = c.get("label", ""), c.get("text", "")
                choices.append(f"{lab}) {txt}" if lab else txt)
        else:
            choices = [str(ch)]
        out.append({
            "q": r["question"] + "\nChoices: " + " | ".join(choices),
            "a": r.get("answerKey", "")
        })
    return out


def load_mmlu(subjects: List[str], split: str = "test", max_samples: int | None = None) -> List[Dict[str, Any]]:
    from itertools import chain

    out = []
    for subj in subjects:
        ds = load_dataset("cais/mmlu", subj)[split]
        rows = ds.select(range(min(len(ds), max_samples))) if max_samples else ds
        for r in rows:
            choices = r.get("choices", [])
            q = r["question"] + "\nChoices: " + " | ".join(choices)
            a = r.get("answer", "")
            out.append({"q": q, "a": a, "subject": subj})
    return out
