

import re
from typing import Dict, List, Tuple

##############################
# Normalization helpers
##############################

def _normalize_num(s: str) -> Tuple[bool, str]:
    if s is None:
        return False, ""
    m = re.findall(r"[-+]?\d*\.?\d+", str(s))
    if m:
        return True, m[-1]
    return False, str(s).strip()

_ARTICLES = {"a", "an", "the"}

def _normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"[^0-9a-z\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    tokens = [t for t in s.split() if t not in _ARTICLES]
    return " ".join(tokens)

##############################
# Core metrics
##############################

def accuracy_numeric(rows: List[Dict]) -> float:
    correct = 0
    total = 0
    for r in rows:
        gold = r.get("gold", "")
        pred = r.get("final", r.get("winner", ""))
        g_has, g_val = _normalize_num(gold)
        p_has, p_val = _normalize_num(pred)
        if g_has and p_has:
            ok = (g_val == p_val)
        else:
            ok = exact_match_single(pred, gold)
        correct += 1 if ok else 0
        total += 1
    return correct / max(total, 1)

def exact_match_single(pred: str, gold: str) -> bool:
    return _normalize_text(pred) == _normalize_text(gold)

def exact_match(rows: List[Dict]) -> float:
    correct = 0
    total = 0
    for r in rows:
        gold = r.get("gold", "")
        pred = r.get("final", r.get("winner", ""))
        if exact_match_single(pred, gold):
            correct += 1
        total += 1
    return correct / max(total, 1)

def f1_token(pred: str, gold: str) -> float:
    p_toks = _normalize_text(pred).split()
    g_toks = _normalize_text(gold).split()
    if not p_toks and not g_toks:
        return 1.0
    if not p_toks or not g_toks:
        return 0.0
    from collections import Counter
    p_cnt, g_cnt = Counter(p_toks), Counter(g_toks)
    overlap = sum((p_cnt & g_cnt).values())
    if overlap == 0:
        return 0.0
    prec = overlap / max(1, sum(p_cnt.values()))
    rec = overlap / max(1, sum(g_cnt.values()))
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

def f1_average(rows: List[Dict]) -> float:
    total = 0.0
    n = 0
    for r in rows:
        gold = r.get("gold", "")
        pred = r.get("final", r.get("winner", ""))
        total += f1_token(pred, gold)
        n += 1
    return total / max(n, 1)

##############################
# Multiple-choice accuracy
##############################
_MC_CHOICE_RE = re.compile(r"\b([A-E])\b", re.IGNORECASE)

def _extract_mc_label(text: str) -> str:
    if not text:
        return ""
    m = _MC_CHOICE_RE.search(text)
    return m.group(1).upper() if m else ""

def mc_accuracy(rows: List[Dict]) -> float:
    correct = 0
    total = 0
    for r in rows:
        gold = str(r.get("gold", "")).strip().upper()
        pred = r.get("final", r.get("winner", ""))
        guess = _extract_mc_label(str(pred))
        ok = (guess == gold) if gold else False
        correct += 1 if ok else 0
        total += 1
    return correct / max(total, 1)

##############################
# Annotation helper
##############################

def annotate_rows(rows: List[Dict], metric: str) -> List[Dict]:
    out: List[Dict] = []
    for r in rows:
        item = dict(r)
        gold = r.get("gold", "")
        pred = r.get("final", r.get("winner", ""))
        correct = False
        score = None
        if metric == "accuracy":
            g_has, g_val = _normalize_num(gold)
            p_has, p_val = _normalize_num(pred)
            correct = (g_has and p_has and g_val == p_val) or (not g_has and not p_has and exact_match_single(pred, gold))
        elif metric == "em":
            correct = exact_match_single(pred, gold)
        elif metric == "f1":
            score = f1_token(pred, gold)
            correct = (score == 1.0)
        elif metric == "mc":
            correct = (_extract_mc_label(pred) == str(gold).strip().upper())
        item["metric_name"] = metric
        item["metric_correct"] = bool(correct)
        if score is not None:
            item["metric_score"] = float(score)
        out.append(item)
    return out