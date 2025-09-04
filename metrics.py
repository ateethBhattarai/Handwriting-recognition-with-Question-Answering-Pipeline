# metrics.py
import time
import csv
import os
from contextlib import contextmanager
from difflib import SequenceMatcher

# Optional: faster edit distance if installed
try:
    import Levenshtein
    _HAS_LEV = True
except Exception:
    _HAS_LEV = False

METRICS_CSV = os.environ.get("METRICS_CSV", "./metrics_log.csv")

@contextmanager
def timer(stage: str, extra: dict | None = None):
    """Context manager to time code blocks in milliseconds."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dur_ms = (time.perf_counter() - t0) * 1000.0
        log_metric("latency_ms", round(dur_ms, 3), {"stage": stage, **(extra or {})})

def log_metric(metric: str, value, extra: dict | None = None):
    """Append a metric row to CSV (ts, metric, value, stage, doc_id, user_q, notes)."""
    headers = ["ts", "metric", "value", "stage", "doc_id", "user_q", "notes"]
    row = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metric": metric,
        "value": value,
        "stage": (extra or {}).get("stage", ""),
        "doc_id": (extra or {}).get("doc_id", ""),
        "user_q": (extra or {}).get("user_q", ""),
        "notes": (extra or {}).get("notes", ""),
    }
    exists = os.path.exists(METRICS_CSV)
    with open(METRICS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        if not exists:
            w.writeheader()
        w.writerow(row)

# --- Optional accuracy utilities ---

def _lev(a: str, b: str) -> int:
    if _HAS_LEV:
        return Levenshtein.distance(a, b)
    # Fallback via similarity ratio (approximation of edit distance)
    return int(round((1 - SequenceMatcher(None, a, b).ratio()) * max(len(a), len(b))))

def cer(pred: str, truth: str) -> float:
    """Character Error Rate."""
    if not truth:
        return 0.0
    return _lev(pred, truth) / len(truth)

def wer(pred: str, truth: str) -> float:
    """Word Error Rate (rough)."""
    pw, tw = pred.split(), truth.split()
    if not tw:
        return 0.0
    if _HAS_LEV:
        return Levenshtein.distance(" ".join(pw), " ".join(tw)) / len(tw)
    return int(round((1 - SequenceMatcher(None, pw, tw).ratio()) * max(len(pw), len(tw)))) / max(1, len(tw))

# Retrieval-quality helpers (use offline with a small labeled set)
def precision_at_k(relevant_ids: set[str], retrieved_ids: list[str], k: int) -> float:
    if k <= 0: return 0.0
    topk = retrieved_ids[:k]
    hits = sum(1 for _id in topk if _id in relevant_ids)
    return hits / k

def recall_at_k(relevant_ids: set[str], retrieved_ids: list[str], k: int) -> float:
    if not relevant_ids: return 0.0
    return len(set(retrieved_ids[:k]) & relevant_ids) / len(relevant_ids)

def mrr(relevant_ids: set[str], retrieved_ids: list[str]) -> float:
    for rank, _id in enumerate(retrieved_ids, start=1):
        if _id in relevant_ids:
            return 1.0 / rank
    return 0.0