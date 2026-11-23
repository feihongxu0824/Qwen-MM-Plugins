"""How text is matched and how rankings are fused — no memory, no model, no I/O.

Tokenising (CJK-aware), BM25 tuning constants, cosine similarity, reciprocal-rank fusion, and the
phonetic name matching that lets a mis-heard name still find its person.
"""

import difflib
import re

import numpy as np

from .omni_core import NAME_STOP
from .retrieval_text import BM25_B, BM25_BOOST, BM25_K1, kw, tok_list, toks  # noqa: F401


def soundex(s):
    """Cheap phonetic code — matches a mis-heard STORED name to the query name (Lewis↔Louise)."""
    s = "".join(c for c in (s or "").upper() if c.isalpha())
    if not s:
        return ""
    codes = {
        **dict.fromkeys("BFPV", "1"),
        **dict.fromkeys("CGJKQSXZ", "2"),
        **dict.fromkeys("DT", "3"),
        "L": "4",
        **dict.fromkeys("MN", "5"),
        "R": "6",
    }
    out = s[0]
    prev = codes.get(s[0], "")
    for ch in s[1:]:
        cc = codes.get(ch, "")
        if cc and cc != prev:
            out += cc
        if ch not in "HW":
            prev = cc
    return (out + "000")[:4]


_Q_STOP = {
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "how",
    "is",
    "are",
    "does",
    "do",
    "did",
    "the",
    "based",
    "after",
    "before",
    "during",
    "robot",
    "cola",
    "juice",
    "orange",
    "coffee",
}


def query_person_names(q):
    """Person-name candidates in a question (Capitalized words minus stop / question words)."""
    return [
        t for t in re.findall(r"\b[A-Z][a-z]{2,}\b", q or "") if t.lower() not in NAME_STOP and t.lower() not in _Q_STOP
    ]


def phonetic_name_match(a, b):
    """True if two names sound alike (same soundex or high edit-similarity) — for QA recall of an
    entity whose stored name was mis-transcribed. Threshold 0.72 keeps distinct names apart
    (lily/emma=0, jack/robert=0) while catching mis-hears (lewis/louise, cora/cara, isha/aisha)."""
    a, b = (a or "").lower().strip(), (b or "").lower().strip()
    if not a or not b:
        return False
    if a == b:
        return True
    return soundex(a) == soundex(b) or difflib.SequenceMatcher(None, a, b).ratio() >= 0.72


# ============================================================ RETRIEVAL HELPERS
def cosine(a, b):
    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def rrf(rank_lists, rrf_k=60):
    score = {}
    for lst in rank_lists:
        for rank, iid in enumerate(lst):
            score[iid] = score.get(iid, 0.0) + 1.0 / (rrf_k + rank + 1)
    return score
