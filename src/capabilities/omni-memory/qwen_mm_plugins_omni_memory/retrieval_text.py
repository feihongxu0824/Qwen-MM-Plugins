"""Portable tokenization shared by the memory builder and query server.

Keep the server and standalone skill copies byte-identical. Persisted indexes use these
tokens with a whitespace tokenizer, so a source query has the same term boundaries.
"""

import re

BM25_K1 = 1.5
BM25_B = 0.75
BM25_BOOST = 2.0
TOKENIZER_VERSION = 1

_STOP = frozenset(
    "the a an is are was were of to in on at and or do does did you i he she it they we "
    "this that what who which how when where why please hmm um uh 的 了 是 在 吗 呢 我 你 他 她 它".split()
)


def toks(s):
    return set(re.findall(r"[a-z0-9']+|[一-鿿]", str(s or "").lower()))


def kw(s):
    return toks(s) - _STOP


def tok_list(s):
    """Keep repeated terms for BM25 term frequency."""
    return [t for t in re.findall(r"[a-z0-9']+|[一-鿿]", str(s or "").lower()) if t not in _STOP]
