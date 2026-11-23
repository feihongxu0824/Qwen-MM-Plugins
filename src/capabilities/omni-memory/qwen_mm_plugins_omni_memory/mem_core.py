"""Reading a memory: the store plus hybrid retrieval over it.

MemoryStore is StoreBase (the containers, from omni_core) plus recall — dense embeddings and BM25,
RRF-fused, with entity-anchored multi-hop and phonetic name matching.

This is the READ half and it is all the MCP server has; building lives in skill/script/build_memory/
as a standalone program. Answering off the video itself is watch.py; the matching primitives are
text_match.py.
"""

import time

from . import config
from .memory_backend import JsonBackend
from .memory_query import Predicate, Selection, select
from .omni_core import (
    NAME_STOP,
    RESIDENT_ENTITY_CAP,
    SALIENT_EMO,
    SALIENT_TONE,
    StoreBase,
    diag,
    embed_texts,
    get_embed_client,
    normalize_acoustic_events,
)
from .text_match import (
    BM25_B,
    BM25_BOOST,
    BM25_K1,
    kw,
    phonetic_name_match,
    query_person_names,
    rrf,
)
from .watch import REPLAY_N

RESIDENT_KEY_CAP = 80  # max semantic keys kept resident

SCENE_RECALL_N = 4  # how many scene items to recall when the plan asks for SCENE_ENV

EP_TOPN = 6


class MemoryStore(StoreBase):
    """StoreBase plus hybrid retrieval over it — what every query tool goes through."""

    def semantic_key_directory(self, limit=None):
        """Resident directory of semantic KEYS only. Under the buffer cap keep ALL (insertion
        order); over the cap keep the most frequently/recently used (freq×recency)."""
        cap = RESIDENT_KEY_CAP if limit is None else limit
        active = self.records("facts", Selection().where(Predicate("status", "ne", "superseded")))
        if len(active) > cap:
            active = sorted(active, key=lambda t: (t.get("freq", 0), t.get("last_used", 0)), reverse=True)[:cap]
        return [t.get("key") for t in active]

    # ---------- resident (the cast of people + the directory of keys) ----------
    def _ledger_candidates(self):
        """person_id → the names it was most often heard called, for people with no resolved name.

        The ledger key splits into `key_parts`; `parts` is resident_context's output list and must not
        be rebound here.
        """
        cands: dict[str, list] = {}
        if not (config.get_bool_env("MEM_ANON_ENTITIES", True) and self.name_ledger):
            return cands
        for key, val in self.name_ledger.items():
            key_parts = key.split("|")
            if len(key_parts) != 4:
                continue
            nm, sid = key_parts[0], key_parts[1]
            if sid.startswith("P") and val.get("count", 0) >= 2:
                cands.setdefault(sid, []).append((nm, val["count"]))
        for sid in cands:
            cands[sid] = sorted(cands[sid], key=lambda x: -x[1])[:3]
        return cands

    def entity_directory(self, appearance_chars=120):
        """The resident cast of people, as data.

        Same content resident_context() renders as text, so a caller planning its own retrieval sees
        exactly what the pipeline's planner used to see — including `also_heard_as`, which for a
        person with no resolved name is the ONLY way to map a name someone says onto a person_id.
        """
        ents = self.entities or []
        if len(ents) > RESIDENT_ENTITY_CAP:
            ents = sorted(ents, key=lambda e: (e.get("freq", 0), e.get("last_used", 0)), reverse=True)[
                :RESIDENT_ENTITY_CAP
            ]
        cands = self._ledger_candidates()
        out = []
        for e in ents:
            pid = e.get("person_id")
            rec = {"person_id": pid, "name": e.get("name")}
            if e.get("appearance"):
                rec["appearance"] = (e.get("appearance") or "")[:appearance_chars]
            if not e.get("name") and cands.get(pid):
                rec["also_heard_as"] = [nm for nm, _ in cands[pid]]
            out.append(rec)
        return out

    # ---------- structured retrievers (deterministic, no model calls) ----------
    def _augment_phonetic(self, ents, sem, query):
        """③ Resolve person names in the question to entities, by exact name or by SOUND.

        Exact match first, then soundex / edit-similarity against entity names and against ledger
        candidate names — the latter covers anonymous entities whose real name was only ever heard,
        and mis-heard names ("Dara" for "Lara"). Each resolved person brings its triples along.
        """
        have = {e.get("person_id") for e in ents}
        seen_keys = {t.get("key") for t in sem}

        def _adopt(e, why):
            pid = e.get("person_id")
            if not pid or pid in have:
                return False
            ents.append(e)
            have.add(pid)
            for t in self.triples_of(pid):
                if t.get("key") not in seen_keys and t.get("status") != "superseded":
                    sem.append(t)
                    seen_keys.add(t.get("key"))
            diag(f"[MEM] > NAME-RESOLVED {why} ({pid})", flush=True)
            return True

        for qn in query_person_names(query):
            known = self.get_entity(qn)
            if known:
                # Naming a KNOWN person in the question used to resolve to nothing here, on the
                # assumption that whoever wrote the plan had already listed its person_id. Callers
                # now often pass just the question, so an exact name has to resolve too.
                _adopt(known, f"query '{qn}' = stored name")
                continue
            matched = False
            # Then: entity names that only SOUND like the queried one
            for e in self.entities:
                pid, nm = e.get("person_id"), e.get("name")
                if not nm or pid in have or nm.lower() in NAME_STOP:
                    continue
                if phonetic_name_match(qn, nm):
                    matched = _adopt(e, f"query '{qn}' ~ stored '{nm}'")
                    break
            if matched:
                continue
            # Last: ledger candidate names, which is where an anonymous entity's real name lives
            for key in self.name_ledger:
                parts = key.split("|")
                if len(parts) != 4:
                    continue
                ledger_nm, sid = parts[0], parts[1]
                if not sid.startswith("P") or sid in have or ledger_nm.lower() in NAME_STOP:
                    continue
                if self.name_ledger[key].get("count", 0) < 2:
                    continue
                if phonetic_name_match(qn, ledger_nm):
                    e = self.get_entity(sid)
                    if e and _adopt(e, f"query '{qn}' ~ ledger '{ledger_nm}'"):
                        matched = True
                        break
        return ents, sem

    def get_entity(self, name_or_id):
        return self._by_id.get(name_or_id) or self._by_name.get(str(name_or_id).lower())

    @property
    def backend(self):
        if not hasattr(self, "_backend"):
            self._backend = JsonBackend({"episodic": self.episodic, "semantic": self.semantic})
        return self._backend

    def records(self, pool, selection=Selection()):
        source = self.episodic if pool == "episodes" else self.semantic
        return [source[f["ordinal"]] for f in self.backend.scan(pool, selection)]

    def restrict(self, pool, records, selection):
        if not selection.clauses:
            return records
        allowed = {id(r) for r in self.records(pool, selection)}
        return [r for r in records if id(r) in allowed]

    def triples_of(self, subject_id):
        return self.records(
            "facts",
            Selection().where(
                Predicate("subject_id", "eq", subject_id),
                Predicate("subject", "eq", str(subject_id).lower()),
            ),
        )

    def facts_by_prefix(self, prefixes):
        return self.records("facts", Selection().where(*(Predicate("key", "prefix", p) for p in prefixes)))

    def get_triples_by_keys(self, keys):
        return self.records(
            "facts",
            Selection().where(Predicate("key", "in", tuple(keys or []))).where(Predicate("status", "ne", "superseded")),
        )

    def recall_scene_env(self, query, k=SCENE_RECALL_N):
        """① on-demand scene recall: return the top-k scene_env items most relevant to the query
        (keyword overlap; fallback to first-k). Only called when PLAN picks the SCENE_ENV key."""
        if not self.scene_env:
            return []
        q = kw(query or "")
        ranked = sorted(self.scene_env, key=lambda s: -len(q & kw(s)))
        hit = [s for s in ranked if q & kw(s)]
        return (hit or list(self.scene_env))[:k]

    def episodic_in_time(self, start, end):
        return self.records("episodes", select({"start_sec": start, "end_sec": end}))

    def episodic_of_entity(self, person_id):
        return self.records("episodes", select({"person_ids": [person_id]}))

    def dialogue_rows(self, filters=None):
        return self.backend.scan("utterances", select(filters, "utterances"))

    def _rank(self, pool, query, qvec, k, selection=Selection()):
        limit = max(500, 5 * k)
        lists = [
            self.backend.sparse(
                pool,
                query,
                selection,
                k1=config.get_float_env("MEM_BM25_K1", BM25_K1, min_value=0),
                b=config.get_float_env("MEM_BM25_B", BM25_B, min_value=0, max_value=1),
                boost=config.get_float_env("MEM_BM25_BOOST", BM25_BOOST, min_value=0),
                limit=limit,
            )
        ]
        if qvec is not None:
            lists.append(self.backend.dense(pool, qvec, selection, limit=limit))
        scores = rrf(lists)
        order = sorted(scores, key=lambda i: (-scores[i], i))[:k]
        source = self.episodic if pool == "episodes" else self.semantic
        return [source[i] for i in order]

    def _query_vector(self, query, client):
        if not self.dense_ok:
            return None
        client = client or get_embed_client()
        if client is not None and self.embedding_model and self.embedding_model != config.embed_config()[1]:
            raise ValueError("query embedding model differs from this memory; use its model or rebuild")
        v = embed_texts(client, [query])
        return v[0] if v else None

    @staticmethod
    def _filters(filters):
        if hasattr(filters, "model_dump"):
            filters = filters.model_dump(exclude_none=True)
        filters = filters or {}
        if filters.keys() - {"episodes", "facts"}:
            raise ValueError("filters must contain episodes and/or facts")
        return select(filters.get("episodes")), select(filters.get("facts"), "facts")

    def _restrict_people(self, people, episode_selection, fact_selection):
        if episode_selection.clauses:
            allowed = set()
            for row in self.backend.scan("episodes", episode_selection):
                allowed.update(row["person_ids"])
            people = [e for e in people if e.get("person_id") in allowed]
        elif fact_selection.clauses:
            allowed = {r["subject_id"] for r in self.backend.scan("facts", fact_selection)}
            people = [e for e in people if e.get("person_id") in allowed]
        return people

    def search(self, query, client=None, k=5, filters=None):
        ep_sel, fact_sel = self._filters(filters)
        qvec = self._query_vector(query, client)
        ep = self._rank("episodes", query, qvec, k, ep_sel)
        sem = self._rank("facts", query, qvec, k, fact_sel)
        qk = kw(query)
        matched = [e for e in self.entities if qk & (kw(e.get("name")) | kw(e.get("appearance")) | kw(e.get("ref")))]
        matched = self._restrict_people(matched, ep_sel, fact_sel)
        for entity in matched:
            for fact in self.restrict("facts", self.triples_of(entity.get("person_id")), fact_sel):
                if not any(fact is r for r in sem):
                    sem.append(fact)
            related = self.restrict("episodes", self.episodic_of_entity(entity.get("person_id")), ep_sel)
            for episode in related[:k]:
                if not any(episode is r for r in ep):
                    ep.append(episode)
        if not ep and not sem and not matched:
            ep = self.records("episodes", ep_sel)[:k]
            sem = self.records("facts", fact_sel)[:k]
        return {
            "entities": matched,
            "semantic": sem,
            "episodic": ep,
            "mode": "hybrid" if qvec is not None else "keyword",
        }

    def search_episodic(self, query, client=None, k=5, filters=None, selection=None):
        selected = selection if selection is not None else select(filters)
        return self._rank("episodes", query, self._query_vector(query, client), k, selected)

    def search_semantic(self, query, client=None, k=5, filters=None):
        selected = select(filters, "facts").where(Predicate("status", "ne", "superseded"))
        return self._rank("facts", query, self._query_vector(query, client), k, selected)

    def search_dialogue(self, query, client=None, k=8, filters=None):
        rows = self.dialogue_rows(filters)
        selection = Selection().where(Predicate("idx", "in", tuple(sorted({f["clip_idx"] for f in rows}))))
        episodes = (
            self.search_episodic(query, client, k, selection=selection)
            if filters
            else self.search_episodic(query, client, k)
        )
        chosen = {e.get("idx") for e in episodes}
        return episodes, [row for row in rows if row["clip_idx"] in chosen]

    def rerank_episodic(self, cands, query, k=None, client=None):
        k = k or config.get_int_env("MEM_EP_TOPN", EP_TOPN, min_value=1)
        if len(cands) <= k:
            return cands
        selection = Selection().where(Predicate("idx", "in", tuple(e["idx"] for e in cands)))
        ranked = self.search_episodic(query, client, k, selection=selection)
        return ranked or cands[:k]

    def run_plan(self, plan, query, client=None, k=5, filters=None):
        ep_sel, fact_sel = self._filters(filters)
        keys = list(plan.get("need_keys") or [])
        want_scene = "SCENE_ENV" in keys or bool(plan.get("need_scene"))
        sem = self.get_triples_by_keys([x for x in keys if x != "SCENE_ENV"])
        ents = [e for e in (self.get_entity(x) for x in (plan.get("need_entities") or [])) if e]
        if config.get_bool_env("MEM_PHONETIC_FALLBACK", True):
            ents, sem = self._augment_phonetic(ents, sem, query)
        ents = self._restrict_people(ents, ep_sel, fact_sel)
        sem = self.restrict("facts", sem, fact_sel)
        for item in ents + sem:
            item["freq"] = item.get("freq", 0) + 1
            item["last_used"] = time.time()
        queries = list(plan.get("episode_queries") or [])
        if query not in queries:
            queries.append(query)
        scene = self.recall_scene_env(" ".join(queries) or query) if want_scene else []
        ep, seen = [], set()
        for q in queries:
            for item in self.search_episodic(q, client, k, selection=ep_sel):
                if item["idx"] not in seen:
                    seen.add(item["idx"])
                    ep.append(item)
        for tr in plan.get("time_ranges") or []:
            try:
                start, end = float(tr[0]), float(tr[1])
            except (ValueError, TypeError, IndexError):
                continue
            for item in self.restrict("episodes", self.episodic_in_time(start, end), ep_sel):
                if item["idx"] not in seen:
                    seen.add(item["idx"])
                    ep.append(item)
        if config.get_bool_env("MEM_EP_RERANK"):
            ep = self.rerank_episodic(ep, query, client=client)
        replays = self._resolve_replays(plan, list(ep), sem)
        # Replay fallback must not escape a hard episode restriction.
        if ep_sel.clauses:
            eligible = {r["idx"] for r in self.backend.scan("episodes", ep_sel)}
            replays = [c for c in replays if c.get("idx") in eligible]
        ep.sort(key=lambda item: item.get("win_start", 0))
        return {
            "entities": ents,
            "semantic": sem,
            "episodic": ep,
            "scene": scene,
            "replay": replays[0] if replays else None,
            "replays": replays,
            "mode": "plan",
        }

    def _resolve_replay(self, plan, ep):
        """Pick EXACTLY ONE clip to re-watch (strict). Returns a clip dict or None."""
        tgt = plan.get("replay_target")
        tgt = tgt if isinstance(tgt, dict) else {}
        by, val = tgt.get("by"), tgt.get("value")
        idx = None
        if by == "clip_idx":
            try:
                idx = int(val)
            except Exception:
                idx = None
        elif by == "time":
            try:
                t = float(val[0]) if isinstance(val, (list, tuple)) else float(val)
            except Exception:
                t = None
            if t is not None:
                for c in self.clips:
                    if c.get("win_start", 0) <= t <= c.get("win_end", 0):
                        idx = c.get("idx")
                        break
        if idx is None and plan.get("time_ranges"):
            try:
                s = float(plan["time_ranges"][0][0])
                for c in self.clips:
                    if c.get("win_start", 0) <= s <= c.get("win_end", 1e18):
                        idx = c.get("idx")
                        break
            except Exception:
                pass
        if idx is None and ep:
            idx = ep[0].get("idx")
        if idx is None and self.clips:
            idx = self.clips[0].get("idx")
        for c in self.clips:
            if c.get("idx") == idx:
                return c
        return None

    def _clip_playable(self, c):
        return bool(c and (c.get("oss_key") or c.get("path")))

    def _resolve_replays(self, plan, ep, sem_hits=None, n=None):
        """Pick which clips to re-watch: dense episodic ranking first, time hints only as tiebreaker.

        Ordering by time or semantic anchors ahead of the dense ranking was measured to surface worse
        clips, so it is deliberately not done.
        """
        if n is None:
            n = config.get_int_env("MEM_REPLAY_N", REPLAY_N, min_value=1)
        order = []
        for o in ep:  # dense relevance order (primary signal)
            order.append(o.get("idx"))
        primary = self._resolve_replay(plan, ep)  # time-anchor as secondary
        if primary and primary.get("idx") not in set(o.get("idx") for o in ep[:3]):
            order.insert(0, primary.get("idx"))  # only boost if NOT already in top-3
        by_idx = {c.get("idx"): c for c in self.clips}
        _MIN_GAP = 30  # minimum seconds between selected replay clips
        out, seen = [], set()
        selected_times = []  # (win_start, win_end) of selected clips
        for idx in order:
            if idx in seen:
                continue
            seen.add(idx)
            c = by_idx.get(idx)
            if not self._clip_playable(c):
                continue
            # temporal diversity: skip if too close to an already-selected clip
            cs, ce = c.get("win_start", 0), c.get("win_end", 0)
            if n >= 3 and len(out) >= 1:
                too_close = any(abs(cs - ss) < _MIN_GAP and abs(ce - se) < _MIN_GAP for ss, se in selected_times)
                if too_close:
                    continue
            out.append(c)
            selected_times.append((cs, ce))
            if len(out) >= max(1, n):
                break
        return out

    # ---------- evidence assembly (pyramid: Core → Semantic → Episode) ----------
    def name_of(self, pid):
        """L3 query-time mapping: person_id → current canonical name (or None)."""
        e = self._by_id.get(pid)
        return e.get("name") if e else None

    def _ep_evidence_line(self, o):
        p = o.get("parsed") or {}
        vis = p.get("visual", {}) or {}
        aud = p.get("audio", {}) or {}
        head = f"- [{o.get('win_start')}-{o.get('win_end')}s] {vis.get('visual_caption', '')}"
        lines = [head]
        for u in aud.get("utterances") or []:
            sid = u.get("speaker_id") or "unknown"
            who = self.name_of(sid) or sid  # L3: show live name, not frozen speaker_id
            pl = u.get("paralinguistic") or {}
            tags = [pl.get("emotion")] if (pl.get("emotion") or "") not in SALIENT_EMO else []
            if (pl.get("tone") or "") not in SALIENT_TONE:
                tags.append(pl["tone"])
            tg = f"[{','.join(t for t in tags if t)}]" if tags else ""
            ts = f"[{u.get('start_sec')}-{u.get('end_sec')}s]" if u.get("start_sec") is not None else ""
            lines.append(f"    🗣{ts}[{who}]{tg} {u.get('text', '')}")
        ac = aud.get("acoustic", {}) or {}
        if ac.get("events"):
            evts = normalize_acoustic_events(ac["events"])
            parts = []
            for e in evts:
                label = e.get("event", str(e)) if isinstance(e, dict) else str(e)
                ts = ""
                if isinstance(e, dict) and e.get("start_sec") is not None:
                    ts = f"[{e['start_sec']}-{e.get('end_sec', '?')}s]"
                    cont = e.get("continues_from_previous")
                    if cont:
                        ts += "(cont)"
                parts.append(f"{ts}{label}" if ts else label)
            lines.append(f"    🔊 {'; '.join(parts)}")
        ac_scene = ac.get("scene", "")
        if ac_scene and ac_scene not in ("unknown", ""):
            lines.append(f"    🔊env: {ac_scene}")
        return "\n".join(lines)

    def evidence_text(self, hits):
        out = []
        if hits.get("entities"):
            out.append("## Entities (Core)")
            for e in hits["entities"]:
                nm = e.get("name") or e.get("ref") or e.get("person_id")
                out.append(
                    f"- {e.get('person_id')} = {nm}: {e.get('appearance', '')}"
                    + (f"; last: {e.get('last_action', '')}" if e.get("last_action") else "")
                )
        if hits.get("scene"):  # ① on-demand environment (recalled only when PLAN picked SCENE_ENV)
            out.append("## Environment / layout (Scene)")
            for s in hits["scene"]:
                out.append(f"- {s}")
        if hits.get("semantic"):
            out.append("## Stable facts (Semantic)")
            for f in hits["semantic"]:
                out.append(f"- {f.get('statement', '')}  (conf={f.get('confidence', '')})")
        if hits.get("episodic"):
            out.append("## Relevant moments (Episode)")
            for o in hits["episodic"]:
                out.append(self._ep_evidence_line(o))
        return "\n".join(out) if out else "(no relevant memory found)"
