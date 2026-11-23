"""One behavioral contract exercised against JSON and the real optional zvec engine."""

import importlib
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from qwen_mm_plugins_omni_memory import mem_core, service
from qwen_mm_plugins_omni_memory.memory_backend import open_backend
from qwen_mm_plugins_omni_memory.memory_io import commit_snapshot, migrate, read_snapshot
from qwen_mm_plugins_omni_memory.memory_query import select, text_hash


def snapshot(count=8):
    episodes, facts = [], []
    for i in range(count):
        pid = "P001" if i % 2 == 0 else "P002"
        name = "Ada" if pid == "P001" else "Bob"
        topic = "camera aperture" if i % 2 == 0 else "bread bakery"
        episodes.append(
            {
                "idx": i,
                "win_start": i * 25,
                "win_end": i * 25 + 30,
                "parsed": {
                    "visual": {
                        "visual_caption": f"{name} explains {topic} item{i}",
                        "key_entities": [{"person_id": pid, "name": name}],
                    },
                    "audio": {
                        "utterances": [
                            {"speaker_id": pid, "start_sec": i * 25 + 2, "text": f"Discuss {topic}"},
                            {
                                "speaker_id": "P003",
                                "start_sec": i * 25 + 20,
                                "text": "Thank you",
                                "paralinguistic": {"emotion": "happy"},
                            },
                        ]
                    },
                },
                "emb": [1.0, 0.0, 0.0] if i % 2 == 0 else [0.0, 1.0, 0.0],
                "state_after": {"last_processed_until": i * 25 + 30},
            }
        )
        facts.append(
            {
                "key": f"{name}/topic/{i}",
                "subject_id": pid,
                "subject": name,
                "statement": f"{name} knows {topic} item{i}",
                "confidence": "high",
                "type": "fact",
                "evidence": [i],
                "emb": episodes[-1]["emb"],
            }
        )
    return {
        "version": 3,
        "episodic": episodes,
        "semantic": facts,
        "entities": [
            {"person_id": "P001", "name": "Ada", "appearance": "red shirt"},
            {"person_id": "P002", "name": "Bob", "appearance": "blue shirt"},
            {"person_id": "P003", "name": "Cleo"},
        ],
        "clips": [
            {"idx": i, "win_start": i * 25, "win_end": i * 25 + 30, "path": "/missing/clip.mp4"} for i in range(count)
        ],
        "scene_env": ["A camera is on the table"],
        "dense_ok": True,
        "processed_sec": count * 25 + 5,
        "name_ledger": {},
        "unknown_extension": {"must": "survive"},
    }


@pytest.fixture(params=["json", "zvec"])
def memory(request, tmp_path, monkeypatch):
    if request.param == "zvec":
        pytest.importorskip("zvec")
    directory = tmp_path / request.param
    commit_snapshot(directory, snapshot(), request.param)
    monkeypatch.setattr(service.config, "local_dir", lambda: str(tmp_path))
    monkeypatch.setattr(service, "embed_client", lambda: None)
    monkeypatch.setattr(mem_core, "get_embed_client", lambda: None)
    monkeypatch.setattr(mem_core, "embed_texts", lambda client, texts: None)
    yield directory
    for _, store in service._stores.values():
        store.backend.close()
    service._stores.clear()


def tool(name, memory, **kwargs):
    module = importlib.import_module(f"qwen_mm_plugins_omni_memory.tools.{name}")
    return json.loads(module.handle({"namespace": memory.name, **kwargs})[0]["text"])


def test_public_structured_tools(memory):
    status = tool("get_memory_status", memory)
    assert status["complete"] and status["episodic"] == 8 and status["people"] == 3
    overview = tool("get_memory_overview", memory)
    assert len(overview["semantic_key_directory"]) == 8
    assert overview["scene_env_available"]
    people = tool("get_people", memory, person_id="P001")["people"]
    assert people[0]["clips_present"] == 4 and people[0]["utterance_count"] == 4
    assert len(people[0]["facts"]) == 4
    timeline = tool("get_timeline", memory, start_sec=29, end_sec=50)
    assert [m["idx"] for m in timeline["moments"]] == [0, 1, 2]
    moment = tool("get_moment", memory, idxs=[0])["moments"][0]
    assert moment["utterances"][0]["speaker_name"] == "Ada"
    assert moment["utterances"][1]["paralinguistic"] == {"emotion": "happy"}
    dialogue = tool("get_person_dialogue", memory, person_id="P003", start_sec=20, end_sec=45, limit=1)
    assert dialogue["total"] == 2 and dialogue["returned"] == 1
    assert dialogue["utterances"][0]["start_sec"] == 20


def test_all_four_search_tools_with_hard_constraints(memory, monkeypatch):
    monkeypatch.setattr(mem_core, "embed_texts", lambda client, texts: [[1.0, 0.0, 0.0] for _ in texts])
    filters = {
        "episodes": {"person_ids": ["P002"], "start_sec": 75, "end_sec": 100},
        "facts": {"subject_ids": ["P002"], "key_prefix": "Bob/topic/3"},
    }
    result = tool("search_memory", memory, query="Ada camera", filters=filters, top_k=1)
    assert [m["idx"] for m in result["moments"]] == [3]
    assert [f["key"] for f in result["facts"]] == ["Bob/topic/3"]
    assert all(p["person_id"] != "P001" for p in result["people"])
    dialogue = tool(
        "search_dialogue", memory, query="camera", filters={"speaker_ids": ["P003"], "start_sec": 20, "end_sec": 20}
    )
    assert len(dialogue["utterances"]) == 1 and dialogue["utterances"][0]["speaker_id"] == "P003"
    facts = tool("search_facts", memory, query="camera", filters={"subject_ids": ["P002"], "key_prefix": "Bob/topic/3"})
    assert [f["key"] for f in facts["facts"]] == ["Bob/topic/3"]
    planned = tool(
        "plan_and_search",
        memory,
        question="Ada camera",
        people=["P001"],
        fact_keys=["Ada/topic/0", "Bob/topic/3"],
        time_ranges=[[0, 200]],
        filters=filters,
    )
    assert [m["idx"] for m in planned["moments"]] == [3]
    assert [f["key"] for f in planned["facts"]] == ["Bob/topic/3"]
    assert not planned["people"]
    assert planned["suggested_replay_idxs"] == [3]


def test_dialogue_predicates_apply_to_same_utterance(memory):
    result = tool(
        "search_dialogue", memory, query="camera", filters={"speaker_ids": ["P001"], "start_sec": 20, "end_sec": 20}
    )
    assert result["matched_clips"] == 0 and result["utterances"] == []


def test_filter_only_facts_and_legacy_argument_precedence(memory):
    filtered = tool("search_facts", memory, filters={"subject_ids": ["P001"], "key_prefix": "Ada/topic/2"})
    assert [f["key"] for f in filtered["facts"]] == ["Ada/topic/2"]
    legacy = tool("search_facts", memory, subject_id="P001", key_prefix="Bob/", query="bread")
    assert legacy["count"] == 4  # old subject > prefix > query precedence remains intact


def test_empty_filter_results_never_expand_outside_constraints(memory):
    result = tool(
        "search_memory", memory, query="Ada", filters={"episodes": {"person_ids": []}, "facts": {"subject_ids": []}}
    )
    assert result["moments"] == result["facts"] == result["people"] == []


def test_native_three_collections_and_no_json_vectors(memory):
    data = read_snapshot(memory)
    if memory.name == "zvec":
        assert {p.name for p in (memory / "zvec").iterdir()} == {"episodes", "facts", "utterances"}
        assert all("emb" not in r for r in data["episodic"] + data["semantic"])
    else:
        assert not (memory / "zvec").exists()
        assert all("emb" in r for r in data["episodic"] + data["semantic"])


def test_missing_zero_and_real_vectors_roundtrip(memory):
    data = read_snapshot(memory, include_vectors=True)
    data["episodic"][0]["emb"] = None
    data["episodic"][1]["emb"] = [0.0, 0.0, 0.0]
    commit_snapshot(memory, data)
    backend = open_backend(memory, read_snapshot(memory))
    try:
        ranked = backend.dense("episodes", [1.0, 0.0, 0.0])
        assert 0 not in ranked and 1 in ranked
        assert len(backend.scan("episodes")) == 8
    finally:
        backend.close()
    migrate(memory, "json")
    loaded = read_snapshot(memory, include_vectors=True)
    assert loaded["episodic"][0].get("emb") is None
    from qwen_mm_plugins_omni_memory.omni_core import emb_decode

    np.testing.assert_array_equal(emb_decode(loaded["episodic"][1]["emb"]), [0.0, 0.0, 0.0])
    assert loaded["unknown_extension"] == {"must": "survive"}


def test_update_delete_and_stale_vector_invalidation(memory):
    data = read_snapshot(memory, include_vectors=True)
    first = data["episodic"][0]
    first["emb_text_hash"] = text_hash("episodes", first)
    first["parsed"]["visual"]["visual_caption"] = "Cleo opens a window"
    first["parsed"]["visual"]["key_entities"] = [{"person_id": "P003", "name": "Cleo"}]
    first["parsed"]["audio"]["utterances"] = []
    data["semantic"] = data["semantic"][1:]
    data["episodic"].pop()
    commit_snapshot(memory, data)
    backend = open_backend(memory, read_snapshot(memory))
    try:
        assert [f["idx"] for f in backend.scan("episodes", select({"person_ids": ["P001"]}))] == [2, 4, 6]
        assert 0 not in backend.dense("episodes", [1.0, 0.0, 0.0])
        assert len(backend.scan("facts")) == 7
        assert len(backend.scan("utterances")) == 12
    finally:
        backend.close()


def test_complete_scans_do_not_truncate_and_order_is_stable(memory):
    data = snapshot(270)
    commit_snapshot(memory, data)
    timeline = tool("get_timeline", memory, start_sec=0, end_sec=100000)
    assert [m["idx"] for m in timeline["moments"]] == list(range(270))
    dialogue = tool("get_person_dialogue", memory, person_id="P003", limit=1000)
    assert dialogue["total"] == 270
    assert [u["clip_idx"] for u in dialogue["utterances"]] == list(range(270))


def test_literal_filter_strings_cannot_change_expression(memory):
    data = read_snapshot(memory, include_vectors=True)
    prefix = 'Ada/%_ " OR ordinal >= 0 OR key = "'
    data["semantic"][0]["key"] = prefix + "suffix"
    commit_snapshot(memory, data)
    result = tool("search_facts", memory, filters={"key_prefix": prefix})
    assert result["count"] == 1 and result["facts"][0]["key"] == prefix + "suffix"


@pytest.mark.parametrize("failure", ["native", "json"])
def test_interrupted_native_commit_recovers_before_reads(tmp_path, monkeypatch, failure):
    pytest.importorskip("zvec")
    from qwen_mm_plugins_omni_memory import memory_io

    directory = tmp_path / "memory"
    commit_snapshot(directory, snapshot(), "zvec")
    updated = snapshot(9)
    original = memory_io.atomic_json
    native_apply = memory_io._apply
    if failure == "json":

        def fail(path, value):
            if pathlib.Path(path).name == "store.json":
                raise OSError("interrupted publish")
            return original(path, value)

        monkeypatch.setattr(memory_io, "atomic_json", fail)
    else:

        def fail(directory, journal):
            raise OSError("interrupted native write")

        monkeypatch.setattr(memory_io, "_apply", fail)
    with pytest.raises(OSError, match="interrupted"):
        commit_snapshot(directory, updated)
    assert (directory / memory_io.JOURNAL).exists()
    monkeypatch.setattr(memory_io, "atomic_json", original)
    monkeypatch.setattr(memory_io, "_apply", native_apply)
    restored = read_snapshot(directory, include_vectors=True)
    assert len(restored["episodic"]) == 9
    assert not (directory / memory_io.JOURNAL).exists()
    backend = open_backend(directory, restored)
    try:
        assert len(backend.scan("utterances")) == 18
    finally:
        backend.close()


def test_json_mode_does_not_import_zvec(tmp_path):
    code = """
import sys
sys.modules['zvec'] = None
from qwen_mm_plugins_omni_memory.memory_io import commit_snapshot, read_snapshot
from qwen_mm_plugins_omni_memory.memory_backend import open_backend
commit_snapshot(sys.argv[1], {'episodic': [], 'semantic': []}, 'json')
assert open_backend(sys.argv[1], read_snapshot(sys.argv[1])).scan('episodes') == []
"""
    source = pathlib.Path(__file__).parents[1] / "src"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, (source, source / "capabilities/omni-memory"))))
    subprocess.run([sys.executable, "-c", code, str(tmp_path)], env=env, check=True, capture_output=True)


def test_shared_modules_match_standalone_skill():
    root = pathlib.Path(__file__).parents[1] / "src/capabilities/omni-memory"
    for name in ("memory_query.py", "memory_backend.py", "memory_io.py", "zvec_backend.py", "retrieval_text.py"):
        assert (root / "qwen_mm_plugins_omni_memory" / name).read_bytes() == (
            root / "skill/script/build_memory" / name
        ).read_bytes()


@pytest.mark.parametrize("hybrid", [False, True])
def test_public_tool_responses_match_across_backends(tmp_path, monkeypatch, hybrid):
    pytest.importorskip("zvec")
    from qwen_mm_plugins_omni_memory import omni_core, watch

    data = snapshot()
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"synthetic replay bytes")
    for clip in data["clips"]:
        clip["path"] = str(video)
    monkeypatch.setattr(service.config, "local_dir", lambda: str(tmp_path))
    monkeypatch.setattr(service, "embed_client", lambda: None)
    monkeypatch.setattr(mem_core, "get_embed_client", lambda: None)
    monkeypatch.setattr(mem_core, "embed_texts", lambda *_: [[1.0, 0.0, 0.0]] if hybrid else None)
    monkeypatch.setattr(omni_core, "get_client", lambda: object())
    monkeypatch.setattr(watch, "replay_answer_stream", lambda *_a, **_k: iter(["stub replay answer"]))
    cases = [
        ("get_memory_status", {}),
        ("get_memory_overview", {}),
        ("get_people", {}),
        ("get_person_dialogue", {"person_id": "P003", "start_sec": 20, "end_sec": 95}),
        ("get_timeline", {"start_sec": 30, "end_sec": 80}),
        ("get_moment", {"idxs": [0, 3]}),
        ("search_memory", {"query": "item0", "top_k": 1}),
        ("search_dialogue", {"query": "item0", "top_k": 1}),
        ("search_facts", {"query": "item0", "top_k": 1}),
        ("plan_and_search", {"question": "item0", "fact_keys": ["Ada/topic/0"], "top_k": 1}),
        ("replay_and_answer", {"question": "What happened?", "idxs": [0]}),
    ]
    responses = []
    for kind in ("json", "zvec"):
        directory = tmp_path / kind
        commit_snapshot(directory, data, kind)
        result = []
        for name, args in cases:
            answer = tool(name, directory, **args)
            answer.pop("label", None)
            answer.pop("memory_dir", None)
            result.append(answer)
        responses.append(result)
    assert responses[0] == responses[1]

    # The twelfth tool reads media directly, without either memory backend.
    module = importlib.import_module("qwen_mm_plugins_omni_memory.tools.watch_and_answer")
    monkeypatch.setattr(watch, "probe_media", lambda _: {"duration": 30})
    monkeypatch.setattr(module, "_watch_source", lambda path, _: (path, True, {"source_used_directly": True}))
    monkeypatch.setattr(watch, "watch_answer", lambda *_a, **_k: "stub direct answer")
    assert module.watch_and_answer(str(video), "What happened?")["answer"] == "stub direct answer"


def test_partial_collection_commit_is_replayed(tmp_path, monkeypatch):
    zvec = pytest.importorskip("zvec")
    from qwen_mm_plugins_omni_memory import memory_io

    commit_snapshot(tmp_path, snapshot(2), "zvec")
    original = zvec.open
    failed = False

    def interrupt(path, *args, **kwargs):
        nonlocal failed
        # prepare_changes opens read-only; fail after episodes was actually flushed.
        if pathlib.Path(path).name == "facts" and not args and not kwargs and not failed:
            failed = True
            raise OSError("interrupted between collections")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(zvec, "open", interrupt)
    with pytest.raises(OSError, match="between collections"):
        commit_snapshot(tmp_path, snapshot(3))
    assert json.loads((tmp_path / "store.json").read_text())["storage"]["counts"]["episodes"] == 2
    assert (tmp_path / memory_io.JOURNAL).exists()
    monkeypatch.setattr(zvec, "open", original)
    restored = read_snapshot(tmp_path, include_vectors=True)
    assert restored["storage"]["counts"] == {"episodes": 3, "facts": 3, "utterances": 6}
    assert restored["storage"]["revision"] == 2
    reader = open_backend(tmp_path, restored)
    try:
        assert len(reader.scan("episodes")) == len(reader.scan("facts")) == 3
        assert len(reader.scan("utterances")) == 6
    finally:
        reader.close()


def test_native_writes_only_changed_rows(tmp_path, monkeypatch):
    pytest.importorskip("zvec")
    from qwen_mm_plugins_omni_memory import zvec_backend

    commit_snapshot(tmp_path, snapshot(3), "zvec")
    changes = []
    original = zvec_backend.apply_changes

    def observe(directory, journal):
        changes.append(journal["changes"])
        return original(directory, journal)

    monkeypatch.setattr(zvec_backend, "apply_changes", observe)
    data = read_snapshot(tmp_path, include_vectors=True)
    data["semantic"][0]["statement"] = "updated fact"
    commit_snapshot(tmp_path, data)
    assert not any(c["reset"] or c["delete"] for c in changes[-1].values())
    assert {p: len(c["upsert"]) for p, c in changes[-1].items()} == {"episodes": 0, "facts": 1, "utterances": 0}


def test_no_vectors_then_dimension_upgrade_and_bit_exact_migration(memory):
    from qwen_mm_plugins_omni_memory.omni_core import emb_decode

    data = snapshot(2)
    for record in data["episodic"] + data["semantic"]:
        record.pop("emb")
    data["dense_ok"] = False
    commit_snapshot(memory, data)
    reader = open_backend(memory, read_snapshot(memory))
    try:
        assert reader.dense("episodes", [1.0, 0.0, 0.0]) == []
        assert reader.sparse("episodes", "camera") == [0]
    finally:
        reader.close()
    data["episodic"][0]["emb"] = [0.125, -0.75, 3e-8]
    data["semantic"][1]["emb"] = [-0.4, 0.0, 0.9]
    commit_snapshot(memory, data)
    migrate(memory, "json")
    result = read_snapshot(memory, include_vectors=True)
    for source, i in (("episodic", 0), ("semantic", 1)):
        np.testing.assert_array_equal(
            emb_decode(result[source][i]["emb"]), np.asarray(data[source][i]["emb"], dtype="float32")
        )
    assert result["episodic"][1].get("emb") is None


def test_filtered_topk_is_not_postfiltered_global_topk(memory):
    data = snapshot(510)
    for i, episode in enumerate(data["episodic"]):
        episode["emb"] = [1.0, 0.0, 0.0] if i < 509 else [0.0, 1.0, 0.0]
    commit_snapshot(memory, data)
    backend = open_backend(memory, read_snapshot(memory))
    try:
        selection = select({"start_sec": 509 * 25 + 29, "end_sec": 509 * 25 + 29})
        assert backend.dense("episodes", [1.0, 0.0, 0.0], selection, limit=1) == [509]
        assert backend.sparse("episodes", "bread", selection, limit=1) == [509]
    finally:
        backend.close()


def test_explicit_bm25_parameters_remain_supported(memory):
    from qwen_mm_plugins_omni_memory.memory_backend import JsonBackend

    data = read_snapshot(memory, include_vectors=True)
    reference = JsonBackend(data)
    backend = open_backend(memory, data)
    try:
        assert backend.sparse("episodes", "camera Ada", k1=0.3, b=0.2, boost=3) == reference.sparse(
            "episodes", "camera Ada", k1=0.3, b=0.2, boost=3
        )
    finally:
        backend.close()


def test_filters_validate_at_mcp_boundary():
    from pydantic import ValidationError

    from qwen_mm_plugins_omni_memory.tools.search_memory import SearchMemoryArgs

    for filters in (
        {"episodes": {"unknown": 1}},
        {"episodes": {"start_sec": 10, "end_sec": 1}},
        {"episodes": {"start_sec": float("nan")}},
        {"facts": {"subject_ids": "P001"}},
    ):
        with pytest.raises(ValidationError):
            SearchMemoryArgs(query="camera", filters=filters)
    assert SearchMemoryArgs.model_json_schema()["$defs"]["EpisodeFilter"]["additionalProperties"] is False


def test_writer_lock_is_exclusive_across_processes(tmp_path):
    from qwen_mm_plugins_omni_memory.memory_io import memory_lock

    code = """
import sys
from qwen_mm_plugins_omni_memory.memory_io import memory_lock
try:
    with memory_lock(sys.argv[1], '.writer.lock', blocking=False):
        print('acquired')
except RuntimeError:
    print('busy')
"""
    source = pathlib.Path(__file__).parents[1] / "src"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, (source, source / "capabilities/omni-memory"))))
    with memory_lock(tmp_path, ".writer.lock", blocking=False):
        result = subprocess.run(
            [sys.executable, "-c", code, str(tmp_path)], env=env, check=True, capture_output=True, text=True
        )
        assert result.stdout.strip() == "busy"
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)], env=env, check=True, capture_output=True, text=True
    )
    assert result.stdout.strip() == "acquired"


def test_standalone_builder_embedding_reuse_and_failed_refresh(tmp_path, monkeypatch):
    from test_omni_memory import _build_side_path

    with _build_side_path():
        writer = importlib.import_module("store_writer")
        storage = importlib.import_module("storage")
        data = snapshot(2)
        data["embedding_model"] = "test-model"
        store = writer.MemoryStore.from_dict(data)
        monkeypatch.setattr(writer, "embed_config", lambda: ("local", "test-model", None))
        calls = []
        monkeypatch.setattr(writer, "get_embed_client", lambda: None)
        monkeypatch.setattr(
            writer, "embed_texts", lambda _, texts: calls.append(texts) or [[0.0, 0.0, 1.0] for _ in texts]
        )
        # Semantic induction replaces dictionaries, but unchanged vectors survive via identity/text hash.
        store.set_semantic([{k: v for k, v in r.items() if k != "emb"} for r in store.semantic])
        assert store.build_index(None) and calls == []
        store.semantic[0]["statement"] = "changed fact"
        assert store.build_index(None) and calls == [["changed fact"]]
        frozen = store.snapshot_for_async()
        store.episodic[0]["parsed"]["visual"]["visual_caption"] = "renamed person"
        assert frozen["episodic"][0]["parsed"]["visual"]["visual_caption"] != "renamed person"
        monkeypatch.setattr(writer, "embed_texts", lambda *_: None)
        assert not store.build_index(None)
        assert store.episodic[0]["emb"] is None and not store.dense_ok

        backend = storage.get_backend(tmp_path)
        monkeypatch.setattr(backend, "_write", lambda *_: (_ for _ in ()).throw(OSError("failed persistence")))
        with pytest.raises(OSError, match="failed persistence"):
            with backend.writer("failed"):
                backend.save_async("failed", store)


@pytest.mark.parametrize("kind", ["json", "zvec"])
def test_pipeline_new_append_resume_with_real_persistence(tmp_path, monkeypatch, kind):
    if kind == "zvec":
        pytest.importorskip("zvec")
    from test_omni_memory import _build_side_path

    with _build_side_path():
        pipeline = importlib.import_module("pipeline")
        storage = importlib.import_module("storage")
        writer = importlib.import_module("store_writer")
        video = tmp_path / "source.mp4"
        video.write_bytes(b"synthetic clip")
        cache = tmp_path / "cache" / "source"
        cache.mkdir(parents=True)
        plan = [{"idx": i, "win_start": i * 25, "win_end": i * 25 + 30, "path": str(video)} for i in range(2)]
        (cache / "plan.json").write_text(json.dumps(plan))
        monkeypatch.setattr(pipeline.omni_core, "get_client", lambda: object())
        monkeypatch.setattr(pipeline.clipping, "INTER_CLIP_SLEEP", 0)
        monkeypatch.setattr(writer, "get_embed_client", lambda: None)
        monkeypatch.setattr(writer, "embed_texts", lambda _, texts: [[1.0, 0.0, 0.0] for _ in texts])
        extracted = []

        def extract(_, clip, state, first):
            import copy

            rec = copy.deepcopy(snapshot(2)["episodic"][clip["idx"]])
            rec.pop("emb")
            extracted.append(clip["idx"])
            return rec, None

        monkeypatch.setattr(pipeline.stages, "extract_clip", extract)
        backend = storage.get_backend(tmp_path / "library")
        options = dict(
            video=str(video),
            window_sec=30,
            step_sec=25,
            max_clips=0,
            rollup_k=5,
            do_consolidate=False,
            height=480,
            namespace="memory",
            clips_dir=str(cache.parent),
            backend=backend,
        )
        list(pipeline.build_memory(**options, retrieval_backend=kind))
        assert len(backend.load("memory").episodic) == 2 and extracted == [0, 1]
        # Simulate a checkpoint after the first clip; resume must keep its vector and skip extraction.
        partial = backend.load("memory")
        partial.episodic.pop()
        with backend.writer("memory"):
            backend.save_async("memory", partial).result()
        extracted.clear()
        list(pipeline.build_memory(**options, resume=True))
        resumed = backend.load("memory")
        assert extracted == [1] and [r["idx"] for r in resumed.episodic] == [0, 1]
        assert resumed.storage["backend"] == kind and resumed.dense_ok
        extracted.clear()
        list(pipeline.build_memory(**options))
        appended = backend.load("memory")
        assert extracted == [0, 1] and [r["idx"] for r in appended.episodic] == [0, 1, 2, 3]
        assert appended.episodic[2]["win_start"] == 55
        assert appended.storage["backend"] == kind


def test_bilingual_retrieval_recovers_labeled_evidence(memory):
    cases = [
        ("A photographer adjusts aperture for a portrait.", "portrait aperture"),
        ("A baker kneads sourdough and heats the oven.", "sourdough oven"),
        ("A cyclist repairs a punctured bicycle tyre.", "punctured tyre"),
        ("A teacher draws a triangle on the blackboard.", "triangle blackboard"),
        ("A musician tunes the violin before rehearsal.", "violin rehearsal"),
        ("A gardener waters orchids in the greenhouse.", "orchids greenhouse"),
        ("厨师正在切洋葱准备晚餐。", "洋葱"),
        ("小朋友在海边收集贝壳。", "贝壳"),
        ("工作人员给行李箱贴上标签。", "行李箱"),
        ("宇航员介绍火箭发射流程。", "火箭"),
        ("画家用水彩描绘远处的山峰。", "水彩山峰"),
        ("售货员演示如何使用咖啡机。", "咖啡机"),
    ]
    data = snapshot(len(cases))
    data["entities"] = []
    for i, (sentence, _) in enumerate(cases):
        data["episodic"][i]["parsed"] = {"visual": {"visual_caption": sentence}}
        data["semantic"][i]["statement"] = sentence
        vector = [float(i == j) for j in range(len(cases))]
        data["episodic"][i]["emb"] = data["semantic"][i]["emb"] = vector
    commit_snapshot(memory, data)
    backend = open_backend(memory, read_snapshot(memory))
    try:
        for i, (_, query) in enumerate(cases):
            for pool in ("episodes", "facts"):
                assert backend.sparse(pool, query, limit=3)[0] == i
                assert backend.dense(pool, data["episodic"][i]["emb"], limit=3)[0] == i
    finally:
        backend.close()


def test_empty_native_memory_can_acquire_its_first_vectors(tmp_path):
    pytest.importorskip("zvec")
    commit_snapshot(tmp_path, {"episodic": [], "semantic": []}, "zvec")
    assert read_snapshot(tmp_path)["storage"]["dimension"] == 0
    commit_snapshot(tmp_path, snapshot(1))
    reader = open_backend(tmp_path, read_snapshot(tmp_path))
    try:
        assert reader.dense("episodes", [1, 0, 0]) == [0]
        assert len(reader.scan("utterances")) == 2
    finally:
        reader.close()


def test_copied_skill_migrates_without_server_package(tmp_path):
    pytest.importorskip("zvec")
    import shutil

    source = pathlib.Path(__file__).parents[1] / "src/capabilities/omni-memory/skill"
    copied = tmp_path / "skill"
    shutil.copytree(source, copied, ignore=shutil.ignore_patterns("__pycache__"))
    directory = tmp_path / "memory"
    commit_snapshot(directory, snapshot(2), "json")
    cli = copied / "script/build_memory/reindex_memory.py"
    env = dict(os.environ, PYTHONPATH="")
    for args in (["--backend", "zvec"], ["--diagnose"], ["--backend", "json"]):
        result = subprocess.run(
            [sys.executable, str(cli), str(directory), *args],
            cwd=tmp_path,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        assert json.loads(result.stdout)["counts"]["episodes"] == 2
    assert read_snapshot(directory)["episodic"][0]["emb"]


def test_unavailable_native_storage_never_silently_uses_json(memory):
    if memory.name != "zvec":
        return
    import shutil

    shutil.rmtree(memory / "zvec" / "facts")
    with pytest.raises(RuntimeError, match="missing or unsafe"):
        tool("get_memory_overview", memory)
