from support_rag.llm_cache import LLMCache, cache_key

REQUEST = {
    "model": "claude-haiku-5-5",
    "system": "Answer from the help-center articles only.",
    "messages": [{"role": "user", "content": "How do I connect a domain?"}],
    "max_tokens": 1024,
    "temperature": 0,
}
USAGE = {"input_tokens": 4500, "output_tokens": 300}


def test_key_ignores_the_order_of_fields():
    reordered = dict(reversed(list(REQUEST.items())))
    assert cache_key(reordered) == cache_key(REQUEST)


def test_any_change_to_the_request_changes_the_key():
    one_word = REQUEST | {"system": "Answer from the help-center articles alone."}
    other_model = REQUEST | {"model": "claude-sonnet-5-5"}
    hotter = REQUEST | {"temperature": 1}
    keys = {cache_key(r) for r in (REQUEST, one_word, other_model, hotter)}
    assert len(keys) == 4


def test_a_request_never_seen_is_a_miss(tmp_path):
    with LLMCache(tmp_path / "cache.sqlite") as cache:
        assert cache.get(REQUEST) is None


def test_a_hit_returns_the_original_usage_and_latency(tmp_path):
    with LLMCache(tmp_path / "cache.sqlite") as cache:
        cache.put(REQUEST, {"text": "Go to Domains."}, USAGE, latency_ms=1840.0)
        hit = cache.get(REQUEST)
    assert hit.response == {"text": "Go to Domains."}
    assert hit.usage == USAGE
    assert hit.latency_ms == 1840.0


def test_the_cache_survives_reopening(tmp_path):
    with LLMCache(tmp_path / "cache.sqlite") as cache:
        cache.put(REQUEST, {"text": "Go to Domains."}, USAGE, latency_ms=1840.0)
    with LLMCache(tmp_path / "cache.sqlite") as cache:
        assert len(cache) == 1
        assert cache.get(REQUEST).response == {"text": "Go to Domains."}


def test_the_first_response_is_kept(tmp_path):
    with LLMCache(tmp_path / "cache.sqlite") as cache:
        cache.put(REQUEST, {"text": "first"}, USAGE, latency_ms=1.0)
        cache.put(REQUEST, {"text": "second"}, USAGE, latency_ms=2.0)
        assert cache.get(REQUEST).response == {"text": "first"}
        assert len(cache) == 1


def test_a_batch_call_is_remembered_as_one(tmp_path):
    with LLMCache(tmp_path / "cache.sqlite") as cache:
        cache.put(REQUEST, {"text": "Go to Domains."}, USAGE, latency_ms=0.0, batch=True)
        assert cache.get(REQUEST).batch is True


def test_a_cache_from_before_batch_calls_is_upgraded_in_place(tmp_path):
    import sqlite3

    path = tmp_path / "cache.sqlite"
    db = sqlite3.connect(path)
    db.execute(
        "CREATE TABLE calls (key TEXT PRIMARY KEY, request TEXT NOT NULL, response TEXT NOT "
        "NULL, usage TEXT NOT NULL, latency_ms REAL NOT NULL, created_at TEXT NOT NULL)"
    )
    db.execute(
        "INSERT INTO calls VALUES (?, '{}', '{\"text\": \"old\"}', '{}', 1.0, 'then')",
        (cache_key(REQUEST),),
    )
    db.commit()
    db.close()
    with LLMCache(path) as cache:
        hit = cache.get(REQUEST)
    assert hit.response == {"text": "old"}
    assert hit.batch is False
