from memex.common.http import DEFAULT_CONNECT_TIMEOUT_S, http_timeout


def test_http_timeout_tight_connect_task_sized_read() -> None:
    # The point of the helper: a dead host fails fast at connect while a
    # slow-but-valid response gets the full task-sized read budget.
    t = http_timeout(120.0)
    assert t.connect == DEFAULT_CONNECT_TIMEOUT_S == 10.0
    assert t.read == 120.0
    assert t.write == 30.0
    assert t.pool == 10.0


def test_http_timeout_overrides() -> None:
    t = http_timeout(5.0, connect=2.0, write=3.0, pool=1.0)
    assert (t.connect, t.read, t.write, t.pool) == (2.0, 5.0, 3.0, 1.0)
