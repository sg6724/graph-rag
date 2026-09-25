import pytest

from medgraph.llm_router import Router


@pytest.mark.live
def test_router_live_roundtrip(tmp_path):
    r = Router(cache_dir=tmp_path).complete("Reply with exactly: OK")
    assert "OK" in r.text.upper()
    print(r.provider, r.model, round(r.latency_ms))
