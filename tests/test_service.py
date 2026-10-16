"""Service-level tests; skipped when trained artifacts are absent (e.g. a fresh clone before training)."""
import io

import numpy as np
import pytest
from PIL import Image

from geoint import data as D

pytestmark = pytest.mark.skipif(not (D.ART / "chipnet.onnx").exists(), reason="run training + export first")


def _png(arr):
    b = io.BytesIO()
    Image.fromarray(arr).save(b, format="PNG")
    return b.getvalue()


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from serve.app import app
    return TestClient(app)


@pytest.fixture(scope="module")
def engine():
    from geoint.engine import TriageEngine
    return TriageEngine()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and len(r.json()["classes"]) == 8


def test_classify_schema(client, engine):
    ids, chips = engine.load_pass("passA")
    files = [("files", (f"{i}.png", _png(c), "image/png")) for i, c in zip(ids[:5], chips[:5])]
    rows = client.post("/classify", files=files).json()["results"]
    assert len(rows) == 5
    for r in rows:
        assert r["route"] in ("auto", "analyst") and r["label"] in engine.classes
        assert 0 <= r["confidence"] <= 1


def test_bad_image_400(client):
    r = client.post("/classify", files=[("files", ("x.png", b"not an image", "image/png"))])
    assert r.status_code == 400


def test_drift_requires_batch(client, engine):
    _, chips = engine.load_pass("passA")
    files = [("files", (f"{i}.png", _png(c), "image/png")) for i, c in enumerate(chips[:5])]
    assert client.post("/drift", files=files).status_code == 400


def test_onnx_matches_torch(engine):
    torch = pytest.importorskip("torch")
    if not (D.ART / "chipnet_robust.pt").exists():
        pytest.skip("torch checkpoint not committed")
    from geoint.infer import load_net, run
    _, chips = engine.load_pass("passA")
    lt, _ = run(load_net("robust", dev=torch.device("cpu")), chips)
    lo, _ = engine._forward(chips)
    assert (lt.argmax(1) == lo.argmax(1)).all() and np.abs(lt - lo).max() < 1e-3


def test_pass_path_traversal_rejected(engine):
    with pytest.raises(ValueError):
        engine.load_pass("../../geoint")


def test_novel_pass_flags_more_than_routine(engine):
    a, c = engine.triage_pass("passA"), engine.triage_pass("passC")
    assert c["n_novel"] > a["n_novel"]


def test_mcp_tools_listed():
    import asyncio

    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    from geoint.agent import mcp_params

    async def go():
        async with stdio_client(mcp_params()) as (r, w), ClientSession(r, w) as s:
            await s.initialize()
            names = {t.name for t in (await s.list_tools()).tools}
            res = await s.call_tool("triage_pass", {"pass_id": "passA"})
            return names, res
    names, res = asyncio.run(go())
    assert names == {"list_passes", "triage_pass", "classify_chip", "find_similar", "drift_check"}
    assert not res.is_error
