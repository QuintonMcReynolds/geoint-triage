"""MCP server exposing chip triage as tools an LLM agent (or Claude Desktop / Cursor) can call.

    python -m geoint.mcp_server          # stdio transport
"""
from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from .engine import TriageEngine

mcp = MCPServer("geoint-triage")
_engine: TriageEngine | None = None


def engine() -> TriageEngine:
    global _engine
    if _engine is None:
        _engine = TriageEngine()
    return _engine


@mcp.tool()
def list_passes() -> list[dict]:
    """List the imagery collection passes available for triage, with their chip counts."""
    return engine().passes()


@mcp.tool()
def triage_pass(pass_id: str) -> dict:
    """Classify every chip in a collection pass. Returns how many were auto-labeled (and as what) and the
    analyst queue: chips held back as novel (unlike anything in training) or low confidence."""
    r = engine().triage_pass(pass_id)
    r["analyst_queue"] = r["analyst_queue"][:25]
    return r


@mcp.tool()
def classify_chip(chip_id: str) -> dict:
    """Classify one chip (e.g. 'passB-017'). Returns label, calibrated confidence, top-3, novelty score and route."""
    return engine().classify(engine().load_chip(chip_id)[None], [chip_id])[0]


@mcp.tool()
def find_similar(chip_id: str, k: int = 5) -> dict:
    """Nearest labeled training chips to this chip in embedding space. Useful to explain a novel or uncertain chip."""
    return engine().similar(engine().load_chip(chip_id), k=max(1, min(int(k), 20)))


@mcp.tool()
def drift_check(pass_id: str) -> dict:
    """Test whether a whole pass is distributed differently from what the model was validated on
    (sensor/weather/compression change or new content). Returns an MMD p-value and a drift_alarm flag."""
    _, chips = engine().load_pass(pass_id)
    return {"pass_id": pass_id, **engine().drift(chips)}


if __name__ == "__main__":
    mcp.run()
