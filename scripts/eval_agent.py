"""Score the LangGraph+MCP agent on 30 analyst questions with checkable answers.

Expected answers come from calling the engine directly (the ground truth for "did the agent relay the
tool output faithfully"), not from the chips' true labels, so this measures the agent and not the classifier.

    python scripts/eval_agent.py --model Qwen/Qwen2.5-1.5B-Instruct
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geoint import data as D  # noqa: E402
from geoint.agent import ask_many  # noqa: E402
from geoint.engine import TriageEngine  # noqa: E402

NEG = re.compile(r"\b(no|not|none|isn't|is not|doesn't|does not|without)\b", re.I)


def build_cases(eng: TriageEngine):
    passes = [p["pass_id"] for p in eng.passes()]
    tri = {p: eng.triage_pass(p) for p in passes}
    dr = {p: eng.drift(eng.load_pass(p)[1]) for p in passes}
    C = []

    C.append({"q": "What collection passes are available for triage?", "tools": [("list_passes", {})],
              "check": ("all", passes)})
    for p in passes:
        C.append({"q": f"How many chips in {p} need an analyst to review them?", "tools": [("triage_pass", {"pass_id": p})],
                  "check": ("all", [str(tri[p]["n_analyst"])])})
    for p in passes:
        C.append({"q": f"Is there any sign of drift in {p}?", "tools": [("drift_check", {"pass_id": p})],
                  "check": ("drift", dr[p]["drift_alarm"])})
    for p in passes:
        C.append({"q": f"In {p}, how many chips look like nothing the model was trained on?",
                  "tools": [("triage_pass", {"pass_id": p})], "check": ("all", [str(tri[p]["n_novel"])])})
    for p in passes:
        top = next(iter(tri[p]["auto_label_counts"]), None)
        C.append({"q": f"What land-cover class was auto-labeled most often in {p}?",
                  "tools": [("triage_pass", {"pass_id": p})], "check": ("all", [top])})
    # single chips: a mix of auto and analyst chips
    chip_ids = []
    for p in ("passA", "passB", "passD"):
        ids, x = eng.load_pass(p)
        chip_ids += [ids[3], ids[17]]
    for cid in chip_ids:
        r = eng.classify(eng.load_chip(cid)[None], [cid])[0]
        C.append({"q": f"Classify chip {cid}.", "tools": [("classify_chip", {"chip_id": cid})], "check": ("all", [r["label"]])})
    novel_ids = [q["chip_id"] for q in tri["passC"]["analyst_queue"] if q["reason"] == "novel"][:4]
    for cid in novel_ids:
        s = eng.similar(eng.load_chip(cid), k=5)
        C.append({"q": f"Chip {cid} was sent to an analyst. Which training classes is it most similar to?",
                  "tools": [("find_similar", {"chip_id": cid})], "check": ("all", [s["neighbor_vote"]])})
    most = max(passes, key=lambda p: tri[p]["n_novel"])
    C.append({"q": "Which collection pass has the most chips flagged as novel?",
              "tools": [("triage_pass", {"pass_id": most})], "check": ("all", [most])})
    return C, {"triage": tri, "drift": dr}


def check_answer(ans: str, check) -> bool:
    kind, val = check
    if kind == "all":
        return all(v is not None and re.search(rf"\b{re.escape(str(v))}\b", ans, re.I) for v in val)
    if kind == "drift":
        said_none = bool(NEG.search(ans))
        return (not said_none) if val else said_none
    raise ValueError(kind)


def grounded(ans: str, question: str, tool_text: str) -> bool:
    """Every multi-digit or decimal number in the answer must come from a tool output or the question.
    Percentages count if they are a tool fraction x100 (0.9978 -> 99.78%)."""
    src = tool_text + " " + question
    src_vals = [float(v) for v in re.findall(r"\d+(?:\.\d+)?", src)]
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(%?)", ans):
        n, pct = m.group(1), m.group(2)
        if len(n) < 2 and "." not in n:
            continue
        if n in src:
            continue
        if pct and any(abs(float(n) / 100 - v) < 6e-5 for v in src_vals):
            continue
        return False
    return True


def score(cases, results):
    rows = []
    for c, r in zip(cases, results):
        calls = [cl for t in r["trace"] for cl in t.get("calls", [])]
        tool_text = r.get("tool_text") or " ".join(t.get("content", "") for t in r["trace"] if "tool_result" in t)
        tool_ok = all(any(cl["name"] == name and all(str(cl["args"].get(k)) == str(v) for k, v in args.items())
                          for cl in calls) for name, args in c["tools"])
        rows.append({"tool_text": tool_text, "q": c["q"], "tool_ok": tool_ok, "answer_ok": check_answer(r["answer"], c["check"]),
                     "grounded": grounded(r["answer"], c["q"], tool_text), "n_calls": len(calls),
                     "malformed": sum(t.get("malformed_calls", 0) for t in r["trace"]),
                     "errors": tool_text.count("ERROR:"), "sec": r["sec"], "answer": r["answer"], "calls": calls})
    n = len(rows)
    summ = {k: sum(r[k] for r in rows) / n for k in ("tool_ok", "answer_ok", "grounded")}
    summ.update({"n": n, "mean_calls": sum(r["n_calls"] for r in rows) / n,
                 "malformed_calls": sum(r["malformed"] for r in rows), "tool_errors": sum(r["errors"] for r in rows),
                 "mean_sec": sum(r["sec"] for r in rows) / n})
    return summ, rows


def replay(calls) -> str:
    """Re-run recorded tool calls directly (deterministic) to recover tool output for re-scoring."""
    from geoint import mcp_server as M
    out = []
    for c in calls:
        try:
            out.append(json.dumps(getattr(M, c["name"])(**c["args"])))
        except Exception as e:
            out.append(f"ERROR: {e}")
    return " ".join(out)


def rescore(path: Path):
    d = json.loads(path.read_text())
    cases, _ = build_cases(TriageEngine())
    results = [{"answer": r["answer"], "sec": r["sec"], "tool_text": r.get("tool_text") or replay(r["calls"]),
                "trace": [{"calls": r["calls"]}]} for r in d["rows"]]
    summ, rows = score(cases, results)
    for r, old in zip(rows, d["rows"]):
        r["malformed"], r["errors"] = old["malformed"], old["errors"]
    summ["malformed_calls"] = sum(r["malformed"] for r in rows)
    summ["tool_errors"] = sum(r["errors"] for r in rows)
    path.write_text(json.dumps({"model": d["model"], "summary": summ, "rows": rows}, indent=2))
    print(path.name, json.dumps(summ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--rescore", type=Path, help="re-score a saved agent_eval_*.json without re-running the LLM")
    a = ap.parse_args()
    if a.rescore:
        return rescore(a.rescore)
    eng = TriageEngine()
    cases, truth = build_cases(eng)
    results = asyncio.run(ask_many([c["q"] for c in cases], a.model))
    summ, rows = score(cases, results)
    tag = a.model.split("/")[-1]
    out = D.ROOT / "reports" / f"agent_eval_{tag}.json"
    out.write_text(json.dumps({"model": a.model, "summary": summ, "rows": rows}, indent=2))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
