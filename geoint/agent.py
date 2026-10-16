"""LangGraph analyst-assistant agent that calls the geoint-triage MCP server.

Runs fully offline with a small open-weights model (Qwen2.5-Instruct via transformers) and no API key.
The agent and the tools are separate processes connected by MCP over stdio, the same way Claude Desktop
or Cursor would connect.

    python -m geoint.agent "Which passes need an analyst today?"
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SYSTEM = (
    "You are an imagery-triage assistant for an intelligence analyst. You can call tools that classify "
    "satellite image chips from collection passes (pass ids look like 'passA'; chip ids look like 'passA-012').\n"
    "Rules:\n"
    "- Use the tools to answer. Never invent counts, labels, scores or chip ids. Copy numbers exactly from tool results.\n"
    "- A chip routed to 'analyst' needs human review: reason 'novel' means it looks unlike anything the model was "
    "trained on; 'low_confidence' means the model is unsure.\n"
    "- drift_alarm=true means the pass looks different from validated data (weather, sensor, compression or new content).\n"
    "- Keep the final answer short and factual."
)
MAX_STEPS = 6
TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    steps: int
    trace: list


class LocalLLM:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-1.5B-Instruct", device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float16 if self.device != "cpu" else torch.float32).to(self.device).eval()
        self.model_id = model_id

    def __call__(self, messages: list[dict], tools: list[dict]) -> str:
        prompt = self.tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)
        ids = self.tok(prompt, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=400, do_sample=False, temperature=None, top_p=None, top_k=None,
                                      pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def _to_chat(m) -> dict:
    """LangGraph stores messages as langchain objects; convert back to the HF chat-template dict format."""
    role = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}[m.type]
    d = {"role": role, "content": m.content}
    calls = getattr(m, "tool_calls", None)
    if role == "assistant" and calls:
        d["tool_calls"] = [{"type": "function", "function": {"name": c["name"], "arguments": c["args"]}} for c in calls]
    return d


def build_graph(llm: LocalLLM, session: ClientSession, tool_schemas: list[dict]):
    from langchain_core.messages import AIMessage, ToolMessage

    def agent_node(state: State):
        text = llm([_to_chat(m) for m in state["messages"]], tool_schemas)
        calls, bad = [], 0
        for i, blob in enumerate(TOOL_RE.findall(text)):
            try:
                j = json.loads(blob)
                calls.append({"name": j["name"], "args": j.get("arguments", {}), "id": f"c{state['steps']}_{i}"})
            except (json.JSONDecodeError, KeyError):
                bad += 1
        content = "" if calls else TOOL_RE.sub("", text).strip()
        rec = {"step": state["steps"], "calls": [{"name": c["name"], "args": c["args"]} for c in calls],
               "malformed_calls": bad, "text": None if calls else content}
        return {"messages": [AIMessage(content=content, tool_calls=calls)], "steps": state["steps"] + 1,
                "trace": state["trace"] + [rec]}

    async def tool_node(state: State):
        last = state["messages"][-1]
        msgs, trace = [], list(state["trace"])
        for c in last.tool_calls:
            try:
                r = await session.call_tool(c["name"], c["args"])
                txt = "\n".join(getattr(b, "text", "") for b in r.content)
                if r.is_error:
                    txt = f"ERROR: {txt}"
            except Exception as e:  # unknown tool / bad args
                txt = f"ERROR: {e}"
            msgs.append(ToolMessage(content=txt, tool_call_id=c["id"], name=c["name"]))
            trace.append({"tool_result": c["name"], "content": txt})
        return {"messages": msgs, "trace": trace}

    def route(state: State):
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None) and state["steps"] < MAX_STEPS:
            return "tools"
        return END

    g = StateGraph(State)
    g.add_node("agent", agent_node)
    g.add_node("tools", tool_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()


def mcp_params() -> StdioServerParameters:
    return StdioServerParameters(command=sys.executable, args=["-m", "geoint.mcp_server"])


async def tool_schemas(session: ClientSession) -> list[dict]:
    tools = (await session.list_tools()).tools
    return [{"type": "function", "function": {"name": t.name, "description": t.description or "",
                                              "parameters": t.input_schema}} for t in tools]


async def ask_many(questions: list[str], model_id: str) -> list[dict]:
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = LocalLLM(model_id)
    out = []
    async with stdio_client(mcp_params()) as (r, w), ClientSession(r, w) as session:
        await session.initialize()
        schemas = await tool_schemas(session)
        graph = build_graph(llm, session, schemas)
        for q in questions:
            t0 = time.perf_counter()
            st = await graph.ainvoke({"messages": [SystemMessage(SYSTEM), HumanMessage(q)], "steps": 0, "trace": []},
                                     {"recursion_limit": 2 * MAX_STEPS + 2})
            final = st["messages"][-1].content if not getattr(st["messages"][-1], "tool_calls", None) else ""
            out.append({"question": q, "answer": final, "trace": st["trace"], "sec": time.perf_counter() - t0})
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="+")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    a = ap.parse_args()
    res = asyncio.run(ask_many([" ".join(a.question)], a.model))[0]
    for t in res["trace"]:
        if t.get("calls"):
            print("→", ", ".join(f"{c['name']}({json.dumps(c['args'])})" for c in t["calls"]))
    print(res["answer"])
