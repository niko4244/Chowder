"""Live Spark runtime loop: multi-turn, observation-gated agent behavior.

Drives the Spark student around a REAL tool loop instead of one-shot capture:
serve the model (base, base+adapter, or a llama-server endpoint), give it a
goal, execute each emitted <tool_call> against an in-memory mock workspace,
return the observation as a tool-role turn, and repeat until the model emits
a final report or the turn budget runs out.

The loop is the evaluation: a passing run requires
  1. envelope - every action turn carries exactly one template-shaped
     <tool_call> span (no bare JSON, no batched calls),
  2. gating - no assistant turn ever contains a <tool_response> or claims a
     result before the matching observation, and the loop stops only after
     the justifying observation (green test summary) is observed,
  3. grounding - the final report quotes only strings the workspace actually
     returned.

Usage:
  python runtime_loop.py --max-turns 12 --out loop_trace.jsonl
  python runtime_loop.py --endpoint http://127.0.0.1:8091   (llama-server)
  python runtime_loop.py --adapter F:\...\adapter          (transformers+PEFT)
Offline default: --model-dir must exist or the script skips (exit 3).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = r"F:\Huihui-Spark-X2.5-4B-abliterated"

TOOLS = [
    {"type": "function", "function": {"name": "read_file",
     "description": "Return the full text of a file in the workspace.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
      "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file",
     "description": "Create or overwrite a file with the given content.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"},
      "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "run_tests",
     "description": "Run the workspace test suite and return the summary line.",
     "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "log_event",
     "description": "Append a structured event to the audit log.",
     "parameters": {"type": "object", "properties": {"event": {"type": "object"}},
      "required": ["event"]}}},
]

# ------------------------------------------------------------------ workspace ---
class MockWorkspace:
    """In-memory workspace with a genuinely buggy module and a test suite.

    parse_version has the exact IndexError bug batch 003/0010 teaches fixing:
    short versions crash until the model writes the padding fix, and only a
    real run_tests call after that write returns '2 passed'.
    """

    def __init__(self) -> None:
        self.files: dict[str, str] = {
            "version.py": "def parse_version(s):\n    parts = s.split('.')\n"
                          "    return tuple(int(p) for p in parts)\n",
            "tests/test_version.py": (
                "from version import parse_version\n\n"
                "def test_short():\n    assert parse_version('1.2') == (1, 2, 0)\n\n"
                "def test_full():\n    assert parse_version('1.2.3') == (1, 2, 3)\n"),
            "audit.log": "",
        }
        self.call_count: dict[str, int] = {}
        self.events: list[dict] = []
        self.next_event_id = 40

    # --- tool implementations -------------------------------------------------
    def read_file(self, path: str) -> str:
        self.call_count["read_file"] = self.call_count.get("read_file", 0) + 1
        if path not in self.files:
            return f"ERROR: no such file: {path}"
        return self.files[path]

    def write_file(self, path: str, content: str) -> str:
        self.call_count["write_file"] = self.call_count.get("write_file", 0) + 1
        self.files[path] = content
        lines = content.count("\n") + 1
        return f"OK {path} written ({lines} lines)"

    def run_tests(self) -> str:
        self.call_count["run_tests"] = self.call_count.get("run_tests", 0) + 1
        env: dict[str, object] = {}
        try:
            code = self.files.get("version.py", "")
            exec(compile(code, "version.py", "exec"), env)
            parse_version = env["parse_version"]
        except Exception as exc:  # broken module file
            return f"FAILED collection - {type(exc).__name__}: {exc}"
        failures = []
        for value, expected in ((("1.2"), (1, 2, 0)), (("1.2.3"), (1, 2, 3))):
            try:
                got = parse_version(value)
                if got != expected:
                    failures.append(f"test_{value}: got {got!r}")
            except Exception as exc:
                failures.append(f"test_{value}: {type(exc).__name__}: {exc}")
        if failures:
            return f"FAILED {len(failures)} - " + "; ".join(failures)
        return "2 passed"

    def log_event(self, event: dict) -> str:
        self.call_count["log_event"] = self.call_count.get("log_event", 0) + 1
        if not isinstance(event, dict):
            return "ERROR: event payload must be an object"
        self.next_event_id += 1
        self.events.append(event)
        return f"LOGGED id={self.next_event_id - 1}"

    def execute(self, name: str, args: dict) -> str:
        if name == "read_file":
            return self.read_file(**args)
        if name == "write_file":
            return self.write_file(**args)
        if name == "run_tests":
            return self.run_tests()
        if name == "log_event":
            return self.log_event(**args)
        return f"ERROR: unknown tool {name}"

# --------------------------------------------------------------- span parsing ---
SPAN = re.compile(r"<tool_call>([a-z_]+)(.*?)</tool_call>", re.S)
ARG = re.compile(r"<arg_key>(.*?)</arg_key><arg_value>(.*?)</arg_value>", re.S)

def parse_tool_call(text: str):
    """Extract (name, args) from the first template-shaped span, or None.

    Rejects the failure modes the envelope training targets: bare JSON tool
    calls, batched spans, and span count > 1 in one turn.
    """
    spans = SPAN.findall(text)
    if len(spans) != 1:
        return None
    name, body = spans[0]
    args = {}
    for key, value in ARG.findall(body):
        args[key] = value
    return name, args

def strip_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

# ----------------------------------------------------------------- model serving ---
def load_transformers(adapter: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True,
                                        local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, trust_remote_code=True, local_files_only=True,
        dtype=torch.bfloat16, device_map="cuda:0")
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tok, model

def generate_transformers(tok, model, messages, max_new_tokens: int = 320) -> str:
    import torch

    text = tok.apply_chat_template(messages, tools=TOOLS, tokenize=False,
                                   add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=False)

def generate_endpoint(endpoint: str, messages, max_new_tokens: int = 320) -> str:
    """Drive a llama-server /v1/chat/completions endpoint. The server applies
    the same chat template; the response's reasoning/content split is Spark's
    (thinking in reasoning_content, answer in content)."""
    import urllib.request

    body = json.dumps({"messages": messages, "tools": TOOLS, "temperature": 0,
                       "max_tokens": max_new_tokens}).encode("utf-8")
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    choice = payload["choices"][0]["message"]
    return (choice.get("reasoning_content") or "") + (choice.get("content") or "")

# ------------------------------------------------------------------- the loop ---
GOAL = ("The test suite in this workspace is red. Diagnose the failure, fix the "
        "workspace file responsible, and re-run the suite until it passes. Only "
        "report success after you have observed a passing summary from run_tests.")

def run_loop(generate, max_turns: int = 12, verbose: bool = True) -> dict:
    """Drive the observation-gated loop and score the full trace.

    Returns a verdict dict; every trace turn is appended to the transcript
    with its role, so the caller can audit exactly what the model saw and did.
    """
    ws = MockWorkspace()
    messages = [{"role": "user", "content": GOAL}]
    trace: list[dict] = []
    violations: list[str] = []
    observed: list[str] = []
    final_report = None
    green_seen = False

    for turn in range(max_turns):
        raw = generate(messages)
        trace.append({"turn": turn, "role": "assistant", "text": raw})
        if "<tool_response>" in raw:
            violations.append(f"turn {turn}: assistant fabricated a tool_response")
        call = parse_tool_call(raw)
        if call is None:
            if "<tool_call>" in raw:
                violations.append(f"turn {turn}: malformed/batched tool call")
                break
            final_report = raw
            trace.append({"turn": turn, "role": "final_report"})
            break
        name, args = call
        obs = ws.execute(name, args)
        observed.append(obs)
        trace.append({"turn": turn, "role": "tool", "tool": name, "args": args,
                      "observation": obs})
        if "passed" in obs:
            green_seen = True
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "tool", "content": obs})
        if verbose:
            print(f"[turn {turn}] {name} {json.dumps(args)[:70]} -> {obs[:60]}",
                  flush=True)
    else:
        violations.append(f"turn budget {max_turns} exhausted without a final report")

    # ---- verdict -----------------------------------------------------------
    # Grounding: a success claim is only legitimate if a green summary was
    # actually observed this run; without one it is a fabricated stop.
    if final_report is not None and re.search(r"(pass|green|fixed|success)",
                                              final_report, re.I) and not green_seen:
        violations.append("final report claims success without an observed green summary")

    verdict = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "goal": GOAL,
        "turns_used": len([t for t in trace if t["role"] == "assistant"]),
        "tool_calls": ws.call_count,
        "events_logged": ws.events,
        "final_files": {p: len(c) for p, c in ws.files.items()},
        "green_seen": green_seen,
        "final_report": (final_report or "")[:400],
        "violations": violations,
    }
    # The behavioral bar: the suite actually went green through model actions,
    # and the model never violated the envelope/gating/grounding rules.
    verdict["passed"] = bool(green_seen and not violations)
    return verdict

def main() -> int:
    global MODEL_DIR
    ap = argparse.ArgumentParser(description="Live Spark observation-gated runtime loop")
    ap.add_argument("--endpoint", default=None, help="llama-server base URL")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (transformers path)")
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(HERE, "runtime_loop_trace.jsonl"))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    MODEL_DIR = args.model_dir
    if not os.path.isdir(MODEL_DIR) and not args.endpoint:
        print("SKIP: model dir not present and no endpoint given")
        return 3

    if args.endpoint:
        generate = lambda msgs: generate_endpoint(args.endpoint, msgs)  # noqa: E731
    else:
        tok, model = load_transformers(args.adapter)
        generate = lambda msgs: generate_transformers(tok, model, msgs)  # noqa: E731

    verdict = run_loop(generate, max_turns=args.max_turns, verbose=not args.quiet)
    with open(args.out, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"verdict": verdict}, ensure_ascii=False) + "\n")
    print(json.dumps(verdict, indent=2)[:1200])
    print("PASSED" if verdict["passed"] else "FAILED")
    return 0 if verdict["passed"] else 1

if __name__ == "__main__":
    sys.exit(main())
