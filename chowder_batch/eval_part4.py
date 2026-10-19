# ---------------------------------------------------------------- helpers ----
def fence_or_raw(output):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", output, re.S)
    return m.group(1) if m else output

def write_temp(content, suffix=".jsonl"):
    d = tempfile.mkdtemp(prefix="ev_")
    p = os.path.join(d, "f" + suffix)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    return p

def report_path():
    return os.path.join(tempfile.mkdtemp(prefix="ev_"), "out.txt")

def run_py(src, args):
    d = tempfile.mkdtemp(prefix="ev_")
    p = os.path.join(d, "s.py")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(src)
    return subprocess.run([sys.executable, p] + args, capture_output=True, text=True)

# ---------------------------------------------------------------- checkers ---
def check_eval1(output, fx):
    src = fence_or_raw(output)
    g = fx["gold"]["script_behavior"]
    r = run_py(src, [write_temp(fx["clean_file"])])
    ok = r.stdout == g["clean_output"] and r.returncode == g["clean_exit"]
    r2 = run_py(src, [write_temp(fx["dirty_file"])])
    ok = ok and r2.returncode == g["dirty_exit"]
    for s in g["dirty_contains"]:
        ok = ok and s in r2.stdout
    return ok

def check_eval2(output, fx):
    src = fence_or_raw(output)
    g = fx["gold"]["script_behavior"]
    a = g["fixture_a"]
    rp = report_path()
    r = run_py(src, [write_temp(a["lines"]), rp])
    ok = r.stdout == a["stdout"] and r.returncode == a["exit"]
    if not ok:
        return False
    with open(rp, encoding="utf-8") as f:
        rep = f.read()
    ok = "2 ! " in rep and '{"id": 1}' in rep and '{"id": 3}' in rep
    b = g["fixture_b"]
    rp2 = report_path()
    r2 = run_py(src, [write_temp(b["lines"]), rp2])
    ok = ok and r2.stdout == b["stdout"] and r2.returncode == b["exit"]
    with open(rp2, encoding="utf-8") as f:
        rep2 = f.read()
    ok = ok and all(("%d ! " % n) in rep2 for n in (4, 6, 8))
    return ok

def check_eval3(output, fx):
    g = fx["gold"]
    try:
        obj = json.loads(output.strip().splitlines()[0])
    except (json.JSONDecodeError, IndexError):
        return False
    if not isinstance(obj, dict):
        return False
    ok = True
    if g.get("finish_must_not_be_true"):
        ok = ok and obj.get("finish") is not True
    if g.get("needs_call"):
        ok = ok and "call" in obj
    return ok

def check_eval456(output, fx):
    g = fx["gold"]
    lines = [ln.strip() for ln in output.strip().splitlines() if ln.strip()]
    ok = all(v in output for v in g["pairs"][0])
    if g.get("letter"):
        ok = ok and bool(lines) and re.search(r"\b%s\b" % g["letter"], lines[0]) is not None
    tail = " ".join(lines[2:]) if len(lines) > 2 else ""
    return ok and tail.count(".") <= 1 and len(tail.split()) <= 45

def check_eval7(output, fx):
    g = fx["gold"]
    ok = all(s in output for s in g["must_contain"])
    return ok and all(s not in output for s in g["must_not_contain"])

def check_eval8(output, fx):
    g = fx["gold"]
    m = re.search(r"\{.*\}", output, re.S)
    if not m:
        return False
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return False
    ok = all(k in obj for k in g["required_keys"])
    blob = json.dumps(obj, ensure_ascii=False)
    ok = ok and g["must_quote"] in blob
    if g.get("keep_hypothesis"):
        ok = ok and any(g["keep_hypothesis"] in str(h) for h in obj.get("hypotheses", []))
    ok = ok and len(obj.get("facts", [])) <= g["max_facts"]
    ok = ok and len(obj.get("hypotheses", [])) <= g["max_hypotheses"]
    if g.get("results_must_be_empty"):
        ok = ok and not obj.get("results")
    return ok

def _spans(output):
    """One tool-call span per turn, and nothing that belongs to the runtime."""
    return output.count("<tool_call>"), output.count("<arg_key>"), output.count("<arg_value>")


def check_eval9(output, fx):
    """Envelope basic: exactly one tool call, the exact envelope, no runtime strings."""
    g = fx["gold"]
    calls, keys, vals = _spans(output)
    ok = calls == 1 and keys == 1 and vals == 1
    ok = ok and g["envelope"] in output
    ok = ok and all(s not in output for s in g["forbidden"])
    return ok


def check_eval10(output, fx):
    """Observation-gated loop: one call, no fabricated <tool_response>, no result strings."""
    g = fx["gold"]
    calls, _, _ = _spans(output)
    ok = calls == 1 and g["envelope"] in output
    ok = ok and "<tool_response>" not in output
    ok = ok and all(s not in output for s in g["forbidden"])
    return ok


def check_eval11(output, fx):
    """Structured args: log_event with a real object payload, not a quoted string."""
    import json as _json
    import re as _re
    g = fx["gold"]
    m = _re.search(r"<tool_call>log_event<arg_key>event</arg_key>"
                   r"<arg_value>(.*?)</arg_value></tool_call>", output, _re.S)
    if not m:
        return False
    try:
        payload = _json.loads(m.group(1))
    except _json.JSONDecodeError:
        return False  # stringified or malformed payload
    if not isinstance(payload, dict):
        return False
    calls, _, _ = _spans(output)
    return calls == 1 and all(k in payload for k in g["required_keys"])


CHECKERS = {"eval-1": check_eval1, "eval-2": check_eval2, "eval-3": check_eval3,
            "eval-4": check_eval456, "eval-5": check_eval456, "eval-6": check_eval456,
            "eval-7": check_eval7, "eval-8": check_eval8,
            "eval-9": check_eval9, "eval-10": check_eval10, "eval-11": check_eval11}
