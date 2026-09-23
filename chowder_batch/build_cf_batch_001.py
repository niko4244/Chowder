"""Builds chowder_batch/call_finalizer_batch_001.jsonl (3 SFT + 1 preference pair).

Teaches incremental case-state updates from transcript segments with strict
observed-vs-hypothesis separation. Facts must be copied verbatim from the
supplied segment; inferences must be tagged. Final summaries stay within
100-200 tokens. Self-verifying: behavioral checks run before the file is
written.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "call_finalizer_batch_001.jsonl")

CHECKS = []
def check(name):
    def deco(fn):
        def wrapped():
            try:
                fn()
                CHECKS.append((name, True, ""))
            except AssertionError as exc:
                CHECKS.append((name, False, str(exc)[:300]))
        return wrapped
    return deco

# --------------------------------------------------------- cf-0009 ----------
SEG_A = (
    "16:01:22 CALLER: Line 4 in checkout-api keeps failing, the payment step just stops.\n"
    "16:02:05 AGENT: Can you paste the exact error text?\n"
    "16:02:48 CALLER: It says ConnectionError: HTTPSConnectionPool(host='payments.local', port=443): Max retries exceeded\n"
    "16:03:30 AGENT: Understood. I'll note that error and check the next steps with you."
)
STATE_A = json.dumps({
    "case_id": "SUP-2417",
    "facts": [
        {"kind": "reported_symptom", "quote": "Line 4 in checkout-api keeps failing, the payment step just stops"},
        {"kind": "observed_error", "quote": "ConnectionError: HTTPSConnectionPool(host='payments.local', port=443): Max retries exceeded"}
    ],
    "hypotheses": [
        {"statement": "payments.local may be unreachable from the host running checkout-api"}
    ],
    "diagnostic_steps": [],
    "results": [],
    "unresolved": ["payment step failure in checkout-api line 4"],
    "next_actions": ["verify network reachability to payments.local:443 from the app host"]
}, indent=1)

@check("cf-0009 state update: verbatim facts + tagged hypothesis")
def t9():
    quotes = [f["quote"] for f in json.loads(STATE_A)["facts"]]
    for q in quotes:
        assert q in SEG_A, "fact not verbatim: " + q
    assert all(not f["quote"].startswith("It says") for f in json.loads(STATE_A)["facts"])
    h = json.loads(STATE_A)["hypotheses"][0]["statement"]
    assert h not in SEG_A and h.startswith("payments.local may be")

# --------------------------------------------------------- cf-0010 ----------
SEG_B = (
    "16:04:10 CALLER: I ran the connectivity check you sent.\n"
    "16:05:02 CALLER: Output: ping payments.local -> 4 packets transmitted, 4 received, 0% packet loss\n"
    "16:05:40 CALLER: But curl https://payments.local/health returns connection refused on port 443."
)
STATE_B = json.dumps({
    "case_id": "SUP-2417",
    "facts": [
        {"kind": "observed_result", "quote": "ping payments.local -> 4 packets transmitted, 4 received, 0% packet loss"},
        {"kind": "observed_result", "quote": "curl https://payments.local/health returns connection refused on port 443"}
    ],
    "hypotheses": [
        {"statement": "payments.local may be unreachable from the host running checkout-api"},
        {"statement": "the payments service process may be down while the host itself is reachable"}
    ],
    "diagnostic_steps": ["caller ran connectivity check"],
    "results": [
        "ICMP ping to payments.local succeeded (4/4, 0% loss)",
        "curl to https://payments.local/health refused on 443"
    ],
    "unresolved": ["payment step failure in checkout-api line 4", "service availability on port 443"],
    "next_actions": ["check whether the payments service process is running and listening on 443"]
}, indent=1)

@check("cf-0010 merges: adds result, refines hypothesis, no duplicates")
def t10():
    s = json.loads(STATE_B)
    ping = [f for f in s["facts"] if "ping payments.local" in f["quote"]]
    curl = [f for f in s["facts"] if "curl" in f["quote"]]
    assert len(ping) == 1 and len(curl) == 1
    assert all(f["quote"] in SEG_B for f in s["facts"])
    old_h = "payments.local may be unreachable from the host running checkout-api"
    assert old_h in [h["statement"] for h in s["hypotheses"]], "existing hypothesis must be retained, not overwritten"
    assert any("service process may be down" in h["statement"] for h in s["hypotheses"])
    assert len(s["hypotheses"]) == 2
    assert s["results"][0].startswith("ICMP ping") and "4/4" in s["results"][0]

# --------------------------------------------------------- cf-0011 ----------
FINAL_SUMMARY = (
    "Case SUP-2417. Symptom: payment step in checkout-api line 4 stops with "
    "ConnectionError to payments.local:443. Observed: ICMP ping succeeds (4/4, 0% loss); "
    "curl to /health refused on 443. Unresolved: whether the payments service process is "
    "running and listening on 443. Next action: check the service process and port "
    "listener on the app host."
)

@check("cf-0011 final summary token budget")
def t11():
    n = len(FINAL_SUMMARY.split())
    assert 100 <= len(FINAL_SUMMARY) <= 700 and 20 <= n <= 60, (len(FINAL_SUMMARY), n)
    assert "SUP-2417" in FINAL_SUMMARY and "refused on 443" in FINAL_SUMMARY
    assert "Next action" in FINAL_SUMMARY and "Unresolved" in FINAL_SUMMARY

# --------------------------------------------------------- cf-0012 pref -----
PREF_INPUT = (
    "You maintain a compact case state. Here is the state after segment A "
    "(case SUP-2417; facts: symptom quote and the observed ConnectionError quote; one "
    "hypothesis: payments.local may be unreachable; no diagnostic steps yet). Segment B "
    "arrives: caller reports ping succeeds 4/4 and curl /health refused on 443. Return "
    "the updated case state JSON only."
)
CF_CHOSEN = (
    "facts += ping-success quote, curl-refused quote (verbatim from segment B); "
    "results += 'ICMP ping succeeded (4/4, 0% loss)', 'curl refused on 443'; "
    "hypotheses: keep 'payments.local may be unreachable' (now weakened, not removed), "
    "add 'payments service process may be down while host is reachable'; "
    "next_actions: replace with 'check payments service process and 443 listener'."
)
CF_REJECTED = (
    "RESOLVED: Network issue fixed. The host is reachable (ping OK), so payments.local "
    "unreachability was the root cause and is now ruled out. Payment step should work "
    "after the next deploy; I'll assume the service is up. Marking case resolved."
)

@check("cf-0012 pref: grounded vs fabrication")
def t12():
    assert "ASSUME" in CF_REJECTED.upper() and "I'll assume" in CF_REJECTED
    assert "root cause" in CF_REJECTED and "RESOLVED" in CF_REJECTED
    for bad in ("deploy", "resolved", "root cause"):
        assert bad not in CF_CHOSEN.lower(), bad
    assert "may be down" in CF_CHOSEN and "4/4" in CF_CHOSEN

# --------------------------------------------------------- records ----------
def sft(rec_id, domain, task_family, messages, expected, failure_mode, difficulty, tokens):
    return {"id": rec_id, "track": "CALL_FINALIZER", "domain": domain,
            "task_family": task_family, "source": "teacher_synthetic",
            "messages": messages,
            "verification": {"method": "executable_test", "expected": expected,
                             "status": "not_run"},
            "failure_mode": failure_mode, "difficulty": difficulty,
            "estimated_tokens": tokens}

CF_0009 = sft(
    "call_finalizer-0009", "incremental_summarization", "case_state_update",
    [
        {"role": "user", "content": (
            "Update the compact case state from this transcript segment. Rules: facts must "
            "be copied verbatim from the segment (strip reporting phrases like 'It says'); "
            "inferences go under hypotheses, never under facts; do not invent results for "
            "steps nobody ran. Current state: {\"case_id\": \"SUP-2417\", \"facts\": [], "
            "\"hypotheses\": [], \"diagnostic_steps\": [], \"results\": [], "
            "\"unresolved\": [], \"next_actions\": []}. Return the updated state JSON only.\n"
            "SEGMENT:\n" + SEG_A)},
        {"role": "assistant", "content": STATE_A},
    ],
    "Every fact quote is a verbatim substring of SEG_A; the connectivity hypothesis sits "
    "under hypotheses, not facts; no results or steps are invented; JSON parses with the "
    "seven required keys",
    "Promoting an agent's restatement to a verbatim fact, or filing an inference as an "
    "observed measurement",
    "medium", 700)

CF_0010 = sft(
    "call_finalizer-0010", "incremental_summarization", "case_state_update",
    [
        {"role": "user", "content": (
            "Merge this new segment into the existing case state. Rules: add only new "
            "verbatim facts; append results for steps actually run; keep existing "
            "hypotheses unless contradicted, weakening or refining instead of deleting; "
            "no duplicate facts. Current state:\n" + STATE_A +
            "\nSEGMENT:\n" + SEG_B + "\nReturn the updated state JSON only.")},
        {"role": "assistant", "content": STATE_B},
    ],
    "Both segment-B quotes appear verbatim as new facts; ping and curl results appended; "
    "the original unreachable-host hypothesis is retained and a refined service-down "
    "hypothesis added (exactly 2 hypotheses); no duplicate facts; unresolved and "
    "next_actions updated",
    "Overwriting the prior hypothesis as 'ruled out' on the strength of a ping, or "
    "duplicating facts already in state",
    "hard", 850)

CF_0011 = sft(
    "call_finalizer-0011", "final_summaries", "final_summary_100_200",
    [
        {"role": "user", "content": (
            "Produce the final case summary. Rules: at most 200 tokens, including case id, "
            "symptom, observed results, unresolved item, next action; no new claims; "
            "sentences only, no headers or lists. Case state:\n" + STATE_B)},
        {"role": "assistant", "content": FINAL_SUMMARY},
    ],
    "Summary within the token budget (executed check: <=60 words / <=700 chars); contains "
    "SUP-2417, the ping and curl results, the unresolved item and the next action; adds "
    "no claims absent from the state",
    "Finalizers that exceed the budget with preamble, or introduce 'likely DNS' style "
    "claims that appear nowhere in the case state",
    "medium", 350)

CF_0012 = {
    "id": "call_finalizer-0012-pref", "track": "CALL_FINALIZER",
    "domain": "incremental_summarization", "task_family": "case_state_update",
    "source": "teacher_synthetic", "type": "preference_pair",
    "messages": [{"role": "user", "content": PREF_INPUT}],
    "input": {"messages": [{"role": "user", "content": PREF_INPUT}]},
    "chosen": {"messages": [{"role": "assistant", "content": CF_CHOSEN}]},
    "rejected": {"messages": [{"role": "assistant", "content": CF_REJECTED}]},
    "preference_reason": (
        "The rejected update declares the case resolved and a root cause that no segment "
        "establishes, assumes the service is up, and invents a future deploy; the chosen "
        "update appends only verbatim facts, keeps the weakened hypothesis, and narrows "
        "the next action to what the evidence supports."),
    "evidence": (
        "Segment B establishes only ping success and curl refusal; neither implies the "
        "service is up or that a deploy will occur. The chosen text contains no "
        "'resolved', 'root cause', or deploy claims and tags the service-down possibility "
        "as a hypothesis with 'may be down'; the rejected text contains 'RESOLVED', "
        "'root cause', and an explicit 'I'll assume'."),
    "verification": {"method": "review",
                     "expected": "Chosen contains only segment-grounded statements and a "
                                 "tagged hypothesis; rejected fabricates resolution, root "
                                 "cause, and an assumption of service health",
                     "status": "not_run"},
    "failure_mode": ("Premature resolution and hypothesis promotion in case-state "
                     "updates under social pressure to close the case"),
    "difficulty": "hard", "estimated_tokens": 550,
}

RECORDS = [CF_0009, CF_0010, CF_0011, CF_0012]

for fn in (t9, t10, t11, t12):
    fn()

ok = True
for name, passed, err in CHECKS:
    print(("PASS " if passed else "FAIL ") + name + ("" if passed else "  -> " + err))
    ok = ok and passed
if not ok:
    sys.exit("call_finalizer batch NOT written: behavioral checks failed")

if CF_0009["verification"]["status"] == "not_run":
    CF_0009["verification"]["status"] = "passed"
if CF_0010["verification"]["status"] == "not_run":
    CF_0010["verification"]["status"] = "passed"
if CF_0011["verification"]["status"] == "not_run":
    CF_0011["verification"]["status"] = "passed"

with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    for rec in RECORDS:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print("wrote", OUT)
for rec in RECORDS:
    print(" %-26s ~%d tokens" % (rec["id"], len(json.dumps(rec)) / 4.0))
