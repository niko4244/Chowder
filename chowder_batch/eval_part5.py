# ------------------------------------------------------------ anti-leak ------
def load_training_texts():
    texts = []
    for name in BATCH_FILES:
        p = os.path.join(HERE, name)
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8"):
            rec = json.loads(line)
            for m in rec.get("messages", []):
                texts.append((rec["id"], m["content"]))
    return texts

def anti_leak_check(fixtures, training_texts, min_words=6):
    problems = []
    for fx in fixtures:
        words = fx["user"].split()
        for i in range(0, max(1, len(words) - min_words + 1)):
            window = " ".join(words[i:i + min_words])
            for rid, content in training_texts:
                if window in content:
                    problems.append((fx["fixture_id"], rid, window[:60]))
                    break
    return problems

# ------------------------------------------------------------ self-checks ----
def sanity_selfcheck(fixtures, training_texts):
    failures = []
    for fx in fixtures:
        checker = CHECKERS[fx["fixture_id"]]
        try:
            if not checker(fx["gold_answer"], fx):
                failures.append((fx["fixture_id"], "gold did not pass"))
            if checker(fx["anti_answer"], fx):
                failures.append((fx["fixture_id"], "anti wrongly passed"))
        except Exception as exc:
            failures.append((fx["fixture_id"], "exception: %r" % exc))
    for fid, rid, w in anti_leak_check(fixtures, training_texts):
        failures.append((fid, "leak from " + rid + ": " + w))
    return failures

# ---------------------------------------------------------------- CLI --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default="eval_results.jsonl")
    ap.add_argument("--skip-selfcheck", action="store_true")
    args = ap.parse_args()

    fixtures = [fixture_planner(), fixture_importer(), fixture_stopper(),
                fixture_decimals(), fixture_urljoin(), fixture_decimal_mul(),
                fixture_repeat(), fixture_state()]
    training_texts = load_training_texts()

    if not args.skip_selfcheck:
        fails = sanity_selfcheck(fixtures, training_texts)
        if fails:
            for fid, why in fails:
                print("SELFCHECK FAIL", fid, "->", why)
            sys.exit("self-check failed; fix fixtures before scoring a model")
        print("self-check: gold passes, anti fails, no leakage (%d fixtures)" % len(fixtures))

    if not args.model:
        print("fixtures ready:", [fx["fixture_id"] for fx in fixtures])
        print("use --model '<command>' to capture outputs to", args.out)
        return

    results = []
    for fx in fixtures:
        try:
            proc = subprocess.run(args.model, input=fx["user"], capture_output=True,
                                  text=True, timeout=300, shell=True)
            answer = proc.stdout
        except subprocess.TimeoutExpired:
            answer = ""
        results.append({"fixture_id": fx["fixture_id"], "family": fx["family"],
                        "answer": answer, "scored": False})
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print("captured %d answers to %s (not scored; scoring is explicit)" % (len(results), args.out))

if __name__ == "__main__":
    main()
