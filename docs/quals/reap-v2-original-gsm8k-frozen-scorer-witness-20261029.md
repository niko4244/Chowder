# Original GSM8K scorer source-binding witness (research only)

The original frozen `reap_eval._score` uses `terminated and
score(text, expected, "final_number_match") == 1.0`. The historical v2
bridge's contributor-owned GSM8K regrader currently imports `score` from the
ambient Python environment. On this Windows host the ambient import resolves
to `C:\Users\nikma\Chowder\src\chowder\evaluators\scoring.py`, **outside**
the authorized worktree, with SHA-256
`c79a34d4a0c20a4a64438ff80133eabdfc4bde5f5a42c2a445684b70eacd579b`.
The reviewed worktree scorer is SHA-256
`d2e59aca5c0a621bac8d4979d522c9b6107f811610567d21c92a2b53960860eb`.

The newly added `chowder_batch.reap_original_gsm8k_frozen_scorer_20261029`
reproduces the exact original GSM8K final-number rule independently of that
ambient import, and **fails closed** if the worktree's original scorer bytes
do not match the pinned SHA-256. The tests show that monkeypatching the
ambient scorer to return success cannot make the independent witness approve
an incorrect prediction. Source drift, symlinks, wrong types, unclosed
thinking, numeric commas, and termination are also tested.

This standalone witness is **not yet wired into** the contributor-owned dirty
frozen-gate evidence bridge; that integration requires coordinated review
without overwriting concurrent work. It does not authenticate historical
native prompt-to-response lineage, approve an original Gate 1/2 result,
alter the frozen selector, or make a release decision. The production source
approval maps remain empty. Gen-0 remains
`{status: BLOCKED, chosen: null, release_approved: false}`.
