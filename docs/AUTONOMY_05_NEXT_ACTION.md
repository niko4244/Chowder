# The one next action

**Wire the two gates into the runner's advance/promote path — one commit:
settlement-refused attempts stop advancing, and the promotion path
consults the preregistered retention profile + tier isolation before
anything promotes.**

Why this one, of everything: it converts the two library-proven invariants
most likely to be bypassed by convenience (the settlement hole found in the
#204 audit, and the retention/isolation gates) into structural rules the
runner cannot forget, and it is small, testable, and unblocks the campaign
design's launch conditions. Everything else — Kaggle R1–R8, ladder
escalation, EI backtests — is additive; this one is the difference between
"gates exist" and "gates hold".
