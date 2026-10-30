# Frozen selector: runtime dependency integrity (2026-10-30)

**Disposition:** CPU-only, source-bound admission engineering. **Research only.**
No original F: evidence, model weights, serving process, frozen scientific
policy, approval map, or completed HumanEval journal is changed. Gen-0 remains
`BLOCKED`, `chosen: null`, `release_approved: false`.

## Reproduced vulnerabilities before this correction

The existing original selector source witness compared Python function bytecode
and selected literals, but several *executing dependencies* remained mutable.
Independent CPU-only adversarial reviews demonstrated these genuine reversals
without modifying frozen model responses or scored-row files:

1. For 1,319 paired GSM8K rows with 31 candidate misses, the original
   `reap_gates._lb_gate` **FAIL** became **PASS** when the mutable
   `statistics._sqrt_bit_width` was set to zero. The previous witness accepted
   the altered statistical runtime. Replacing `statistics.Fraction` with a
   zero-producing factory had the same effect. In-place replacement of
   `fractions.Fraction.__new__`, or both dynamically generated addition
   operators, likewise undermined exact sample variance.
2. A source-identical `FunctionType` can capture a forged `__builtins__`
   dictionary containing `sum=lambda ...: 0`. Restoring its module globals
   does **not** restore the function's captured builtins. Replacing
   `reap_gates._rates` this way reversed an actual Gate0 **FAIL** to **PASS**;
   bytecode, globals identity and defaults still matched.
3. Mutating `json.decoder._CONSTANTS['NaN']` to `1.0` made the original JSON
   parser return a finite Gate0 trigram ratio from a literal `NaN`, while its
   verified function code and bound decoder identity were unchanged.
4. A monkeypatched `ast.literal_eval` that interpreted the original `-0.03`
   Gate1 threshold as `-1.0` let a matching runtime threshold mutation evade
   the original source-policy comparison.
5. Replacing only `reap_gates.Path` with a subclass that altered
   `read_text()` for candidate rows changed Gate1 **FAIL** to **PASS** while
   the original file bytes remained unchanged. The independently imported
   selector path reader still hashed the unaltered bytes. A coordinated swap
   of the witness and selector `Path` aliases also evaded the earlier
   same-module alias check.

## Scope of the correction

The unchanged frozen selector is still evaluated through
`reap_v2_frozen_gate_evidence_bridge_20261027.py`. Its independently pinned
`reap_v2_frozen_selector_executable_witness_20261030.py` now additionally:

- Requires the original CPython 3.11.9 `statistics._sqrt_bit_width == 109`,
  independently checks the expected binary64 mantissa/radix, and verifies
  statistical imported aliases and critical builtins.
- Source-pins `fractions.py` and verifies the original `Fraction` constructor,
  exact arithmetic primitives, and dynamically generated arithmetic methods
  **including their closure bindings**.
- Requires each checked selector, statistics, JSON and witness function to
  capture the original builtins dictionary, not an independently forged copy.
- Checks the exact nonfinite JSON constant mapping and independently pins
  the original AST parser and literal-evaluator functions before interpreting
  source policy.
- Requires all four frozen selector modules to retain the original `Path`
  class, and requires the witness's own `Path` binding to match the bridge's
  separately imported `pathlib.Path`.

The bridge's frozen witness SHA-256 is updated to the reviewed exact Git-LF
source bytes. Synthetic adversarial tests cover each demonstrated mutation,
positive original-runtime verification, and coordinated alias substitutions.

## Scientific limitations

These checks establish **same-process source/runtime consistency only**. They
do not provide external process-instance attestation, immutable interpreter
memory, native HTTP request-to-response provenance, independent timestamp
authority, a genuine same-session dense control, or a valid frozen Gate1/Gate2
comparison. They cannot establish that an attacker with arbitrary ability to
rewrite the bridge itself is excluded. Other prospective signed native-capture
limitations, including independent usage-token-budget and timing verification,
remain separate pre-admission requirements.

The original HumanEval Q4 151/164 and Q8 152/164 results and the original
cross-certified paired artifact remain research-only and unchanged. Historical
guard.v3 AutomationBench source-locked accounting remains 88/600 scored with
three strict passes; it is not a frozen four-slot qualification.
