# Original frozen selector: executable source admission (2026-10-30)

## Scope and authority

The original `chowder_batch.reap_select`, `reap_gates` and
`reap_v2_source_identity` implementations are **unchanged**. The new
`reap_v2_frozen_selector_executable_witness_20261030.py` is a CPU-only,
read-only admission check. It does not execute generated code, perform model
inference, access original `F:` execution journals, create signer credentials,
or approve release. The original frozen scientific gate definitions, serving
flags, Amendment 4 precedence and unresolved statistical-method issue remain
unchanged.

Previously, the evidence bridge checked independently approved input digests
but called the *currently imported* `reap_select.evaluate` without verifying
that the executing selector/gate functions still matched the original frozen
source. Source-authentic inputs alone cannot protect against a substituted
selector function or modified in-memory gate threshold.

## Original source pins

The witness hardcodes source-owned SHA-256 values for the original Git LF
source blobs (not derived from the current runtime or an evidence receipt).
Windows CRLF checkout bytes are normalized **only** at line endings before
hashing; any lone carriage return is refused:

| Module | SHA-256 |
| --- | --- |
| `chowder_batch/reap_select.py` | `eec6a908433f194b62aa111cbca239be43d22ef711e307c758ae2babf81d7502` |
| `chowder_batch/reap_gates.py` | `524961a0c75c41e7c2d5a0c8538569fdcb123272d0a875835c34cab6c91d428b` |
| `chowder_batch/reap_serve.py` | `d0306415fb3812380153b8b8a81be767ed9e068f9092c6542d067551be8ab4b0` |
| `chowder_batch/reap_v2_source_identity.py` | `c08146f94c5740f588f08fd3acc6cacad02ad8316c7e677b551501562a9c99c2` |
| Windows CPython 3.11.9 `Lib/statistics.py` | `99ea70b550d381eadc2099866206a716273eec54764e27c78daf7a66039588da` |
| Windows CPython 3.11.9 `Lib/json/__init__.py` | `feb17670e443e5db2723f217727dcc5d5e155c40e4e6935b16061c88542f24e7` |
| Windows CPython 3.11.9 `Lib/json/decoder.py` | `b719fbcfcebd2b174f076e71292e22b1a17d9e258dbe896c768325383bad4f80` |
| Windows CPython 3.11.9 `Lib/json/scanner.py` | `9841566fb17315ebdd40a1ca9cb214f02cde7171b187d4dc821c80120ea853c3` |
| Windows CPython 3.11.9 `Lib/json/encoder.py` | `7db65c95f09297d57cc6273d78858b25ffcba65bc83709c4a15584d1dbef3c38` |

The bridge additionally pins the SHA-256 of the witness's own source, rather
than accepting an arbitrary substituted witness module. All 46 original
top-level selector/gate/serving/calibration functions are checked: the witness
compiles the pinned source *without executing it*, compares the resulting
function code objects structurally to live module functions, checks their
globals/module origin and original literal defaults, and compares critical
runtime policy constants with source AST literals. It also verifies that
the selector's imported gate/calibration callables are the same live functions
from the pinned modules, not replacement aliases.

Independent review demonstrated that checking only the three project modules
was insufficient: replacing `statistics.stdev` in memory changed a 50-row
original Gate1 comparison from FAIL to PASS. The witness now also authenticates
the original Windows CPython 3.11.9 `statistics.py` source and 29 live
statistical functions and their defaults, plus the relevant built-in
numeric/hash/JSON bindings.
A different Python runtime is not silently substituted for this reviewed one.

Another independent adversarial review showed that replacing only
`json.loads.__code__` could make malformed JSON decode as forged evidence
while the original project source SHA checks still passed. The witness now
pins the four original JSON runtime sources and checks 22 live JSON functions
and class methods, including the decoder and encoder, plus the default
decoder's native scanner bindings and source-defined function defaults. A
separate adversarial test showed that mutating `JSONDecoder.__init__`'s
`parse_int` keyword default changed a source-authentic `0` into `200` without
changing its code object; this is now refused. Native offload parsing and its five regex
policies are also bound to original source.
The native default JSON scanner's parse-int/parse-float bindings are checked
separately, because a replacement C Scanner instance can preserve its type
while changing numeric decoding behavior.

The bridge independently checks the witness's *live function code* against
its pinned source, rather than trusting the witness's disk hash or self-report.
It checks the bridge's selector/gate aliases and witness policy maps. On
refusal it constructs a literal nested BLOCKED decision, never invoking an
already rejected selector function.

The bridge invokes the witness **before** reading selector input digests and
**after** frozen selector evaluation. Any mismatch returns the existing
`BLOCKED_UNVERIFIED_FROZEN_EVIDENCE` outcome with no chosen arm or release.

## Adversarial coverage and limits

CPU-only tests exercise foreign loaded-module paths, wrong source hashes,
replacement `evaluate`, `select`, `compute`, `_gate0`, `_speed`, source identity
and input digest functions, altered thresholds, altered frozen model/source
hash dictionaries, changed imported aliases, mutated function defaults,
statistics hijacks that flip a real Gate1 result, JSON decoder-code forgery,
native offload regex replacement, witness function replacement, and forged
nested SELECTED outcomes on a blocked path.
Synthetic evidence tests explicitly bypass the witness *inside their test-only
fixture*, because they deliberately monkeypatch original selector functions
and source policies. That bypass does not exist in production source.

This is a same-process consistency check, **not** a separately witnessed
execution attestation. An attacker with arbitrary execution in the verifier
process could replace the bridge itself or alter Python import machinery or
unreviewed runtime dependencies. External trusted process ownership, immutable
runtime deployment, independently approved original model/HTTP evidence and
source-pinned executable HumanEval grading remain mandatory. It cannot
retrospectively authenticate old native HTTP request/response bodies.

All production original-evidence approval maps remain empty. Neither the
previously completed research-only Q4 151/164 and Q8 152/164 HumanEval arms,
nor the cross-host GSM8K research comparison, are frozen Gate 1/Gate 2.

**Gen-0:** `BLOCKED`; `chosen: null`; `release_approved: false`.
