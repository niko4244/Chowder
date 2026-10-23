# Frozen REAP evaluation: model- and source-bound resumability

Before this change, `reap_eval.py` used an existing `<suite>.jsonl` file
as the sole authority for skipped item IDs. Running the same `--arm` folder
against another server, a different model, different prompts/expected answers,
or a changed decoding policy would silently retain previous answers. The
per-file SHA stored in `summary.json` did not attest the model or test input
identity, so a later full gate could misattribute mixed results.

New evaluations require `<arm>/serve.json` from the existing registered
`reap_serve.py` wrapper. That wrapper already records the GGUF's full
SHA-256, launch path and registered serving flags; the evaluator **reuses**
this saved digest and does not rehash multi-GB GGUF weights. Before the first
generation, each suite stores an exclusive-create
`<suite>.inputs.json` witness that binds the model SHA/path/flags,
the entire test population's rendered prompts and scoring inputs,
the frozen decoding settings, and suite name. A limited first pass and
subsequent full pass retain the same full-population identity.

On a resume, the witness must match exactly. A legacy result file without
its original witness **cannot** acquire a retrospective identity; start a
new output directory without rewriting old evidence. Existing results with
unknown or duplicate IDs, malformed JSON, or a changed completed summary
SHA are refused before new requests. The evaluator rechecks model/input
identity before finalizing each suite.

This is a local chain-of-custody precondition, not an independent attestation
of a server's live PID or the GGUF bytes it has mapped. Someone able to forge
both model witness and result files is outside the established provenance
boundary. Registered Gate D0, Gate 0/1/2, controls, speed/fit, paired
statistical method and human promotion requirements remain unchanged.
Historical results remain readable and unchanged; this change does not
retroactively certify them. Neither GPU nor model generation is required
to exercise the adversarial CPU tests:

```powershell
$env:PYTHONPATH='src;tests'
$env:CHOWDER_NO_GPU='1'
python -B -m pytest -p no:cacheprovider -q tests/test_reap_eval_resume_identity.py
```
