# REAP serving and evaluation destination binding

The frozen REAP protocol uses a single llama-server for each separately
identified arm. `chowder_batch.reap_serve` launches that server; its downstream
`reap_eval` command independently accepts `--port`, `--arm`, and `--out`.
Before this binding check, those arguments were not compared. In particular,
`reap_serve --port 8790 ... -- --port 8767 ...` could launch the intended
server and score an unrelated model at port 8767 while attributing the result
to the intended arm. The CLI must not use model-ID or output-directory labels
as proof that actual generation came from the launched server.

Before spawning a process, the wrapper now checks that the evaluator's port,
arm and canonical output directory equal the server invocation. Each identity
option must occur exactly once, including `--option=value` forms. Other
evaluator options are parsed against the supported CLI. The wrapper also
checks that the IPv4 loopback port is not already bound. This prevents a
preexisting unrelated service from satisfying the wrapper's health probe.

This is an **admission guard**, not complete process- or weight-byte
attestation. A listener could still race the port between its availability
check and child startup; the health endpoint itself does not cryptographically
prove the child PID or loaded GGUF bytes. Continue to require independently
bound model outputs and the registered full evaluation, fit/speed and
scientific gates. It does not change the frozen model flags, gates or judges.

CPU adversarial regression: `tests/test_reap_serve.py`, including mismatch,
duplicate-target and occupied-port refusal before any model process starts.
