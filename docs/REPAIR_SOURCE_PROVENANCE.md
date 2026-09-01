# Independent repair source provenance

A clean contamination audit proves that proposed repair examples do not exactly overlap the protected holdout index. It does **not** by itself prove where those examples came from.

Chowder's autonomous repair path therefore requires a source provenance manifest.

## Manifest contents

For each independent source, the manifest records:

- stable source ID;
- source reference/locator;
- declared source-content SHA-256;
- source kind.

For each repair example, it records:

- stable example ID;
- source ID;
- normalized prompt SHA-256;
- normalized prompt/expected-pair SHA-256.

The manifest also binds:

- repair dataset SHA-256;
- repair-example index SHA-256;
- holdout-index SHA-256 values used by the contamination audit.

## Autonomous candidate rule

`build_autonomous_repair_population()` refuses a repair dataset without a verified source manifest. Candidate identity includes the source-manifest SHA-256, so identical examples attributed to different source material become different experiment branches.

This provides the chain:

```text
source material
    ↓ content hash
source manifest
    ↓ example lineage
repair examples
    ↓ contamination audit
verified repair dataset
    ↓ deterministic identity
repair candidate
    ↓ same-protocol evaluation
promotion / rejection
```

## What this proves

The manifest proves internal consistency: the candidate, dataset, audited examples, holdout indexes, and declared source records cannot be silently swapped without changing hashes or failing verification.

For local files or trusted source adapters, Chowder should compute `content_sha256` directly from the source bytes. For remote/abstract source references, the current core treats the supplied content hash as a provenance assertion; a future source adapter must verify the remote bytes or signed content before assigning a trusted provenance level.
