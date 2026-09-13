"""How a prompt was rendered — measured evidence, not a request.

Protocol v3 exists because two parents that tokenize identically can carry
different chat templates (the 2026-09-08 C/D tournament: a 17,232-token exact
match, yet different templates), so the *rendered prompt bytes* are protocol
identity: a suite scored under one rendering is not comparable with a suite
scored under another. The suite spec already recorded the *request*
(`use_chat_template`, `canonical_rendering`); what was missing is what the
worker actually rendered with.

Why one module for both workers
-------------------------------
`scoring.py` had to be extracted because the two workers' copies of the
scoring rule drifted and a whole GSM8K arm became uninterpretable. A rendering
drift is worse than a scoring drift: it changes the prompt bytes the model
sees, so the two arms are not answering the same question. Both text workers
therefore render through `render_prompt`, and the controller binds the
digest it reports.

What the digest is over
-----------------------
The template *source*, not a rendered prompt (the prompt varies per row; the
template decides the bytes for every row). For the canonical path it is the
digest of the pinned canonical template — which is why the canonical path
does not need the tokenizer to carry a template at all, and why the recorded
identity is identical across parents with different re-serialized templates.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

RENDERING_RAW = "raw"
RENDERING_TOKENIZER_TEMPLATE = "tokenizer-template"
RENDERING_CANONICAL_TEMPLATE = "canonical-template"

#: Every rendering a worker may report. Anything else is refused by the
#: controller rather than folded into a nearest match.
RENDERINGS: tuple[str, ...] = (
    RENDERING_RAW,
    RENDERING_TOKENIZER_TEMPLATE,
    RENDERING_CANONICAL_TEMPLATE,
)


def _template_bytes(template: Any) -> bytes:
    if isinstance(template, str):
        return template.encode("utf-8")
    # Newer transformers allow a mapping (or sequence) of named templates.
    # The content still decides the rendered bytes, so hash it canonically
    # rather than refusing a shape that legitimately exists.
    return json.dumps(
        template, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def tokenizer_template_sha256(tokenizer: Any) -> str | None:
    """Digest of the tokenizer's own chat template, or None when it has none."""
    template = getattr(tokenizer, "chat_template", None)
    if template is None:
        return None
    return hashlib.sha256(_template_bytes(template)).hexdigest()


def render_prompt(
    *,
    tokenizer: Any,
    prompt: str,
    suite_name: str,
    use_chat_template: bool,
    canonical_rendering: bool,
) -> tuple[str, dict[str, Any]]:
    """Render one prompt; return ``(text, evidence)``.

    `evidence` always names the rendering that was actually used and carries
    the template digest whenever a template was used, so a caller cannot
    record a rendering it did not perform.
    """
    if canonical_rendering and not use_chat_template:
        raise RuntimeError(
            f"suite {suite_name!r} sets canonical_rendering without "
            "use_chat_template; the request is contradictory and would render "
            "nothing canonical"
        )
    if not use_chat_template:
        return prompt, {"rendering": RENDERING_RAW}
    if canonical_rendering:
        # Local import keeps the embedded template out of the import path of
        # every evaluator that never renders canonically.
        from chowder.canonical_chat_template import (
            canonical_template_sha256,
            render_canonical,
        )

        rendered = render_canonical(tokenizer, prompt)
        return rendered, {
            "rendering": RENDERING_CANONICAL_TEMPLATE,
            "chat_template_sha256": canonical_template_sha256(),
        }
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError(
            f"suite {suite_name!r} requested chat template but tokenizer has no chat template"
        )
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return rendered, {
        "rendering": RENDERING_TOKENIZER_TEMPLATE,
        "chat_template_sha256": tokenizer_template_sha256(tokenizer),
    }


def validate_rendering_evidence(
    *,
    suite_name: str,
    reported: Mapping[str, Any],
    use_chat_template: bool,
    canonical_rendering: bool,
) -> dict[str, Any]:
    """Return the protocol-entry fragment for one suite, or refuse to score it.

    Fails closed on every way the rendering could be unverifiable: a missing or
    unknown `rendering`, a rendering the spec did not ask for (the silent
    fallback case), a template rendering without a digest, a digest on a raw
    suite, or a canonical claim whose digest is not the pinned canonical
    template. Both controllers call this, so the baseline and candidate arms
    cannot drift apart in what they accept.
    """
    rendering = reported.get("rendering")
    digest = reported.get("chat_template_sha256")
    expected = (
        RENDERING_CANONICAL_TEMPLATE
        if (use_chat_template and canonical_rendering)
        else RENDERING_TOKENIZER_TEMPLATE
        if use_chat_template
        else RENDERING_RAW
    )
    if rendering not in RENDERINGS:
        raise RuntimeError(
            f"suite {suite_name!r} has no valid rendering evidence "
            f"({rendering!r}); refusing to assume how its prompts were rendered"
        )
    if rendering != expected:
        raise RuntimeError(
            f"suite {suite_name!r} was rendered as {rendering!r} but the spec asked "
            f"for {expected!r}; the two arms would not be scoring the same prompt bytes"
        )
    if expected == RENDERING_RAW:
        if digest is not None:
            raise RuntimeError(
                f"suite {suite_name!r} reports a chat template digest for raw rendering"
            )
        return {"rendering": RENDERING_RAW}
    if not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError(
            f"suite {suite_name!r} reports {rendering!r} without a 64-hex template digest"
        )
    if expected == RENDERING_CANONICAL_TEMPLATE:
        from chowder.canonical_chat_template import (
            canonical_template_sha256,
            verify_canonical_template,
        )

        # Fail closed on an edited embedded template independently of the
        # worker: whatever a worker reports, the controller knows which
        # template the canonical path is supposed to be.
        verify_canonical_template()
        pinned = canonical_template_sha256()
        if digest != pinned:
            raise RuntimeError(
                f"suite {suite_name!r} claims canonical chat template rendering but "
                f"reports digest {digest[:12]}…, not the pinned canonical chat "
                f"template {pinned[:12]}…"
            )
    return {"rendering": rendering, "chat_template_sha256": digest}
