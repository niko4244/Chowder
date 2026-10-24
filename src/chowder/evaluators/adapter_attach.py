"""Bind adapter bytes immediately around PEFT attachment, not only at launch.

The controller and worker independently authenticate exported adapter bytes,
but loading a large parent can take a long time. Reverify immediately before
and after PEFT consumes the adapter to refuse a changed or substituted input.
This is a fail-closed check, not an atomic immutable-filesystem snapshot.
"""

from __future__ import annotations

from typing import Any, Callable

from .adapter_identity import verify_adapter_reload


def attach_verified_adapter(
    base: Any,
    *,
    adapter_dir: str,
    adapter_sha256: str,
    base_model: str,
    loader: Callable[[Any], Any],
    label: str,
) -> tuple[Any, str]:
    """Use the same attested adapter at the preattach and postattach fences."""
    verify_adapter_reload(
        adapter_dir, adapter_sha256, label=f"{label} before PEFT attach",
        base_model=base_model,
    )
    attached = loader(base)
    verified = verify_adapter_reload(
        adapter_dir, adapter_sha256, label=f"{label} after PEFT attach",
        base_model=base_model,
    )
    assert verified is not None  # an adapter directory was required
    return attached, verified
