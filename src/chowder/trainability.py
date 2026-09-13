"""Exact intended-component trainability: real autograd, not a count.

`target_coverage` answers "did a requested module *name* adapt anything" by leaf
name and returns counts. That is a useful summary and the plan's P5 says
explicitly that it is not the proof. It cannot see that a recipe asked for eight
`q_proj` paths while seven exist (the family is present, so the guard passes),
and it says nothing at all about whether the designated tensors received a
gradient or changed.

This module answers the stricter question, in the order the plan requires:

1. The exact expected parameter paths are resolved from the *loaded
   architecture* for the declared recipe, and compared as a **set** against what
   is actually present. Missing, extra, and unreadable paths each block
   qualification -- a total can hide a substitution, a set cannot.
2. Gradients are observed **after backward and before they are cleared**, over a
   declared window of steps. Every intended component must show at least one
   finite non-zero gradient and a measurable optimizer update *within that
   window*.
3. Frozen tensors are hashed before and after, so "frozen" is verified rather
   than assumed.

Failure modes are kept distinct because they mean different things and only some
are fixable by waiting: ``grad is None`` (not on the graph), exactly zero (on the
graph, no influence), non-finite (a broken run, never a reason to continue),
unreadable (an evidence gap, not a measured zero), and "no update" (the
optimizer never applied anything).

The window is deliberately *not* one step: a LoRA ``A`` factor can legitimately
receive zero gradient on step 1 while ``B`` is still zero, so the policy is
"within the declared window" -- fixed before the run, never renegotiated after
seeing results.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

#: Gradient observation vocabulary. `none`/`zero`/`nonfinite`/`unreadable` are
#: the four ways a component can fail to be trainable; `nonzero` is the healthy
#: observation. They are never collapsed into a boolean.
GRAD_NONE = "grad-none"
GRAD_ZERO = "grad-zero"
GRAD_NONFINITE = "grad-nonfinite"
GRAD_UNREADABLE = "grad-unreadable"
GRAD_NONZERO = "grad-nonzero"

GRADIENT_STATES: tuple[str, ...] = (
    GRAD_NONE,
    GRAD_ZERO,
    GRAD_NONFINITE,
    GRAD_UNREADABLE,
    GRAD_NONZERO,
)

#: Above this many elements a full-tensor digest is replaced by a stride-sampled
#: one, and the report says which was used. A sampled check is never reported as
#: equality.
_FULL_HASH_MAX_ELEMENTS = 1 << 20


class TrainabilityError(RuntimeError):
    """A declared trainable component is not demonstrably trainable."""


@dataclass(frozen=True)
class ComponentPathReport:
    """Exact-path comparison between the declared recipe and the loaded model.

    `require_exact` distinguishes the two questions this module answers. An
    explicitly declared *parameter* set (router-only mode) must match exactly:
    an extra trainable tensor is a scope violation. A declared list of PEFT
    *target modules* tolerates extras, because PEFT matches by suffix and a
    broader match is legitimate -- but a missing path is still a defect, and a
    count cannot see it.
    """

    expected: tuple[str, ...]
    actual: tuple[str, ...]
    missing: tuple[str, ...]
    extra: tuple[str, ...]
    unreadable: tuple[str, ...]
    unknown_suffixes: tuple[str, ...] = ()
    require_exact: bool = True

    @property
    def ok(self) -> bool:
        return not (
            self.missing
            or self.unreadable
            or self.unknown_suffixes
            or (self.require_exact and self.extra)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "require_exact": self.require_exact,
            "expected_count": len(self.expected),
            "actual_count": len(self.actual),
            "missing": list(self.missing),
            "extra": list(self.extra),
            "unreadable": list(self.unreadable),
            "unknown_suffixes": list(self.unknown_suffixes),
        }


def _parameter_items(model: Any) -> list[tuple[str, Any]]:
    try:
        return [(str(name), param) for name, param in model.named_parameters()]
    except Exception as exc:  # pragma: no cover - defensive
        raise TrainabilityError(f"could not enumerate model parameters: {exc}") from exc


def _module_names(model: Any) -> list[str]:
    try:
        return [str(name) for name, _ in model.named_modules()]
    except Exception as exc:  # pragma: no cover - defensive
        raise TrainabilityError(f"could not enumerate model modules: {exc}") from exc


def resolve_expected_module_paths(
    model: Any, target_modules: Sequence[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Exact **module** paths a declared PEFT target list designates.

    This is the pre-injection half of the identity: before anything is adapted or
    frozen, resolve every module the recipe names, from the loaded architecture.
    Returns ``(paths, unknown_targets)``; a target this architecture does not have
    is reported rather than ignored, because it names something that can never be
    adapted.
    """
    wanted = tuple(str(target) for target in target_modules)
    names = _module_names(model)
    paths: list[str] = []
    unknown: list[str] = []
    for target in wanted:
        matches = [name for name in names if name == target or name.endswith("." + target)]
        if not matches:
            unknown.append(target)
        paths.extend(matches)
    return tuple(dict.fromkeys(paths)), tuple(unknown)


def resolve_expected_parameter_paths(
    model: Any, parameter_suffixes: Sequence[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Exact **parameter** paths a declared suffix list designates.

    Used for a parameter-level recipe such as router-only training, where the
    intent is a specific tensor (``mlp.gate.weight``), not a module that an
    adapter will be attached to.
    """
    wanted = tuple(str(suffix) for suffix in parameter_suffixes)
    names = [name for name, _ in _parameter_items(model)]
    paths: list[str] = []
    unknown: list[str] = []
    for suffix in wanted:
        matches = [name for name in names if name == suffix or name.endswith("." + suffix)]
        if not matches:
            unknown.append(suffix)
        paths.extend(matches)
    return tuple(dict.fromkeys(paths)), tuple(unknown)


def adapted_module_paths(model: Any, marker: str = ".lora_A") -> tuple[str, ...]:
    """The full paths of adapted modules on a live PEFT model.

    `target_coverage.adapted_modules_by_leaf` collapses these to leaf counts; this
    keeps the paths so the comparison can name which one is missing.
    """
    targets: set[str] = set()
    for name in _module_names(model):
        index = name.find(marker)
        if index > 0:
            targets.add(name[:index])
    return tuple(sorted(targets))


def component_path_report(
    expected: Iterable[str],
    actual: Iterable[str],
    *,
    unreadable: Iterable[str] = (),
    unknown_suffixes: Iterable[str] = (),
    require_exact: bool = True,
) -> ComponentPathReport:
    """Compare intended paths against present paths as exact sets."""
    expected_set = {str(name) for name in expected}
    actual_set = {str(name) for name in actual}
    return ComponentPathReport(
        expected=tuple(sorted(expected_set)),
        actual=tuple(sorted(actual_set)),
        missing=tuple(sorted(expected_set - actual_set)),
        extra=tuple(sorted(actual_set - expected_set)),
        unreadable=tuple(sorted({str(name) for name in unreadable})),
        unknown_suffixes=tuple(sorted({str(name) for name in unknown_suffixes})),
        require_exact=bool(require_exact),
    )


def assert_components_qualified(report: ComponentPathReport) -> ComponentPathReport:
    """Refuse when the intended component set is not exactly what is present."""
    if report.ok:
        return report
    problems: list[str] = []
    if report.missing:
        problems.append(
            f"missing intended parameter paths ({len(report.missing)}): "
            + ", ".join(report.missing[:10])
        )
    if report.extra and report.require_exact:
        problems.append(
            f"unexpected extra parameter paths ({len(report.extra)}): "
            + ", ".join(report.extra[:10])
        )
    if report.unreadable:
        problems.append(
            "unreadable intended parameter paths: " + ", ".join(report.unreadable[:10])
        )
    if report.unknown_suffixes:
        problems.append(
            "recipe suffixes this architecture does not have: "
            + ", ".join(report.unknown_suffixes)
        )
    raise TrainabilityError(
        "the intended trainable component set is not what the loaded model "
        "presents, so training would not adapt what the recipe declared:\n  "
        + "\n  ".join(problems)
        + "\nCounts are not proof: the same total can hide a missing path "
        "replaced by another."
    )


def _tensor_digest(tensor: Any) -> dict[str, Any]:
    """Digest of tensor values, with the strategy recorded.

    Full sha256 for small tensors; a stride-sampled digest above the element
    threshold, reported as `sampled` so a sampled check can never be mistaken for
    complete equality.
    """
    import torch  # local: this module must import cheaply without torch

    detached = tensor.detach()
    flat = detached.reshape(-1).to(torch.float32).contiguous()
    elements = int(flat.numel())
    if elements <= _FULL_HASH_MAX_ELEMENTS:
        digest = hashlib.sha256(flat.numpy().tobytes()).hexdigest()
        return {"digest": digest, "strategy": "full", "elements": elements}
    # Ceiling division, not floor: with floor, a tensor just above the threshold
    # gets stride 1 and the "sampled" digest covers every element -- a label that
    # claims sampling while doing the full read, which is worse than either.
    stride = max(1, -(-elements // _FULL_HASH_MAX_ELEMENTS))
    sampled = flat[::stride]
    digest = hashlib.sha256(sampled.numpy().tobytes()).hexdigest()
    return {
        "digest": digest,
        "strategy": "sampled",
        "elements": elements,
        "stride": stride,
        "sampled_elements": int(sampled.numel()),
    }


def hash_parameters(parameters: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Digest a set of named tensors for before/after comparison."""
    return {str(name): _tensor_digest(param) for name, param in parameters.items()}


def _classify_gradient(param: Any) -> str:
    import torch

    try:
        grad = param.grad
    except Exception:
        return GRAD_UNREADABLE
    if grad is None:
        return GRAD_NONE
    try:
        detached = grad.detach()
        if not bool(torch.isfinite(detached).all().item()):
            return GRAD_NONFINITE
        nonzero = bool(torch.count_nonzero(detached).item())
    except Exception:
        return GRAD_UNREADABLE
    return GRAD_NONZERO if nonzero else GRAD_ZERO


@dataclass
class _ComponentObservation:
    name: str
    gradient_states: list[str] = field(default_factory=list)
    nonzero_steps: list[int] = field(default_factory=list)
    update_steps: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        states = self.gradient_states
        return {
            "observed_steps": len(states),
            "gradient_states": sorted(set(states)),
            "nonzero_steps": list(self.nonzero_steps),
            "update_steps": list(self.update_steps),
            "trainable": bool(self.nonzero_steps and self.update_steps),
        }


class TrainabilityProbe:
    """Observe real gradients and real updates for a declared component set.

    Usage, and the ordering matters::

        probe = TrainabilityProbe(model, trainable_names, window_steps=4)
        for step in ...:
            loss.backward()
            probe.record_gradients(step)      # after backward, before zeroing
            optimizer.step()
            probe.record_update(step)
            optimizer.zero_grad()
        probe.assert_qualified()

    Frozen tensors are hashed at construction and re-hashed by
    `frozen_report()`; `assert_frozen_unchanged()` refuses when one moved.
    """

    def __init__(
        self,
        model: Any,
        trainable_names: Iterable[str],
        *,
        window_steps: int,
        frozen_names: Iterable[str] | None = None,
    ) -> None:
        if isinstance(window_steps, bool) or not isinstance(window_steps, int):
            raise TypeError("window_steps must be an int")
        if window_steps <= 0:
            raise ValueError("window_steps must be positive: a zero-step window proves nothing")
        self._model = model
        self._names = tuple(str(name) for name in trainable_names)
        if not self._names:
            raise ValueError("a trainability probe needs at least one intended component")
        self.window_steps = window_steps
        params = dict((str(name), param) for name, param in _parameter_items(model))
        missing = [name for name in self._names if name not in params]
        if missing:
            raise TrainabilityError(
                "cannot probe parameters the model does not have: " + ", ".join(missing[:10])
            )
        self._params = params
        if frozen_names is None:
            trainable = set(self._names)
            frozen_names = [name for name in params if name not in trainable]
        self._frozen_names = tuple(str(name) for name in frozen_names)
        self._frozen_before = hash_parameters({n: params[n] for n in self._frozen_names})
        self._observed = {name: _ComponentObservation(name) for name in self._names}
        #: Value sample taken by `record_gradients` (i.e. before the step) so
        #: `record_update` can measure a real change rather than assume one.
        self._update_before: dict[str, dict[str, Any]] = {}

    # -- observation -------------------------------------------------------

    def record_gradients(self, step: int) -> None:
        """Classify each intended component's gradient for one step.

        Called after `loss.backward()` and before the optimizer step, so the
        parameter values are also sampled here for the update comparison.
        """
        self._update_before = {}
        for name in self._names:
            try:
                self._update_before[name] = _tensor_digest(self._params[name])
            except Exception:
                # A parameter whose values cannot be read is already an evidence
                # gap: the gradient classification below records why it cannot be
                # certified, and no update is measurable for it. Crashing here
                # instead would hide that reason behind a traceback.
                self._update_before[name] = None
        for name in self._names:
            state = _classify_gradient(self._params[name])
            observation = self._observed[name]
            observation.gradient_states.append(state)
            if state == GRAD_NONZERO:
                observation.nonzero_steps.append(int(step))

    def record_update(self, step: int) -> None:
        """Measure whether each intended component's values actually changed.

        Called after `optimizer.step()`. A component whose sample is identical
        before and after is not updated, whatever its gradient looked like.
        """
        for name in self._names:
            before = self._update_before.get(name)
            if before is None:
                continue
            after = _tensor_digest(self._params[name])
            if after["digest"] != before["digest"]:
                self._observed[name].update_steps.append(int(step))

    # -- verdict -----------------------------------------------------------

    def report(self) -> dict[str, Any]:
        components = {name: obs.to_dict() for name, obs in self._observed.items()}
        blocking: dict[str, list[str]] = {}
        for name, entry in components.items():
            reasons: list[str] = []
            states = entry["gradient_states"]
            if GRAD_UNREADABLE in states:
                reasons.append(
                    "gradient could not be read (unreadable is an evidence gap, not a zero)"
                )
            if GRAD_NONFINITE in states:
                reasons.append("a non-finite gradient was observed; the run is not healthy")
            if not entry["nonzero_steps"]:
                if GRAD_NONE in states and len(states) == 1:
                    reasons.append(
                        "no gradient ever attached to this parameter (grad is None): it is "
                        "not on the autograd graph for this loss"
                    )
                elif states == [GRAD_ZERO]:
                    reasons.append(
                        "every observed gradient was exactly zero: the parameter is on the "
                        "graph but does not influence the loss"
                    )
                else:
                    reasons.append(
                        "no finite non-zero gradient within the declared probe window"
                    )
            if not entry["update_steps"]:
                reasons.append("no measurable optimizer update within the probe window")
            if reasons:
                blocking[name] = reasons
        return {
            "window_steps": self.window_steps,
            "intended_components": list(self._names),
            "components": components,
            "not_trainable": blocking,
            "ok": not blocking,
        }

    def assert_qualified(self) -> dict[str, Any]:
        report = self.report()
        if report["ok"]:
            return report
        detail = "\n  ".join(
            f"{name}: " + "; ".join(reasons)
            for name, reasons in report["not_trainable"].items()
        )
        raise TrainabilityError(
            "these intended trainable components did not demonstrate real training "
            f"within {self.window_steps} observed step(s):\n  {detail}\n"
            "requires_grad and a parameter count are not evidence; a finite non-zero "
            "gradient plus a measurable update is. Do not widen the window after "
            "seeing this result -- fix the recipe or narrow the declared intent."
        )

    # -- frozen verification ----------------------------------------------

    def frozen_report(self) -> dict[str, Any]:
        after = hash_parameters({n: self._params[n] for n in self._frozen_names})
        changed: dict[str, dict[str, Any]] = {}
        strategies: set[str] = set()
        for name, record in after.items():
            strategies.add(str(record["strategy"]))
            if record["digest"] != self._frozen_before[name]["digest"]:
                changed[name] = {"before": self._frozen_before[name], "after": record}
        return {
            "frozen_parameters": len(self._frozen_names),
            "digest_strategy": sorted(strategies) or ["none"],
            "changed": changed,
            "ok": not changed,
        }

    def assert_frozen_unchanged(self) -> dict[str, Any]:
        report = self.frozen_report()
        if report["ok"]:
            return report
        names = ", ".join(sorted(report["changed"])[:10])
        raise TrainabilityError(
            f"frozen parameters changed during training: {names}. The freeze policy "
            f"was not honoured (digest strategy: {report['digest_strategy']})."
        )


def assert_router_only_scope(
    trainable_names: Iterable[str], model: Any, *, router_suffix: str = "mlp.gate.weight"
) -> dict[str, Any]:
    """Router-only mode must train exactly one router per intended layer.

    Shared experts and their gates are explicitly frozen for this milestone: the
    conversion leaves the shared expert all-zero, and a zero shared expert
    multiplied by its gate has exactly zero derivative with respect to that gate,
    so designating it trainable would consume optimizer state for something that
    can never learn. Expert-row utilisation is a separate diagnostic and is not
    evidence of per-tensor reachability.
    """
    names = [str(name) for name in trainable_names]
    gates = [name for name in names if name.endswith(router_suffix)]
    not_gates = [name for name in names if not name.endswith(router_suffix)]
    if not gates:
        raise TrainabilityError(
            f"router-only mode designated no {router_suffix!r} parameter; there is "
            "nothing for the router to learn"
        )
    if not_gates:
        raise TrainabilityError(
            "router-only mode must train exactly the router gates, but these "
            f"parameters were designated trainable as well: {sorted(not_gates)[:10]}. "
            "Shared experts and their gates stay frozen for this milestone."
        )
    duplicated = sorted({name for name in gates if gates.count(name) > 1})
    if duplicated:
        raise TrainabilityError(f"duplicate router parameters in the intended set: {duplicated}")
    architecture_gates, unknown = resolve_expected_parameter_paths(model, [router_suffix])
    report = component_path_report(architecture_gates, gates, unknown_suffixes=unknown)
    if not report.ok:
        # e.g. eight layers have a router and the recipe designated seven: a
        # count would have said "gates present", the set says which is missing
        assert_components_qualified(report)
    return {
        "router_parameters": sorted(gates),
        "router_count": len(gates),
        "architecture_router_count": len(architecture_gates),
        "unknown_suffix": list(unknown),
        "ok": True,
    }


def utilization_by_expert(
    expert_rows: Mapping[str, Any] | None, *, threshold: float = 0.0
) -> dict[str, Any]:
    """Expert-row utilisation, kept separate from per-tensor reachability.

    A routing table can look healthy (every token routed, top-1 spread across
    experts) while a tensor it feeds is unreachable, and vice versa: a fully dead
    router still produces a table. This is a diagnostic, never a substitute for
    the gradient evidence above.
    """
    if expert_rows is None:
        return {"status": "not_reported", "note": "no router statistics were provided"}
    rows: dict[str, Any] = {}
    dead: list[str] = []
    for layer, counts in expert_rows.items():
        values = [float(value) for value in counts]
        total = sum(values)
        share = [value / total for value in values] if total > 0 else [0.0 for _ in values]
        rows[str(layer)] = {"counts": values, "share": share}
        if total > 0 and any(s <= threshold for s in share):
            dead.append(str(layer))
    return {
        "status": "measured",
        "layers": rows,
        "layers_with_unused_experts": sorted(dead),
        "note": "utilisation is a diagnostic; it is not evidence of per-tensor reachability",
    }


def canonical_report(report: Mapping[str, Any]) -> str:
    """Stable JSON for embedding a probe/report in run evidence."""
    return json.dumps(dict(report), sort_keys=True, separators=(",", ":"), default=str)
