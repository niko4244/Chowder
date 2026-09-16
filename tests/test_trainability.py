"""P5: exact component identity plus real forward/backward/update evidence.

Two different failure modes are covered here, and they are different questions:

* **identity** -- did the recipe's intended component *set* exist on the loaded
  model? `target_coverage` answers by leaf-name counts, which the plan says is a
  summary and not proof: seven of eight expected `q_proj` paths still "has
  q_proj". The set comparison catches that. (The cross-check against the old
  guard lives in `test_target_coverage.py`.)
* **trainability** -- did those components actually receive a finite non-zero
  gradient and then change? `requires_grad` is a declaration, not evidence.

Everything here runs real autograd on real `torch.nn` modules (CPU, no model
download). The gradient states are asserted individually because `grad is None`,
exactly zero, non-finite, unreadable, and "no update" need different fixes.
"""

from __future__ import annotations

import pytest

# Same convention as the other torch-dependent suites (test_router_healing,
# test_activation_census): the plain CI jobs have no torch, and a bare import
# would turn every one of them into a collection error rather than a skip.
# The real-autograd tests run in the job that has the ML stack installed.
torch = pytest.importorskip("torch")

from chowder.trainability import (
    GRAD_NONE,
    GRAD_NONFINITE,
    GRAD_NONZERO,
    GRAD_UNREADABLE,
    GRAD_ZERO,
    _FULL_HASH_MAX_ELEMENTS,
    _sample_stride,
    _tensor_digest,
    TrainabilityError,
    TrainabilityProbe,
    adapted_module_paths,
    assert_components_qualified,
    assert_router_only_scope,
    component_path_report,
    hash_parameters,
    resolve_expected_module_paths,
    resolve_expected_parameter_paths,
    utilization_by_expert,
)


# --------------------------------------------------------------------------
# identity: the intended set, not a count
# --------------------------------------------------------------------------


class _Model(torch.nn.Module):
    def __init__(self, leaves=("q_proj", "v_proj"), layers=8):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {leaf: torch.nn.Linear(4, 4, bias=False) for leaf in leaves}
                )
                for _ in range(layers)
            ]
        )


def test_expected_module_paths_are_resolved_exactly_from_the_architecture():
    paths, unknown = resolve_expected_module_paths(_Model(), ["q_proj", "v_proj"])
    assert unknown == ()
    assert len(paths) == 16
    assert "layers.0.q_proj" in paths and "layers.7.v_proj" in paths


def test_a_target_the_architecture_does_not_have_is_reported_not_ignored():
    _, unknown = resolve_expected_module_paths(_Model(), ["q_proj", "in_proj_qkv"])
    assert unknown == ("in_proj_qkv",)
    with pytest.raises(TrainabilityError, match="does not have"):
        assert_components_qualified(
            component_path_report(["layers.0.q_proj"], ["layers.0.q_proj"], unknown_suffixes=["in_proj_qkv"])
        )


def test_one_missing_path_out_of_eight_is_a_refusal():
    """The plan's exact regression: seven of eight `q_proj` paths are adapted. A
    family- or count-based guard sees 'q_proj present'; the set names the hole.
    The cross-check against the old leaf-count guard lives in
    `test_target_coverage.py`."""
    model = _Model(leaves=("q_proj",))
    expected, _ = resolve_expected_module_paths(model, ["q_proj"])
    adapted = [name for name in expected if name != "layers.3.q_proj"]
    report = component_path_report(expected, adapted, require_exact=False)
    assert report.missing == ("layers.3.q_proj",)
    assert report.extra == ()
    with pytest.raises(TrainabilityError, match="missing intended parameter paths"):
        assert_components_qualified(report)


def test_an_extra_path_is_refused_when_exactness_is_required():
    """Router-only mode declares parameter paths exactly, so a substitution that
    keeps the total constant must not pass."""
    report = component_path_report(["a.q_proj.weight"], ["a.k_proj.weight"])
    assert report.missing == ("a.q_proj.weight",)
    assert report.extra == ("a.k_proj.weight",)
    with pytest.raises(TrainabilityError, match="extra parameter paths"):
        assert_components_qualified(report)


def test_a_broader_match_is_allowed_for_declared_target_modules():
    """PEFT matches by suffix; adapting more than the explicit list is legitimate
    (a preset may be broader). It is recorded, and the missing path still fails."""
    report = component_path_report(
        ["layers.0.q_proj"], ["layers.0.q_proj", "layers.0.k_proj"], require_exact=False
    )
    assert report.ok is True
    assert report.extra == ("layers.0.k_proj",)


def test_unreadable_paths_are_an_evidence_gap_not_a_pass():
    report = component_path_report(["a"], ["a"], unreadable=["a"])
    assert not report.ok
    with pytest.raises(TrainabilityError, match="unreadable"):
        assert_components_qualified(report)


def test_adapted_module_paths_keep_the_paths_not_just_leaf_counts():
    class _Peft(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_model = torch.nn.Module()
            self.base_model.model = torch.nn.Module()
            self.base_model.model.layers = torch.nn.ModuleList()
            for _ in range(2):
                block = torch.nn.Module()
                block.self_attn = torch.nn.Module()
                block.self_attn.q_proj = torch.nn.Module()
                block.self_attn.q_proj.lora_A = torch.nn.Linear(2, 2, bias=False)
                block.self_attn.v_proj = torch.nn.Module()
                self.base_model.model.layers.append(block)

    paths = adapted_module_paths(_Peft())
    assert paths == (
        "base_model.model.layers.0.self_attn.q_proj",
        "base_model.model.layers.1.self_attn.q_proj",
    )
    # the un-adapted sibling is absent: a count would hide which one it was


# --------------------------------------------------------------------------
# trainability: real gradients and real updates
# --------------------------------------------------------------------------


class _Tiny(torch.nn.Module):
    """A real two-linear module whose paths a recipe can designate."""

    def __init__(self):
        super().__init__()
        self.first = torch.nn.Linear(4, 4, bias=False)   # receives gradient
        self.second = torch.nn.Linear(4, 4, bias=False)  # detached: grad is None
        self.third = torch.nn.Linear(4, 4, bias=False)   # multiplied by zero: grad zero
        self.frozen = torch.nn.Linear(4, 4, bias=False)


def _trainable_names(model, *leaves):
    return [
        name
        for name, _ in model.named_parameters()
        if any(name.startswith(leaf) for leaf in leaves)
    ]


def _one_step(model, probe, optimizer, *, poison=None, update=True):
    x = torch.randn(2, 4)
    out = model.first(x).sum()
    # `second` is deliberately left off the graph; `third` contributes zero
    out = out + 0.0 * model.third(x).sum()
    optimizer.zero_grad()
    out.backward()
    if poison is not None:
        poison(model)
    probe.record_gradients(0)
    if update:
        optimizer.step()
    probe.record_update(0)


def _probe(model, names, *, window_steps=1):
    return TrainabilityProbe(model, names, window_steps=window_steps, frozen_names=["frozen.weight"])


def test_a_real_component_demonstrates_nonzero_gradient_and_a_measurable_update():
    torch.manual_seed(0)
    model = _Tiny()
    names = _trainable_names(model, "first")
    probe = _probe(model, names)
    _one_step(model, probe, torch.optim.SGD(model.parameters(), lr=0.1))
    report = probe.assert_qualified()
    entry = report["components"]["first.weight"]
    assert GRAD_NONZERO in entry["gradient_states"]
    assert entry["nonzero_steps"] == [0]
    assert entry["update_steps"] == [0]


def test_a_parameter_off_the_autograd_graph_reports_grad_none_not_zero():
    torch.manual_seed(0)
    model = _Tiny()
    names = _trainable_names(model, "second")
    probe = _probe(model, names)
    _one_step(model, probe, torch.optim.SGD(model.parameters(), lr=0.1))
    report = probe.report()
    assert report["ok"] is False
    assert report["components"]["second.weight"]["gradient_states"] == [GRAD_NONE]
    with pytest.raises(TrainabilityError, match="grad is None"):
        probe.assert_qualified()


def test_a_zero_gradient_is_reported_as_zero_and_is_not_fixable_by_waiting():
    torch.manual_seed(0)
    model = _Tiny()
    names = _trainable_names(model, "third")
    probe = _probe(model, names)
    _one_step(model, probe, torch.optim.SGD(model.parameters(), lr=0.1))
    report = probe.report()
    assert report["components"]["third.weight"]["gradient_states"] == [GRAD_ZERO]
    with pytest.raises(TrainabilityError, match="exactly zero"):
        probe.assert_qualified()


def test_a_nonfinite_gradient_is_refused_as_a_broken_run():
    torch.manual_seed(0)
    model = _Tiny()
    names = _trainable_names(model, "first")
    probe = _probe(model, names)

    def poison(m):
        m.first.weight.grad[0, 0] = float("nan")

    _one_step(model, probe, torch.optim.SGD(model.parameters(), lr=0.1), poison=poison)
    report = probe.report()
    assert GRAD_NONFINITE in report["components"]["first.weight"]["gradient_states"]
    with pytest.raises(TrainabilityError, match="non-finite"):
        probe.assert_qualified()


def test_an_unreadable_gradient_is_an_evidence_gap_not_a_measured_zero():
    torch.manual_seed(0)
    model = _Tiny()
    names = _trainable_names(model, "first")

    class _Unreadable:
        def named_parameters(self):
            return [("first.weight", _UnreadableParam())]

    class _UnreadableParam:
        def __getattr__(self, item):
            if item == "grad":
                raise RuntimeError("unreadable test tensor")
            raise AttributeError(item)

    probe = TrainabilityProbe(
        _Unreadable(), ["first.weight"], window_steps=1, frozen_names=[]
    )
    probe.record_gradients(0)
    report = probe.report()
    assert report["components"]["first.weight"]["gradient_states"] == [GRAD_UNREADABLE]
    with pytest.raises(TrainabilityError, match="unreadable is an evidence gap"):
        probe.assert_qualified()


def test_a_zero_gradient_on_the_first_step_is_not_a_failure_when_the_window_covers_more():
    """LoRA A legitimately starts at zero gradient while B is zero. The window is
    declared up front, so 'not yet' is not the same as 'never'."""
    torch.manual_seed(0)
    model = _Tiny()
    names = _trainable_names(model, "first")
    probe = _probe(model, names, window_steps=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # step 0: no gradient reaches the parameter
    optimizer.zero_grad()
    out = 0.0 * model.first(torch.randn(2, 4)).sum()
    out.backward()
    probe.record_gradients(0)
    probe.record_update(0)

    # step 1: real gradient and a real update
    optimizer.zero_grad()
    model.first(torch.randn(2, 4)).sum().backward()
    probe.record_gradients(1)
    optimizer.step()
    probe.record_update(1)

    entry = probe.assert_qualified()["components"]["first.weight"]
    # the report records the *set* of states plus which steps were non-zero, so
    # "zero on step 0, real on step 1" is exactly what is asserted
    assert set(entry["gradient_states"]) == {GRAD_ZERO, GRAD_NONZERO}
    assert entry["nonzero_steps"] == [1]


def test_a_gradient_without_an_update_is_refused():
    """State that is read but never applied is not training."""
    torch.manual_seed(0)
    model = _Tiny()
    names = _trainable_names(model, "first")
    probe = _probe(model, names)
    _one_step(model, probe, torch.optim.SGD(model.parameters(), lr=0.1), update=False)
    with pytest.raises(TrainabilityError, match="no measurable optimizer update"):
        probe.assert_qualified()


def test_a_zero_step_window_is_refused_at_construction():
    model = _Tiny()
    with pytest.raises(ValueError, match="window_steps must be positive"):
        TrainabilityProbe(model, ["first.weight"], window_steps=0)


def test_probing_a_parameter_the_model_does_not_have_is_refused():
    with pytest.raises(TrainabilityError, match="does not have"):
        TrainabilityProbe(_Tiny(), ["mlp.gate.weight"], window_steps=1)


# --------------------------------------------------------------------------
# frozen verification and router-only scope
# --------------------------------------------------------------------------


def test_frozen_parameters_are_verified_by_digest_not_assumed():
    torch.manual_seed(0)
    model = _Tiny()
    names = _trainable_names(model, "first")
    probe = TrainabilityProbe(
        model, names, window_steps=1, frozen_names=_trainable_names(model, "frozen")
    )
    assert probe.assert_frozen_unchanged()["ok"] is True
    with torch.no_grad():
        model.frozen.weight.add_(1.0)  # a freeze policy that leaked
    report = probe.frozen_report()
    assert report["ok"] is False
    assert "frozen.weight" in report["changed"]
    with pytest.raises(TrainabilityError, match="frozen parameters changed"):
        probe.assert_frozen_unchanged()


def test_a_sampled_digest_says_it_is_sampled():
    """A sampled check must never be reported as complete equality."""
    big = torch.randn(2, 1_000_000)
    record = hash_parameters({"big": big})["big"]
    assert record["strategy"] == "sampled"
    assert record["sampled_elements"] < record["elements"]
    small = hash_parameters({"small": torch.randn(4)})["small"]
    assert small["strategy"] == "full"


def test_router_only_scope_requires_exactly_the_router_gates():
    class _Router(torch.nn.Module):
        def __init__(self, layers=4, with_shared_gate=False):
            super().__init__()
            self.layers = torch.nn.ModuleList()
            for _ in range(layers):
                block = torch.nn.Module()
                block.mlp = torch.nn.Module()
                block.mlp.gate = torch.nn.Linear(4, 2, bias=False)
                block.mlp.experts = torch.nn.Linear(4, 4, bias=False)
                if with_shared_gate:
                    block.mlp.shared_expert_gate = torch.nn.Linear(4, 1, bias=False)
                self.layers.append(block)

    model = _Router()
    gates = [n for n, _ in model.named_parameters() if n.endswith("mlp.gate.weight")]
    report = assert_router_only_scope(gates, model)
    assert report["router_count"] == 4
    assert report["architecture_router_count"] == 4

    # a shared-expert gate designated trainable is refused, not quietly trained
    with pytest.raises(TrainabilityError, match="exactly the router gates"):
        assert_router_only_scope(gates + ["layers.0.mlp.experts.weight"], model)

    # a missing layer's router is caught by the set comparison, not a count
    with pytest.raises(TrainabilityError, match="missing intended parameter paths"):
        assert_router_only_scope(gates[:-1], model)


def test_router_only_scope_refuses_an_empty_selection():
    with pytest.raises(TrainabilityError, match="nothing for the router to learn"):
        assert_router_only_scope([], _Tiny())


def test_router_only_scope_refuses_duplicate_gates():
    class _Router(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList()
            block = torch.nn.Module()
            block.mlp = torch.nn.Module()
            block.mlp.gate = torch.nn.Linear(4, 2, bias=False)
            self.layers.append(block)

    model = _Router()
    gate = [n for n, _ in model.named_parameters() if n.endswith("mlp.gate.weight")][0]
    with pytest.raises(TrainabilityError, match="duplicate router parameters"):
        assert_router_only_scope([gate, gate], model)


# --------------------------------------------------------------------------
# utilisation is a diagnostic, never a reachability proof
# --------------------------------------------------------------------------


def test_utilisation_is_reported_separately_from_reachability():
    rows = {"0": [10, 10, 0, 0], "1": [5, 5, 5, 5]}
    report = utilization_by_expert(rows)
    assert report["status"] == "measured"
    assert report["layers_with_unused_experts"] == ["0"]
    assert "not evidence of per-tensor reachability" in report["note"]
    assert utilization_by_expert(None)["status"] == "not_reported"


# --------------------------------------------------------------------------
# the digest is bounded and device-safe
# --------------------------------------------------------------------------


def test_the_sample_stride_never_labels_a_full_read_as_sampled():
    """Pure arithmetic, because this is the bug that shipped once already.

    Floor division gave stride 1 for a tensor just above the threshold, so a
    digest *labelled* sampled covered every element.
    """
    threshold = _FULL_HASH_MAX_ELEMENTS
    assert _sample_stride(threshold, threshold) == 1
    assert _sample_stride(threshold + 1, threshold) == 2
    assert _sample_stride(threshold * 2, threshold) == 2
    assert _sample_stride(threshold * 2 + 1, threshold) == 3
    # Above the threshold the stride is always at least 2, so "sampled" is true.
    for extra in (1, 7, 1_000, threshold):
        assert _sample_stride(threshold + extra, threshold) >= 2


def test_a_sampled_digest_stays_within_its_declared_budget():
    threshold = _FULL_HASH_MAX_ELEMENTS
    big = torch.randn(threshold + 1)
    record = _tensor_digest(big)
    assert record["strategy"] == "sampled"
    assert record["stride"] >= 2
    assert record["sampled_elements"] <= threshold + 1
    assert record["sampled_elements"] < record["elements"]
    assert record["elements"] == threshold + 1


def test_the_digest_is_the_same_value_on_every_device_it_can_reach():
    """A device is a place a value lives, not part of the value."""
    values = torch.randn(64)
    reference = _tensor_digest(values)
    if torch.cuda.is_available():
        moved = _tensor_digest(values.clone().cuda())
        assert moved["digest"] == reference["digest"]
        assert moved["strategy"] == reference["strategy"]


def test_a_large_digest_on_an_accelerator_matches_the_host_result():
    """The real device path, exercised whenever a device is actually present.

    CI runners without a GPU skip this; the machine that develops Chowder has
    one, so the accelerator path is not left permanently untested.
    """
    if not torch.cuda.is_available():  # pragma: no cover - depends on the host
        pytest.skip("no CUDA device on this host")
    threshold = _FULL_HASH_MAX_ELEMENTS
    values = torch.randn(threshold + 5)
    host = _tensor_digest(values)
    device = _tensor_digest(values.clone().cuda())
    assert host["strategy"] == device["strategy"] == "sampled"
    assert host["stride"] == device["stride"]
    assert host["sampled_elements"] == device["sampled_elements"]
    assert host["digest"] == device["digest"]


class _AcceleratorBoundaryTensor:
    """A CPU tensor that behaves like an accelerator one at the `.numpy()` boundary.

    This is the regression for the shipped defect: the digest converted the whole
    flattened tensor to fp32 *in place on its device* and then called `.numpy()`
    with no host move, so on an accelerator it raised instead of measuring. A
    stub is used rather than a real device so the failure mode is exercised on
    GPU-less CI too, and the contract it enforces is the real torch one: a host
    transfer must happen before `.numpy()`.
    """

    def __init__(self, inner, *, on_host=False):
        self._inner = inner
        self._on_host = on_host
        self.dtype = inner.dtype

    @property
    def device(self):
        return torch.device("cpu") if self._on_host else torch.device("cuda", 0)

    def _wrap(self, inner, *, on_host=None):
        return _AcceleratorBoundaryTensor(
            inner, on_host=self._on_host if on_host is None else on_host
        )

    def detach(self):
        return self

    def numel(self):
        return self._inner.numel()

    def reshape(self, *shape):
        return self._wrap(self._inner.reshape(*shape))

    def __getitem__(self, key):
        return self._wrap(self._inner[key])

    def to(self, destination):
        if isinstance(destination, torch.dtype):
            return self._wrap(self._inner.to(destination))
        if str(destination) == "cpu":
            return self._wrap(self._inner, on_host=True)
        return self

    def cpu(self):
        return self._wrap(self._inner, on_host=True)

    def contiguous(self):
        return self._wrap(self._inner.contiguous())

    def numpy(self):
        if not self._on_host:
            raise TypeError(
                "can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() "
                "to copy the tensor to host memory first."
            )
        return self._inner.numpy()


def test_the_digest_moves_values_to_the_host_before_reading_them():
    """The exact defect: `.numpy()` on a tensor that never left its device."""
    inner = torch.randn(32, dtype=torch.float64)
    reference = _tensor_digest(inner)

    boundary = _AcceleratorBoundaryTensor(inner)
    assert boundary.device.type == "cuda"
    with pytest.raises(TypeError, match="to host memory first"):
        boundary.numpy()  # the old code's final step, on an unmoved tensor

    measured = _tensor_digest(boundary)
    assert measured["digest"] == reference["digest"]
    assert measured["elements"] == reference["elements"]


def test_a_large_accelerator_boundary_digest_gathers_before_it_widens():
    """The fp32 conversion must touch the samples, not the whole tensor."""
    threshold = _FULL_HASH_MAX_ELEMENTS
    inner = torch.randn(threshold + 3, dtype=torch.float64)
    boundary = _AcceleratorBoundaryTensor(inner)
    record = _tensor_digest(boundary)
    assert record["strategy"] == "sampled"
    assert record["elements"] == threshold + 3
    assert record["sampled_elements"] < record["elements"]
    assert record["digest"] == _tensor_digest(inner)["digest"]
