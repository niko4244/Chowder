"""Tests for static FFN channel pruning.

Reuses the hot-core suite's synthetic builder, which carries REAL float32 values
and the vision/MTP `.mlp.` tensors that caught a dispatch bug in the MoE
converter. The decisive test is numerical: the pruned FFN must compute exactly
the dense FFN restricted to the kept channels -- no scaling, no renormalisation,
nothing to compensate for. That is the whole claim of this converter.
"""

import json
import math

import pytest

from chowder.channel_importance import ChannelRanking
from chowder.local_model_manifest import (
    LocalModelManifest,
    build_local_model_manifest,
    verify_local_model_manifest,
)
from chowder.static_prune import (
    StaticPruneError,
    plan_static_prune,
    prune_checkpoint,
)

from test_hot_core_upcycle import (  # reuse the fixture builder
    H,
    I,
    LAYERS,
    _dense_dir,
    _rowmajor,
    _read_tensors,
    _ranking_for,
    _silu,
)

KEEP = 6


@pytest.fixture
def pruned(tmp_path):
    source = _dense_dir(tmp_path / "dense")
    ranking = _ranking_for(source)
    out = tmp_path / "pruned"
    prov = prune_checkpoint(source, out, ranking, keep_channels=KEEP)
    return source, out, ranking, prov


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------


def test_plan_prices_the_prune_and_storage_falls_with_compute(tmp_path):
    source = _dense_dir(tmp_path / "dense")
    plan = plan_static_prune(source, _ranking_for(source), keep_channels=KEEP)
    per_channel = 3 * H * LAYERS
    assert plan.keep_channels == KEEP
    assert plan.dense_ffn_params == I * per_channel
    assert plan.pruned_ffn_params == KEEP * per_channel
    # unlike the MoE, nothing unused is stored, so total == active
    assert plan.total_params == plan.always_on_params + KEEP * per_channel
    assert plan.to_dict()["active_params"] == plan.total_params
    assert plan.to_dict()["ffn_params_removed"] == (I - KEEP) * per_channel


def test_plan_refuses_a_no_op_or_nonsense_keep(tmp_path):
    source = _dense_dir(tmp_path / "dense")
    ranking = _ranking_for(source)
    for bad in (0, I, I + 1, -1):
        with pytest.raises(StaticPruneError, match="keep_channels must be in"):
            plan_static_prune(source, ranking, keep_channels=bad)


def test_plan_refuses_a_ranking_from_another_checkpoint(tmp_path):
    source = _dense_dir(tmp_path / "dense")
    other = _dense_dir(tmp_path / "other", layers=1)
    foreign_manifest = build_local_model_manifest(other, mode="fast")
    foreign = ChannelRanking(
        method="t", source_dir=str(source),
        source_manifest_sha256=foreign_manifest.manifest_sha256,
        intermediate_size=I, ranking={l: tuple(range(I)) for l in range(LAYERS)},
        calibration={}, concentration={})
    with pytest.raises(StaticPruneError, match="different checkpoint"):
        plan_static_prune(source, foreign, keep_channels=KEEP)


def test_plan_refuses_a_ranking_missing_a_layer(tmp_path):
    source = _dense_dir(tmp_path / "dense")
    manifest = build_local_model_manifest(source, mode="fast")
    partial = ChannelRanking(
        method="t", source_dir=str(source),
        source_manifest_sha256=manifest.manifest_sha256, intermediate_size=I,
        ranking={0: tuple(range(I))}, calibration={}, concentration={})
    with pytest.raises(StaticPruneError, match="missing layers"):
        plan_static_prune(source, partial, keep_channels=KEEP)


def test_prune_refuses_a_non_empty_output_dir(tmp_path):
    source = _dense_dir(tmp_path / "dense")
    out = tmp_path / "out"
    out.mkdir()
    (out / "stray").write_text("x", encoding="utf-8")
    with pytest.raises(StaticPruneError, match="not empty"):
        prune_checkpoint(source, out, _ranking_for(source), keep_channels=KEEP)


# ---------------------------------------------------------------------------
# output shape, config, passthrough
# ---------------------------------------------------------------------------


def test_only_intermediate_size_changes_in_the_config(pruned):
    source, out, _, _ = pruned
    before = json.loads((source / "config.json").read_text(encoding="utf-8"))
    after = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert after["text_config"]["intermediate_size"] == KEEP
    # the architecture is NOT changed -- this is the point versus the MoE path
    assert after["model_type"] == before["model_type"] == "qwen3_5"
    assert after["architectures"] == before["architectures"]
    b, a = dict(before["text_config"]), dict(after["text_config"])
    b.pop("intermediate_size"), a.pop("intermediate_size")
    assert a == b, "a prune must change nothing else in text_config"


def test_mlp_tensors_are_narrowed_and_everything_else_is_byte_identical(pruned):
    source, out, _, _ = pruned
    src = _read_tensors(source / "model-00001-of-00001.safetensors")
    dst = _read_tensors(out / "model-00001-of-00001.safetensors")
    assert set(src) == set(dst), "a prune must not add or drop tensors"
    narrowed = 0
    for name, (shape, values) in src.items():
        is_mlp = name.startswith("model.language_model.layers.") and name.endswith(
            ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"))
        if is_mlp:
            narrowed += 1
            want = [H, KEEP] if name.endswith("down_proj.weight") else [KEEP, H]
            assert dst[name][0] == want, name
        else:
            assert dst[name] == (shape, values), name
    assert narrowed == 3 * LAYERS
    # the vision tower keeps its own intermediate size
    for name in ("model.visual.blocks.0.mlp.linear_fc1.weight",
                 "model.visual.blocks.0.mlp.linear_fc2.weight",
                 "mtp.fc.mlp.down_proj.weight"):
        assert dst[name] == src[name], name


def test_manifest_verifies_and_index_matches(pruned):
    _, out, _, prov = pruned
    manifest = LocalModelManifest.from_dict(
        json.loads((out / "conversion.manifest.json").read_text(encoding="utf-8")))
    verification = verify_local_model_manifest(manifest, out)
    assert verification.clean, verification.divergences
    index = json.loads((out / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert set(index["weight_map"]) == set(
        _read_tensors(out / "model-00001-of-00001.safetensors"))
    assert prov["output_manifest_sha256"]


def test_provenance_records_that_nothing_is_scaled(pruned):
    _, _, _, prov = pruned
    assert prov["converter"] == "static_prune"
    assert prov["scaling"].startswith("NONE")
    assert "rank order" in prov["channel_order"]
    assert prov["plan"]["keep_channels"] == KEEP


def test_channels_are_emitted_in_rank_order_so_a_further_prune_is_a_slice(pruned):
    """Rank order is free (the FFN sum is permutation-invariant) and makes a
    deeper prune a prefix slice instead of a re-ranking."""
    source, out, ranking, _ = pruned
    src = _read_tensors(source / "model-00001-of-00001.safetensors")
    dst = _read_tensors(out / "model-00001-of-00001.safetensors")
    for layer in range(LAYERS):
        name = f"model.language_model.layers.{layer}.mlp.gate_proj.weight"
        src_rows = _rowmajor(src[name][1], H)
        dst_rows = _rowmajor(dst[name][1], H)
        expected = [src_rows[c] for c in ranking.ranking[layer][:KEEP]]
        assert dst_rows == expected, f"layer {layer} rows are not in rank order"


# ---------------------------------------------------------------------------
# the decisive numerical test
# ---------------------------------------------------------------------------


def test_pruned_ffn_equals_the_dense_ffn_restricted_to_kept_channels(pruned):
    """The whole claim: no scaling, no approximation. The pruned FFN computes
    exactly `sum over kept channels` of the dense FFN's terms, for any input."""
    source, out, ranking, _ = pruned
    src = _read_tensors(source / "model-00001-of-00001.safetensors")
    dst = _read_tensors(out / "model-00001-of-00001.safetensors")

    for x in ([0.5, -1.25, 2.0, 0.125], [-0.75, 1.5, 0.25, -2.0]):
        for layer in range(LAYERS):
            pre = f"model.language_model.layers.{layer}.mlp."
            kept = ranking.ranking[layer][:KEEP]

            s_gate = _rowmajor(src[pre + "gate_proj.weight"][1], H)
            s_up = _rowmajor(src[pre + "up_proj.weight"][1], H)
            s_down = _rowmajor(src[pre + "down_proj.weight"][1], I)
            expected = [0.0] * H
            for c in kept:
                h = _silu(sum(s_gate[c][k] * x[k] for k in range(H))) * \
                    sum(s_up[c][k] * x[k] for k in range(H))
                for r in range(H):
                    expected[r] += s_down[r][c] * h

            p_gate = _rowmajor(dst[pre + "gate_proj.weight"][1], H)
            p_up = _rowmajor(dst[pre + "up_proj.weight"][1], H)
            p_down = _rowmajor(dst[pre + "down_proj.weight"][1], KEEP)
            got = [0.0] * H
            for j in range(KEEP):
                h = _silu(sum(p_gate[j][k] * x[k] for k in range(H))) * \
                    sum(p_up[j][k] * x[k] for k in range(H))
                for r in range(H):
                    got[r] += p_down[r][j] * h

            for a, b in zip(got, expected):
                assert a == pytest.approx(b, rel=1e-12, abs=1e-12)


def test_a_deeper_prune_of_the_pruned_model_matches_pruning_the_original(tmp_path):
    """Rank-ordered output means prune(prune(x, 6), 3) == prune(x, 3) on the FFN
    weights. Verified rather than asserted, because it is the property that makes
    an already-pruned checkpoint re-usable."""
    source = _dense_dir(tmp_path / "dense")
    ranking = _ranking_for(source)
    deep = tmp_path / "deep"
    prune_checkpoint(source, deep, ranking, keep_channels=3)
    once = tmp_path / "once"
    prune_checkpoint(source, once, ranking, keep_channels=6)

    deep_t = _read_tensors(deep / "model-00001-of-00001.safetensors")
    once_t = _read_tensors(once / "model-00001-of-00001.safetensors")
    for layer in range(LAYERS):
        pre = f"model.language_model.layers.{layer}.mlp."
        for leaf in ("gate_proj", "up_proj"):
            rows_deep = _rowmajor(deep_t[pre + leaf + ".weight"][1], H)
            rows_once = _rowmajor(once_t[pre + leaf + ".weight"][1], H)
            assert rows_deep == rows_once[:3]
        cols_deep = _rowmajor(deep_t[pre + "down_proj.weight"][1], 3)
        cols_once = _rowmajor(once_t[pre + "down_proj.weight"][1], 6)
        assert cols_deep == [row[:3] for row in cols_once]
