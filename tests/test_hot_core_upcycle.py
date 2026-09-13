"""Tests for the hot-core converter.

Offline, deterministic, no torch: synthetic dense qwen3_5-style shards carrying
REAL float32 values (the partition-converter suite writes zero-filled data,
which cannot catch a wrong permutation). F32 is used so `_scale_bytes_exact`
takes its value-multiply path and every comparison below is exact rather than
approximate.

The two tests that matter most are the numerical ones at the bottom. They
reconstruct, in pure Python, what the real `Qwen3_5MoeSparseMoeBlock` computes
at init and assert it equals the dense FFN restricted to the channels the
conversion kept. Those pin the two scaling decisions that are easy to get
silently wrong:

* the shared branch is gated by `sigmoid(shared_expert_gate(x))`, which is
  exactly 0.5 at zero init, so the hot core's `down_proj` must carry x2;
* renormalised routed weights are `1/top_k` each, so cold `down_proj` must carry
  x`top_k` -- and NOT xE, which is what forces `dense_to_moe`'s top_k == E and
  produces the (E/k) blowup at any smaller top_k.
"""

import json
import math
import struct

import pytest

from chowder.channel_importance import ChannelRanking, split_disjoint
from chowder.hot_core_upcycle import (
    HotCorePlan,
    HotCoreScheme,
    HotCoreUpcycleError,
    _gather_columns,
    _gather_rows,
    convert_checkpoint_hot_core,
    plan_hot_core_conversion,
)
from chowder.local_model_manifest import (
    LocalModelManifest,
    build_local_model_manifest,
    verify_local_model_manifest,
)

H = 4       # hidden
I = 16      # intermediate
LAYERS = 2
DT = "F32"
W = 4


# ---------------------------------------------------------------------------
# synthetic checkpoint with real values
# ---------------------------------------------------------------------------


def _f32(values):
    return b"".join(struct.pack("<f", v) for v in values)


def _unpack(raw):
    return [struct.unpack_from("<f", raw, i)[0] for i in range(0, len(raw), 4)]


def _value(layer, kind, a, b):
    """Deterministic, distinct per (layer, tensor, row, col) -- a wrong
    permutation cannot coincidentally match."""
    base = {"gate": 1.0, "up": 2.0, "down": 3.0}[kind]
    return base + layer * 0.5 + a * 0.125 + b * 0.015625


def _write_shard(path, tensors):
    header, offset, blobs = {}, 0, []
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
        blobs.append(raw)
    blob = json.dumps(header).encode("utf-8")
    blob += b" " * ((8 - len(blob) % 8) % 8)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        for raw in blobs:
            fh.write(raw)


def _dense_dir(root, *, layers=LAYERS):
    root.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for layer in range(layers):
        pre = f"model.language_model.layers.{layer}."
        tensors[pre + "mlp.gate_proj.weight"] = (
            DT, [I, H], _f32([_value(layer, "gate", c, h) for c in range(I) for h in range(H)]))
        tensors[pre + "mlp.up_proj.weight"] = (
            DT, [I, H], _f32([_value(layer, "up", c, h) for c in range(I) for h in range(H)]))
        tensors[pre + "mlp.down_proj.weight"] = (
            DT, [H, I], _f32([_value(layer, "down", r, c) for r in range(H) for c in range(I)]))
        tensors[pre + "self_attn.q_proj.weight"] = (
            DT, [H, H], _f32([0.25 * (r + 1) + h for r in range(H) for h in range(H)]))
        tensors[pre + "input_layernorm.weight"] = (DT, [H], _f32([1.0] * H))
    tensors["model.language_model.embed_tokens.weight"] = (
        DT, [8, H], _f32([0.5 * i for i in range(8 * H)]))
    tensors["lm_head.weight"] = (DT, [8, H], _f32([0.25 * i for i in range(8 * H)]))
    # The vision tower's blocks ALSO carry `.mlp.` tensors. A converter that
    # dispatches on `".mlp." in name` sweeps these into the decoder-MLP path and
    # either corrupts the vision tower or crashes on a bogus layer id. The real
    # 9B has 108 of them and caught exactly that bug; these are the regression.
    tensors["model.visual.blocks.0.mlp.linear_fc1.weight"] = (
        DT, [H, H], _f32([0.75 * i for i in range(H * H)]))
    tensors["model.visual.blocks.0.mlp.linear_fc1.bias"] = (DT, [H], _f32([0.1] * H))
    tensors["model.visual.blocks.0.mlp.linear_fc2.weight"] = (
        DT, [H, H], _f32([1.25 * i for i in range(H * H)]))
    tensors["mtp.fc.mlp.down_proj.weight"] = (
        DT, [H, H], _f32([0.6 * i for i in range(H * H)]))
    _write_shard(root / "model-00001-of-00001.safetensors", tensors)
    (root / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {"hidden_size": H, "intermediate_size": I,
                        "num_hidden_layers": layers},
    }), encoding="utf-8")
    (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    return root


def _read_tensors(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
        data = fh.read()
    out = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        lo, hi = entry["data_offsets"]
        out[name] = (entry["shape"], _unpack(data[lo:hi]))
    return out


def _ranking_for(source, *, reverse=False):
    """A ranking that is deliberately NOT index order, so any code path that
    silently falls back to index order fails these tests."""
    manifest = build_local_model_manifest(source, mode="fast")
    order = list(range(I))
    # interleave high/low so hot channels are scattered through the index space
    scattered = []
    lo, hi = 0, I - 1
    while lo <= hi:
        scattered.append(hi if not reverse else lo)
        if lo != hi:
            scattered.append(lo if not reverse else hi)
        lo, hi = lo + 1, hi - 1
    assert sorted(scattered) == order
    return ChannelRanking(
        method="test_fixture",
        source_dir=str(source),
        source_manifest_sha256=manifest.manifest_sha256,
        intermediate_size=I,
        ranking={layer: tuple(scattered) for layer in range(LAYERS)},
        calibration={"prompts": 4, "split": "even"},
        concentration={"top10pct_mass_mean": 0.22},
    )


# ---------------------------------------------------------------------------
# byte-level gathers
# ---------------------------------------------------------------------------


def test_gather_rows_picks_whole_contiguous_rows_in_order():
    raw = _f32([10 * c + h for c in range(5) for h in range(3)])
    got = _unpack(_gather_rows(raw, (3, 0, 4), hidden=3, width=4))
    assert got == [30, 31, 32, 0, 1, 2, 40, 41, 42]


def test_gather_columns_transposes_strided_columns_correctly():
    # (hidden=3, intermediate=5), element (r, c) == 10*r + c
    raw = _f32([10 * r + c for r in range(3) for c in range(5)])
    got = _unpack(_gather_columns(raw, (4, 1), hidden=3, intermediate=5, width=4))
    # expected (hidden=3, n=2) row-major: [(0,4),(0,1)], [(1,4),(1,1)], ...
    assert got == [4, 1, 14, 11, 24, 21]


def test_gather_columns_refuses_a_shape_it_cannot_verify():
    raw = _f32([0.0] * 14)  # not hidden*intermediate for the args below
    with pytest.raises(HotCoreUpcycleError, match="expected hidden\\*intermediate"):
        _gather_columns(raw, (0,), hidden=3, intermediate=5, width=4)


# ---------------------------------------------------------------------------
# scheme validation
# ---------------------------------------------------------------------------


def _scheme(**over):
    ranking = ChannelRanking(
        method="t", source_dir="d", source_manifest_sha256="x", intermediate_size=I,
        ranking={0: tuple(range(I))}, calibration={}, concentration={},
    )
    kwargs = {"num_experts": 4, "top_k": 2, "hot_core_size": 4}
    kwargs.update(over)
    return HotCoreScheme.from_ranking(ranking, 0, **kwargs)


def test_scheme_partitions_every_channel_exactly_once():
    scheme = _scheme()
    seen = list(scheme.hot) + [c for g in scheme.cold_by_expert for c in g]
    assert sorted(seen) == list(range(I))
    assert scheme.moe_intermediate_size == (I - 4) // 4
    assert scheme.active_channels == 4 + 2 * 3


def test_scheme_refuses_non_power_of_two_top_k():
    with pytest.raises(HotCoreUpcycleError, match="power of two"):
        _scheme(top_k=3)


def test_scheme_refuses_top_k_above_num_experts():
    with pytest.raises(HotCoreUpcycleError, match="must be in 1"):
        _scheme(num_experts=2, top_k=4)


def test_scheme_refuses_ragged_cold_split():
    with pytest.raises(HotCoreUpcycleError, match="do not divide evenly"):
        _scheme(hot_core_size=5, num_experts=4)


def test_scheme_refuses_empty_or_full_core():
    with pytest.raises(HotCoreUpcycleError, match="hot_core_size must be in"):
        _scheme(hot_core_size=0)
    with pytest.raises(HotCoreUpcycleError, match="hot_core_size must be in"):
        _scheme(hot_core_size=I)


def test_scheme_refuses_a_layer_the_ranking_never_measured():
    ranking = ChannelRanking(
        method="t", source_dir="d", source_manifest_sha256="x", intermediate_size=I,
        ranking={0: tuple(range(I))}, calibration={}, concentration={},
    )
    with pytest.raises(HotCoreUpcycleError, match="no entry for layer 7"):
        HotCoreScheme.from_ranking(ranking, 7, num_experts=4, top_k=2, hot_core_size=4)


def test_cold_is_dealt_round_robin_so_no_expert_is_the_warm_one():
    """Contiguous rank blocks would hand expert 0 every warm channel and give a
    router a reason to collapse onto it. Round-robin equalises rank profiles."""
    E, h = 4, 4
    scheme = _scheme(num_experts=E, hot_core_size=h)
    cold_ranks = list(range(h, I))          # rank position of each cold channel
    per = len(cold_ranks) // E

    def spread(groups):
        means = [sum(g) / len(g) for g in groups]
        return max(means) - min(means)

    round_robin = spread([cold_ranks[e::E] for e in range(E)])
    contiguous = spread([cold_ranks[e * per:(e + 1) * per] for e in range(E)])
    # round-robin's spread is E-1 by construction; contiguous blocks are per times
    # worse, which is what would hand expert 0 every warm channel.
    assert round_robin == E - 1
    assert contiguous == pytest.approx(per * (E - 1))
    assert round_robin < contiguous
    # and the converter must actually be using round-robin
    observed = spread([[cold_ranks.index(c) + h for c in g] for g in scheme.cold_by_expert])
    assert observed == round_robin


# ---------------------------------------------------------------------------
# plan-time refusals
# ---------------------------------------------------------------------------


def test_plan_refuses_a_ranking_measured_on_another_checkpoint(tmp_path):
    source = _dense_dir(tmp_path / "dense")
    other = _dense_dir(tmp_path / "other", layers=1)
    foreign = _ranking_for(other)
    foreign = ChannelRanking(
        method=foreign.method, source_dir=str(source),
        source_manifest_sha256=foreign.source_manifest_sha256,
        intermediate_size=I, ranking={l: tuple(range(I)) for l in range(LAYERS)},
        calibration={}, concentration={},
    )
    with pytest.raises(HotCoreUpcycleError, match="different checkpoint"):
        plan_hot_core_conversion(source, foreign, num_experts=4, top_k=2, hot_core_size=4)


def test_plan_refuses_a_ranking_with_a_missing_layer(tmp_path):
    source = _dense_dir(tmp_path / "dense")
    manifest = build_local_model_manifest(source, mode="fast")
    partial = ChannelRanking(
        method="t", source_dir=str(source),
        source_manifest_sha256=manifest.manifest_sha256, intermediate_size=I,
        ranking={0: tuple(range(I))}, calibration={}, concentration={},
    )
    with pytest.raises(HotCoreUpcycleError, match="missing layers"):
        plan_hot_core_conversion(source, partial, num_experts=4, top_k=2, hot_core_size=4)


def test_plan_prices_storage_and_active_against_dense(tmp_path):
    source = _dense_dir(tmp_path / "dense")
    plan = plan_hot_core_conversion(
        source, _ranking_for(source), num_experts=4, top_k=2, hot_core_size=4)
    assert plan.moe_intermediate_size == 3
    assert plan.active_channels == 4 + 2 * 3
    per_channel = 3 * H * LAYERS
    assert plan.stored_ffn_params == (4 + 4 * 3) * per_channel
    assert plan.active_ffn_params == (4 + 2 * 3) * per_channel
    assert plan.dense_ffn_params == I * per_channel
    # stored exceeds dense only if the core is replicated; it must not be here
    assert plan.stored_ffn_params == I * per_channel


def test_convert_refuses_a_non_empty_output_dir(tmp_path):
    source = _dense_dir(tmp_path / "dense")
    out = tmp_path / "out"
    out.mkdir()
    (out / "stray.txt").write_text("x", encoding="utf-8")
    with pytest.raises(HotCoreUpcycleError, match="not empty"):
        convert_checkpoint_hot_core(
            source, out, _ranking_for(source), num_experts=4, top_k=2, hot_core_size=4)


# ---------------------------------------------------------------------------
# end-to-end conversion
# ---------------------------------------------------------------------------


@pytest.fixture
def converted(tmp_path):
    source = _dense_dir(tmp_path / "dense")
    ranking = _ranking_for(source)
    out = tmp_path / "moe"
    prov = convert_checkpoint_hot_core(
        source, out, ranking, num_experts=4, top_k=2, hot_core_size=4)
    return source, out, ranking, prov


def test_conversion_writes_a_verifiable_manifest_and_index(converted):
    _, out, _, prov = converted
    manifest_path = out / "conversion.manifest.json"
    assert manifest_path.is_file()
    manifest = LocalModelManifest.from_dict(
        json.loads(manifest_path.read_text(encoding="utf-8")))
    verification = verify_local_model_manifest(manifest, out)
    assert verification.clean, verification.divergences
    index = json.loads((out / "model.safetensors.index.json").read_text(encoding="utf-8"))
    tensors = _read_tensors(out / "model-00001-of-00001.safetensors")
    assert set(index["weight_map"]) == set(tensors)
    assert prov["output_manifest_sha256"]


def test_config_records_top_k_not_num_experts(converted):
    _, out, _, _ = converted
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))["text_config"]
    assert cfg["num_experts"] == 4
    # the whole point: this init does not require every expert to run
    assert cfg["num_experts_per_tok"] == 2
    assert cfg["moe_intermediate_size"] == 3
    assert cfg["shared_expert_intermediate_size"] == 4
    assert cfg["model_type"] == "qwen3_5_moe_text"


def test_everything_the_converter_did_not_create_is_byte_identical(converted):
    """Including tensors whose names contain `.mlp.` but are not decoder MLPs:
    the vision tower and MTP. Only the three dense decoder MLP tensors per layer
    are consumed; every other source tensor must survive untouched."""
    source, out, _, _ = converted
    src = _read_tensors(source / "model-00001-of-00001.safetensors")
    dst = _read_tensors(out / "model-00001-of-00001.safetensors")
    consumed = {
        f"model.language_model.layers.{layer}.mlp.{leaf}.weight"
        for layer in range(LAYERS)
        for leaf in ("gate_proj", "up_proj", "down_proj")
    }
    checked = 0
    for name, (shape, values) in src.items():
        if name in consumed:
            assert name not in dst, f"{name} should have been replaced"
            continue
        assert dst[name] == (shape, values), name
        checked += 1
    assert checked == len(src) - len(consumed)
    # the regression specifically: vision/MTP mlp tensors present and unchanged
    for name in ("model.visual.blocks.0.mlp.linear_fc1.weight",
                 "model.visual.blocks.0.mlp.linear_fc1.bias",
                 "model.visual.blocks.0.mlp.linear_fc2.weight",
                 "mtp.fc.mlp.down_proj.weight"):
        assert dst[name] == src[name], name


def test_router_and_shared_gate_are_zero(converted):
    _, out, _, _ = converted
    dst = _read_tensors(out / "model-00001-of-00001.safetensors")
    for layer in range(LAYERS):
        pre = f"model.language_model.layers.{layer}.mlp."
        assert dst[pre + "gate.weight"][1] == [0.0] * (4 * H)
        assert dst[pre + "shared_expert_gate.weight"][1] == [0.0] * H


def test_provenance_states_plainly_that_this_init_is_not_exact(converted):
    _, _, _, prov = converted
    assert prov["converter"] == "hot_core_upcycle"
    assert prov["exactness_contract"].startswith("NONE")
    # the tie-break caveat must be recorded, not glossed
    assert "torch.topk" in prov["router_init"]
    assert prov["ranking"]["digest"]
    assert len(prov["scheme_digests"]) == LAYERS


# ---------------------------------------------------------------------------
# the numerical checks: does the converted block compute what it should?
# ---------------------------------------------------------------------------


def _silu(v):
    return v / (1.0 + math.exp(-v))


def _rowmajor(values, cols):
    return [values[i * cols:(i + 1) * cols] for i in range(len(values) // cols)]


def _dense_contribution(source, layer, x, channels):
    """Dense FFN output restricted to `channels`: sum_c down[:,c] * h_c."""
    src = _read_tensors(source / "model-00001-of-00001.safetensors")
    pre = f"model.language_model.layers.{layer}.mlp."
    gate = _rowmajor(src[pre + "gate_proj.weight"][1], H)
    up = _rowmajor(src[pre + "up_proj.weight"][1], H)
    down = _rowmajor(src[pre + "down_proj.weight"][1], I)
    out = [0.0] * H
    for c in channels:
        h = _silu(sum(gate[c][k] * x[k] for k in range(H))) * sum(up[c][k] * x[k] for k in range(H))
        for r in range(H):
            out[r] += down[r][c] * h
    return out


def _mlp_branch(gate_rows, up_rows, down_cols, x, n):
    """One SwiGLU MLP from already-gathered weights; down_cols is (H, n)."""
    out = [0.0] * H
    for j in range(n):
        h = _silu(sum(gate_rows[j][k] * x[k] for k in range(H))) * \
            sum(up_rows[j][k] * x[k] for k in range(H))
        for r in range(H):
            out[r] += down_cols[r][j] * h
    return out


def test_shared_branch_reproduces_the_dense_hot_core_through_a_zero_gate(converted):
    """sigmoid(0) == 0.5 exactly, so the x2 on the hot core's down_proj must make
    the gated shared branch equal the dense contribution of those channels."""
    source, out, ranking, _ = converted
    dst = _read_tensors(out / "model-00001-of-00001.safetensors")
    x = [0.5, -1.25, 2.0, 0.125]
    for layer in range(LAYERS):
        scheme = HotCoreScheme.from_ranking(
            ranking, layer, num_experts=4, top_k=2, hot_core_size=4)
        pre = f"model.language_model.layers.{layer}.mlp.shared_expert."
        branch = _mlp_branch(
            _rowmajor(dst[pre + "gate_proj.weight"][1], H),
            _rowmajor(dst[pre + "up_proj.weight"][1], H),
            _rowmajor(dst[pre + "down_proj.weight"][1], 4),
            x, 4,
        )
        gated = [0.5 * v for v in branch]          # sigmoid(shared_expert_gate=0)
        expected = _dense_contribution(source, layer, x, scheme.hot)
        for got, want in zip(gated, expected):
            assert got == pytest.approx(want, rel=1e-12, abs=1e-12)


def test_routed_branch_reproduces_the_dense_cold_channels_of_the_chosen_experts(converted):
    """Renormalised routed weights are 1/top_k each, so the x top_k on cold
    down_proj must make the weighted sum equal those channels' dense
    contribution -- for ANY choice of top_k experts, which is exactly what xE
    fails to do."""
    source, out, ranking, _ = converted
    dst = _read_tensors(out / "model-00001-of-00001.safetensors")
    x = [0.5, -1.25, 2.0, 0.125]
    top_k, num_experts, c = 2, 4, 3
    for layer in range(LAYERS):
        scheme = HotCoreScheme.from_ranking(
            ranking, layer, num_experts=num_experts, top_k=top_k, hot_core_size=4)
        pre = f"model.language_model.layers.{layer}.mlp.experts."
        gu = dst[pre + "gate_up_proj"][1]
        dn = dst[pre + "down_proj"][1]
        gu_per, dn_per = 2 * c * H, H * c
        for chosen in ((0, 1), (1, 3), (2, 3)):
            total = [0.0] * H
            for e in chosen:
                block = gu[e * gu_per:(e + 1) * gu_per]
                rows = _rowmajor(block, H)
                branch = _mlp_branch(
                    rows[:c], rows[c:],
                    _rowmajor(dn[e * dn_per:(e + 1) * dn_per], c), x, c,
                )
                for r in range(H):
                    total[r] += branch[r] / top_k     # renormalised weight
            expected = _dense_contribution(
                source, layer, x,
                [ch for e in chosen for ch in scheme.cold_by_expert[e]])
            for got, want in zip(total, expected):
                assert got == pytest.approx(want, rel=1e-12, abs=1e-12)


def test_full_block_at_top_k_equals_dense_over_exactly_the_kept_channels(converted):
    """The init's real claim: not exactness, but that what IS computed is
    computed correctly -- shared core plus top_k cold slices, nothing rescaled
    wrongly, nothing dropped silently."""
    source, out, ranking, _ = converted
    dst = _read_tensors(out / "model-00001-of-00001.safetensors")
    x = [-0.75, 1.5, 0.25, -2.0]
    top_k, c, chosen = 2, 3, (0, 2)
    for layer in range(LAYERS):
        scheme = HotCoreScheme.from_ranking(
            ranking, layer, num_experts=4, top_k=top_k, hot_core_size=4)
        sp = f"model.language_model.layers.{layer}.mlp.shared_expert."
        ep = f"model.language_model.layers.{layer}.mlp.experts."
        block = [0.5 * v for v in _mlp_branch(
            _rowmajor(dst[sp + "gate_proj.weight"][1], H),
            _rowmajor(dst[sp + "up_proj.weight"][1], H),
            _rowmajor(dst[sp + "down_proj.weight"][1], 4), x, 4)]
        gu, dn = dst[ep + "gate_up_proj"][1], dst[ep + "down_proj"][1]
        for e in chosen:
            rows = _rowmajor(gu[e * 2 * c * H:(e + 1) * 2 * c * H], H)
            branch = _mlp_branch(rows[:c], rows[c:],
                                 _rowmajor(dn[e * H * c:(e + 1) * H * c], c), x, c)
            for r in range(H):
                block[r] += branch[r] / top_k
        kept = list(scheme.hot) + [ch for e in chosen for ch in scheme.cold_by_expert[e]]
        assert len(set(kept)) == 4 + top_k * c
        expected = _dense_contribution(source, layer, x, kept)
        for got, want in zip(block, expected):
            assert got == pytest.approx(want, rel=1e-12, abs=1e-12)


# ---------------------------------------------------------------------------
# ranking artifact
# ---------------------------------------------------------------------------


def test_ranking_roundtrips_and_detects_post_hoc_edits(tmp_path):
    source = _dense_dir(tmp_path / "dense")
    ranking = _ranking_for(source)
    path = ranking.write(tmp_path / "rank.json")
    assert ChannelRanking.read(path).digest() == ranking.digest()

    payload = json.loads(path.read_text(encoding="utf-8"))
    order = payload["ranking"]["0"]
    payload["ranking"]["0"] = [order[1], order[0]] + list(order[2:])
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Exception, match="digest mismatch"):
        ChannelRanking.read(path)


def test_ranking_refuses_a_non_permutation():
    with pytest.raises(Exception, match="permutation"):
        ChannelRanking(
            method="t", source_dir="d", source_manifest_sha256="x", intermediate_size=I,
            ranking={0: tuple(range(I - 1))}, calibration={}, concentration={},
        )


def test_split_disjoint_halves_share_nothing():
    rank, evaluate = split_disjoint([f"p{i}" for i in range(10)])
    assert len(rank) == len(evaluate) == 5
    assert not set(rank) & set(evaluate)


# ---------------------------------------------------------------------------
# ranking split selection: location matters, not just disjointness
# ---------------------------------------------------------------------------


def test_spread_across_covers_the_whole_corpus_not_a_head_slice():
    """A contiguous ranking split cost 1.691x near it and 3.646x far away; the
    same count spread over the corpus cost 1.859x / 3.011x. So the picker must
    actually spread, and a head slice is the thing it exists to avoid."""
    from chowder.channel_importance import spread_across

    corpus = [f"p{i}" for i in range(600)]
    picked = spread_across(corpus, 32)
    assert len(picked) == 32
    positions = [corpus.index(p) for p in picked]
    assert positions == sorted(positions)
    # reaches both ends and is not a head slice
    assert positions[0] < 20 and positions[-1] > 560
    assert positions != list(range(32))
    # roughly even: no gap wildly larger than the mean stride
    gaps = [b - a for a, b in zip(positions, positions[1:])]
    assert max(gaps) <= 2 * (600 / 32)


def test_spread_across_excludes_eval_splits_and_dedups():
    from chowder.channel_importance import spread_across

    corpus = [f"p{i % 50}" for i in range(200)]   # every text repeats 4x
    held = {"p0", "p1", "p2"}
    picked = spread_across(corpus, 20, exclude=held)
    assert not (set(picked) & held), "an excluded text was selected"
    assert len(picked) == len(set(picked)), "duplicates selected"


def test_spread_across_refuses_when_everything_is_excluded():
    from chowder.channel_importance import ChannelImportanceError, spread_across

    with pytest.raises(ChannelImportanceError, match="no eligible texts"):
        spread_across(["a", "b"], 2, exclude={"a", "b"})


def test_spread_across_returns_all_when_asked_for_more_than_exists():
    from chowder.channel_importance import spread_across

    assert spread_across(["a", "b", "c"], 10) == ["a", "b", "c"]
