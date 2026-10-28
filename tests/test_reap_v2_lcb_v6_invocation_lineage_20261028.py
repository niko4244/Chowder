"""CPU-only log lineage tests; no model or candidate-code execution."""

import pytest

from chowder_batch import reap_v2_lcb_v6_invocation_lineage_20261028 as lineage


def _rows():
    return [
        {"question_id": "a", "code": "x", "finish_reason": "stop",
         "elapsed_s": 1.0, "error": None},
        {"question_id": "b", "code": "xy", "finish_reason": "stop",
         "elapsed_s": 2.5, "error": None},
        {"question_id": "c", "code": "", "finish_reason": None,
         "elapsed_s": 18.1, "error": "URLError: connection refused"},
        {"question_id": "d", "code": "z", "finish_reason": "length",
         "elapsed_s": 3.25, "error": None},
    ]


def _log():
    return (
        "total_v6=4 already=1 settings temp=0.2 top_p=0.95 "
        "max_tokens=4096 style=OpenAIChat\n"
        "[2/4] b finish=stop code=2 2.5s err=None\n"
        "[3/4] c finish=None code=0 18.1s err=URLError: connection refused\n"
        "[4/4] d finish=length code=1 3.25s err=None\n"
        "GEN_DONE\n"
    ).encode("utf-8")


def test_source_correlated_boundary_does_not_attest_preexisting_session():
    result = lineage.audit_invocation_log(_log(), _rows(), expected_population=4)
    assert result["preexisting_journal_rows_session_unattributed"] == 1
    assert result["observed_invocation_appended_rows"] == 3
    assert result["observed_invocation_model_responses_unattested"] == 2
    assert result["observed_invocation_transport_failures_unscored"] == 1
    assert result["preexisting_rows_same_model_or_session_proven"] is False
    assert result["invocation_model_session_independently_attested"] is False
    assert result["cross_session_pooling_allowed"] is False
    assert result["official_score"] is None
    assert result["frozen_gen0"] == {
        "status": "BLOCKED", "chosen": None, "release_approved": False,
    }


@pytest.mark.parametrize("bad", [
    lambda raw: raw.replace(b"already=1", b"already=0"),
    lambda raw: raw.replace(b"temp=0.2", b"temp=0.0"),
    lambda raw: raw.replace(b"[2/4]", b"[1/4]"),
    lambda raw: raw.replace(b"[3/4] c", b"[3/4] wrong"),
    lambda raw: raw.replace(b"code=2", b"code=3"),
    lambda raw: raw.replace(b"2.5s", b"2.6s"),
    lambda raw: raw.replace(b"finish=length", b"finish=stop"),
    lambda raw: raw.replace(b"connection refused", b"timeout"),
    lambda raw: raw.replace(b"GEN_DONE\n", b""),
    lambda raw: raw + b"extra\n",
    lambda raw: raw.replace(b"[4/4]", b"[4/5]"),
    lambda raw: raw.replace(b"[4/4]", b"[4/4] d"),
])
def test_malformed_or_forged_invocation_log_refused(bad):
    with pytest.raises(ValueError):
        lineage.audit_invocation_log(bad(_log()), _rows(), expected_population=4)


def test_truncated_log_invalid_utf8_and_wrong_population_refused():
    for raw in (b"", b"\xff", _log().split(b"GEN_DONE")[0]):
        with pytest.raises(ValueError):
            lineage.audit_invocation_log(raw, _rows(), expected_population=4)
    with pytest.raises(ValueError, match="canonical population"):
        lineage.audit_invocation_log(_log(), _rows(), expected_population=1055)


@pytest.mark.parametrize("bad", [
    lambda raw: raw.replace(b"\n", b"\r"),
    lambda raw: raw.replace(b"\n", b"\x0b"),
    lambda raw: raw.replace(b"\n", b"\xe2\x80\xa8"),
    lambda raw: raw.replace(b"\n", b"\r\n", 1),
    lambda raw: raw.replace(b"\n", b"\r\n")[:-2],
    lambda raw: b"\xef\xbb\xbf" + raw,
    lambda raw: raw.replace(b"\n", b"\r\r\n"),
])
def test_nonoriginal_log_framing_must_not_establish_invocation_lineage(bad):
    with pytest.raises(ValueError, match="framing|UTF-8|newline"):
        lineage.audit_invocation_log(bad(_log()), _rows(), expected_population=4)


def test_original_windows_crlf_log_framing_is_accepted():
    log = _log().replace(b"\n", b"\r\n")
    result = lineage.audit_invocation_log(log, _rows(), expected_population=4)
    assert result["observed_invocation_appended_rows"] == 3
