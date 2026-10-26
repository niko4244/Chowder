"""Tests for the HLE collection auditor.

The happy path needs the pinned private parquet, so it is skipped when that file
is absent. The refusal paths only need the auditor's own logic but still load the
population, so they share the same skip. No question or answer text is used.
"""

import hashlib
import json
import pathlib

import pytest

from chowder_batch import reap_hle_protocol as P
from scripts.reap_v2_hle_collect import audit, population

PARQUET = pathlib.Path(
    'C:/Users/nikma/Documents/Codex/2026-10-26/continue-the-complete-chowder-reap128-v2/'
    'work/hle-private/data/test-00000-of-00001.parquet'
)

pytestmark = pytest.mark.skipif(not PARQUET.is_file(), reason='pinned private HLE parquet not present')


def build(tmp_path, n=6, correct=True, tokens=200):
    planned = population(PARQUET)
    rows = []
    for index, item in enumerate(planned[:n]):
        rows.append({
            'id': item['id'],
            'answer_type': item['answer_type'],
            'correct': correct,
            'graded': True,
            'truncated': tokens >= P.MAX_COMPLETION_TOKENS - 1,
            'completion_tokens': tokens,
            'seconds': 2.5 + index,
            'response_chars': 300,
            'eval_session_id': 'a' * 32,
        })
    rows_path = tmp_path / 'reap-q4-hle-textonly-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.jsonl'
    rows_path.write_text('\n'.join(json.dumps(r) for r in rows) + '\n', encoding='utf-8')
    summary = P.summarize(rows, len(planned))
    summary.update({
        'population_text_only_n': len(planned),
        'row_file_sha256': hashlib.sha256(rows_path.read_bytes()).hexdigest(),
        'eval_session_id': 'a' * 32,
        'not_attempted_n': 0,
        'item_errors': [],
        'completion_tokens': sum(r['completion_tokens'] for r in rows),
        'generation_wall_seconds': 1.0,
        'session_id': 'a' * 32,
        'planned_n': len(planned),
    })
    receipt = {
        'schema': 'reap-v2-q4-k-m-text-only-hle-research.v1',
        'status': 'COMPLETE_Q4_HLE_TEXT_ONLY_RESEARCH_NOT_OFFICIAL_JUDGE',
        'protocol_sha256': P.frozen_manifest()['protocol_sha256'],
        'session_id': 'a' * 32,
        'official_hle_score_claimed': False,
        'amendment4_gate_eligible': False,
        'chosen': None,
        'release_approved': False,
        'summary': summary,
    }
    (tmp_path / 'reap-v2-q4-hle-textonly-research.json').write_text(json.dumps(receipt), encoding='utf-8')
    return rows, summary, receipt


def rewrite(tmp_path, rows, summary, receipt):
    rows_path = tmp_path / 'reap-q4-hle-textonly-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.jsonl'
    rows_path.write_text('\n'.join(json.dumps(r) for r in rows) + '\n', encoding='utf-8')
    summary['row_file_sha256'] = hashlib.sha256(rows_path.read_bytes()).hexdigest()
    (tmp_path / 'reap-v2-q4-hle-textonly-research.json').write_text(json.dumps(receipt), encoding='utf-8')


def test_clean_run_is_accepted_and_rederived(tmp_path):
    build(tmp_path)
    report = audit(tmp_path, PARQUET)
    assert report['problems'] == [], report['problems']
    assert report['status'] == 'AUDIT_PASSED_DETERMINISTIC_NON_OFFICIAL'
    assert report['derived']['graded_n'] == 6
    assert report['derived']['accuracy'] == 1.0
    assert report['official_hle_score'] == 'UNAVAILABLE'
    assert report['chosen'] is None
    assert report['derived_stats']['rows_checked'] == 6
    assert report['derived_stats']['total_completion_tokens'] == 1200


def test_summary_that_disagrees_with_rows_is_refused(tmp_path):
    rows, summary, receipt = build(tmp_path)
    summary['correct'] = 5          # claim a different outcome than the rows show
    receipt['summary'] = summary
    rewrite(tmp_path, rows, summary, receipt)
    report = audit(tmp_path, PARQUET)
    assert report['status'] == 'AUDIT_REFUSED'
    assert any('correct' in p for p in report['problems'])


def test_row_over_the_frozen_token_ceiling_is_refused(tmp_path):
    rows, summary, receipt = build(tmp_path)
    rows[0]['completion_tokens'] = P.MAX_COMPLETION_TOKENS + 1
    summary.update(P.summarize(rows, 2158))
    receipt['summary'] = summary
    rewrite(tmp_path, rows, summary, receipt)
    report = audit(tmp_path, PARQUET)
    assert report['status'] == 'AUDIT_REFUSED'
    assert any('token' in p for p in report['problems'])


def test_foreign_session_is_refused(tmp_path):
    rows, summary, receipt = build(tmp_path)
    rows[2]['eval_session_id'] = 'b' * 32
    rewrite(tmp_path, rows, summary, receipt)
    report = audit(tmp_path, PARQUET)
    assert report['status'] == 'AUDIT_REFUSED'
    assert any('foreign session' in p for p in report['problems'])


def test_out_of_order_rows_are_refused(tmp_path):
    rows, summary, receipt = build(tmp_path)
    rows[0], rows[1] = rows[1], rows[0]
    rewrite(tmp_path, rows, summary, receipt)
    report = audit(tmp_path, PARQUET)
    assert report['status'] == 'AUDIT_REFUSED'
    assert any('canonical order' in p for p in report['problems'])


def test_incomplete_kernel_status_is_refused(tmp_path):
    rows, summary, receipt = build(tmp_path)
    receipt['status'] = 'TEXT_ONLY_HLE_COHORT_STARTED'
    rewrite(tmp_path, rows, summary, receipt)
    report = audit(tmp_path, PARQUET)
    assert report['status'] == 'AUDIT_REFUSED'
    assert any('not a completed run' in p for p in report['problems'])


def test_selection_claim_is_refused(tmp_path):
    rows, summary, receipt = build(tmp_path)
    receipt['chosen'] = 'reap128_v2_q4'
    rewrite(tmp_path, rows, summary, receipt)
    report = audit(tmp_path, PARQUET)
    assert report['status'] == 'AUDIT_REFUSED'
    assert any('selection' in p for p in report['problems'])
