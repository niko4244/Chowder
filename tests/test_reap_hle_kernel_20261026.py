"""Drift and honesty guards for the launch-ready Q4 text-only HLE kernel.

These tests never load a model, never touch HLE content and never start a
kernel. They pin the kernel's declared constants to the frozen protocol module
so the two cannot diverge silently before a real run.
"""

import hashlib
import pathlib
import py_compile
import re

import pytest

from chowder_batch import reap_hle_protocol as P

KERNEL = pathlib.Path(__file__).resolve().parents[1] / 'scripts' / 'reap_v2_kaggle_qualification_20261026' / 'q4_hle_textonly.py'


@pytest.fixture(scope='module')
def source():
    return KERNEL.read_text(encoding='utf-8')


def constant(source, name):
    match = re.search(rf"^{name}\s*=\s*'([^']*)'$", source, re.MULTILINE)
    assert match, f'{name} not found as a pinned constant'
    return match.group(1)


def constant_int(source, name):
    match = re.search(rf"^{name}\s*=\s*(\d[\d_]*)$", source, re.MULTILINE)
    assert match, f'{name} not found as a pinned integer constant'
    return int(match.group(1).replace('_', ''))


def test_kernel_compiles():
    py_compile.compile(str(KERNEL), doraise=True)


@pytest.mark.parametrize(
    'name,value',
    [
        ('HLE_PARQUET_SHA256', P.HLE_PARQUET_SHA256),
        ('Q4_GGUF_SHA256', P.Q4_GGUF_SHA256),
        ('LLAMA_SERVER_SHA256', P.PINNED_LLAMA_SERVER_SHA256),
    ],
)
def test_kernel_pins_match_the_frozen_protocol(source, name, value):
    assert constant(source, name) == value


def test_kernel_numeric_pins_match_the_frozen_protocol(source):
    for name, value in [('HLE_PARQUET_BYTES', P.HLE_PARQUET_BYTES),
                        ('Q4_GGUF_BYTES', P.Q4_GGUF_BYTES),
                        ('HLE_TEXT_ONLY_N', 2158)]:
        assert constant_int(source, name) == value


def test_protocol_source_pin_matches_the_real_file(source):
    data = pathlib.Path(P.__file__).read_bytes()
    assert constant(source, 'PROTOCOL_FILE_SHA256') == hashlib.sha256(data).hexdigest()
    assert constant_int(source, 'PROTOCOL_BYTES') == len(data)


def test_kernel_embeds_the_current_manifest_digest(source):
    manifest = P.frozen_manifest()['protocol_sha256']
    assert constant(source, 'PROTOCOL_MANIFEST_SHA256') == manifest
    assert "'protocol_sha256': PROTOCOL_MANIFEST_SHA256" in source
    assert "'protocol_file_sha256': PROTOCOL_FILE_SHA256" in source
    assert 'PROTOCOL_MANIFEST_SHA256' != 'PROTOCOL_FILE_SHA256'
    # The two pins must not be interchangeable.
    assert constant(source, 'PROTOCOL_FILE_SHA256') != constant(source, 'PROTOCOL_MANIFEST_SHA256')


def test_kernel_uses_frozen_protocol_serving_and_decoding(source):
    assert 'P.SERVING_FLAGS' in source
    assert 'P.MAX_COMPLETION_TOKENS' in source
    assert 'P.TEMPERATURE' in source
    assert 'P.canonical_order' in source
    assert 'P.text_only_rows' in source
    # No private literal geometry may override the declared protocol.
    assert "'-np', '4'" not in source and '"-np", "4"' not in source


def test_kernel_refuses_to_claim_official_or_gate_eligibility(source):
    lowered = source.lower()
    assert "'official_hle_score_claimed': False" in source
    assert "'amendment4_gate_eligible': False" in source
    assert "'release_approved': False" in source
    assert "'chosen': None" in source
    assert 'llm judge is never invoked' in lowered


def test_kernel_does_not_persist_questions_or_reference_answers(source):
    block = re.search(r'row = \{(.*?)\}', source, re.S)
    assert block, 'scored row literal not found'
    keys = set(re.findall(r"'([a-z_]+)':", block.group(1)))
    assert keys == {'id', 'answer_type', 'correct', 'completion_tokens', 'seconds',
                    'response_chars', 'eval_session_id'}
    assert 'question' not in keys and 'answer' not in keys


def test_kernel_budget_stays_below_the_kaggle_session_limit(source):
    assert P.WALL_BUDGET_SECONDS <= 11 * 3600
    assert 'P.WALL_BUDGET_SECONDS' in source


def test_kernel_tears_down_its_own_process_group(source):
    assert 'start_new_session=True' in source
    assert 'os.killpg(server.pid, signal.SIGTERM)' in source
    assert 'pkill' not in source and 'taskkill' not in source
