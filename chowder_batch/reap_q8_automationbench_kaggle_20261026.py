"""Prepare a source-pinned, PRIVATE Kaggle T4x2 original-Q8 AutomationBench run.

Builds a private (not uploaded) notebook and a byte-pinned official upstream
source tar. The prime must stage that tar in a PRIVATE Kaggle dataset named by
--official-dataset, then separately authorize/launch the notebook. Neither this
generator nor importing it starts a kernel, CUDA model, or evaluator.

Q4 comparison arms have DIFFERENT native profiles and MUST remain separate:
six=64K, n-predict4096; guard2=72K with explicit decoding modifications.
Neither arm is the original frozen four-way scientific serving protocol.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile

OFFICIAL_COMMIT = "4a8e1061254004d9dac807054eed33fad7d1ff14"
PUBLIC_PROMPTS_SHA256 = "1c8feeec1f680b05b392a5d00fc634edb143c3e7bdd152c476af50cebcec57b5"
Q8_GGUF_BYTES = 19_748_435_968
Q8_GGUF_SHA256 = "89dcaf33b61f59c83a31b718ba98b73c33437d0320ae0cc56737a5d7e43519c6"
Q8_GGUF_NAME = "reap128_v2-q8_0-20261025.gguf"
Q4_GGUF_SHA256 = "2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7"
LLAMA_SERVER_BYTES = 17_888
LLAMA_SERVER_SHA256 = "f10032228f1d80b0f6639cd3bcdcc7194bbc6638368641cac7f1396775b5186d"
PRIVATE_Q4_ROOT = Path("F:/reap-v2-automationbench-private-20261026")
PRIVATE_BUILD_ROOT = Path("F:/reap-v2-q8-automationbench-private-20261026")
OFFICIAL_SOURCE = Path("F:/reap-v2-public-automationbench-source-20261026")
Q8_INPUT_DATASET = "nikmarco/reap128v2-q8-original-20261026"
TOOLCHAIN_KERNEL = "nikmarco/chowder-reap128v2-cuda-toolchain-20261025"
# Kaggle automatically extracts uploaded .tar files and drops the single
# original archive pathname. Preserve the exact Git archive bytes under a
# non-archive extension so the notebook can attest their SHA before extraction.
SOURCE_ARCHIVE_NAME = "automationbench-v1.0.6-official.tar.bin"
DOMAINS = ("sales", "marketing", "operations", "support", "finance", "hr")

# The SHA of the actual original official-result JSON, not a newly fabricated
# model score. Values are kept PRIVATE; only the SHA and task names are used.
PROFILES = {
    "six": {
        "q4_directory": "q4-ctx64k-steps44-cap4096-six-v2",
        "q4_raw_sha256": "12f48a7a60e2d2afff5f937a5f5f4d859f9d1a2ad03fec50f78bf464ab58bba8",
        "task_names": (
            "sales.advance_opportunity_stage", "sales.create_contact_for_account",
            "sales.mark_vip_emails_read", "sales.apply_project_label",
            "marketing.event_followup_personalization", "hr.visa_expiration_monitoring",
        ),
        "flags": ("--fit", "on", "-np", "1", "-c", "65536", "--n-predict", "4096", "--jinja"),
        "context": 65536,
    },
    "guard2": {
        "q4_directory": "q4-guard-smoke2-v1",
        "q4_raw_sha256": "d6c8a57a33508ad82863a982f358be6e4315cbf5ebbdd73e8ba55702788e5b6c",
        "task_names": ("sales.multi_hop_lookup", "sales.negative_selection"),
        "flags": ("--fit", "on", "-np", "1", "-c", "73728", "--repeat-penalty", "1.1",
                  "--dry-multiplier", "0.8", "--temperature", "0.7", "--jinja"),
        "context": 73728,
    },
}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _git(source: Path, *args: str, timeout: int = 60) -> bytes:
    return subprocess.run(["git", "-C", str(source), *args], check=True,
                          capture_output=True, timeout=timeout).stdout


def verify_official_checkout(source: Path) -> None:
    """Require full unmodified tracked/untracked official upstream identity."""
    if source.is_symlink() or not source.is_dir():
        raise ValueError("official checkout must be an existing real directory")
    if _git(source, "rev-parse", "HEAD").decode().strip() != OFFICIAL_COMMIT:
        raise ValueError("official v1.0.6 source commit mismatch")
    if _git(source, "status", "--porcelain=v1", "--untracked-files=all").strip():
        raise ValueError("official source changed or contains untracked shadow modules")
    for name in (".env", "sitecustomize.py", "usercustomize.py"):
        if (source / name).exists():
            raise ValueError("official checkout has local execution/credential override")
    data = (source / "pyproject.toml").read_text(encoding="utf-8")
    if not all(x in data for x in ('name = "automation-bench"', 'version = "1.0.6"',
                                   'requires-python = ">=3.13"',
                                   'auto-bench = "automationbench.scripts.eval:main"')):
        raise ValueError("official evaluator/package identity changed")
    if b"task_completed_correctly" not in (source / "automationbench/rubric/__init__.py").read_bytes():
        raise ValueError("official strict scorer missing")


def read_q4_evidence(profile: str, root: Path = PRIVATE_Q4_ROOT) -> dict:
    """Reject altered/missing real Q4 comparator without reading private text aloud."""
    if profile not in PROFILES:
        raise ValueError("unsupported matched Q4 arm")
    expected = PROFILES[profile]
    p = root / expected["q4_directory"] / "public-automationbench.private.json"
    if p.is_symlink() or not p.is_file() or p.stat().st_size > 256 * 1024 * 1024:
        raise ValueError("protected original Q4 official export unavailable/oversized")
    raw = p.read_bytes()
    if _sha(raw) != expected["q4_raw_sha256"]:
        raise ValueError("original Q4 official export SHA-256 changed")
    parsed = json.loads(raw, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    meta, summary, tasks = parsed["meta"], parsed["summary"], parsed["tasks"]
    names = [row["name"] for row in tasks]
    if (meta.get("benchmark_version") != "1.0.6" or meta.get("toolset") != "api"
            or meta.get("max_steps") != 44 or meta.get("total_tasks") != len(expected["task_names"])
            or names != list(expected["task_names"]) or summary.get("aborted_tasks") != []
            or summary.get("passed_count") != 0 or any(row.get("passed") is not False for row in tasks)):
        raise ValueError("original Q4 task identities/scorer/protocol not matched")
    manifest_path = p.parent / "experimental-manifest.private.json"
    if (manifest_path.is_symlink() or not manifest_path.is_file()
            or manifest_path.stat().st_size > 2 * 1024 * 1024):
        raise ValueError("Q4 native serving manifest missing or unbounded")
    manifest = json.loads(manifest_path.read_bytes())
    geom = manifest.get("observed_context") or {}
    model = manifest.get("model") or {}
    cuda = manifest.get("native_cuda") or {}
    source = manifest.get("official_source") or {}
    if (manifest.get("steering_flags") != list(expected["flags"])
            or geom.get("observed_total_context") != expected["context"]
            or geom.get("observed_context_per_seq") != expected["context"]
            or geom.get("observed_sequences") != 1
            or model.get("gguf_sha256") != Q4_GGUF_SHA256
            or model.get("gguf_bytes") != 11_563_915_264
            or cuda.get("layers_cuda") != "41/41"
            or cuda.get("measured_experts_offloaded") != 0
            or source.get("source_git_commit") != OFFICIAL_COMMIT):
        raise ValueError("Q4 original model/native/protocol attestation mismatch")
    return {"profile": profile, "q4_json_sha256": expected["q4_raw_sha256"],
            "q4_gcc_model_sha256": Q4_GGUF_SHA256,
            "q4_native_manifest_sha256": _sha(manifest_path.read_bytes()),
            "ordered_task_names": names, "task_count": len(names),
            "score_source": "official_public_task_completed_correctly", "q4_strict_pass_count": 0}


def validate_private_target(target: Path, *, owned_roots: tuple[Path, ...],
                            private_root: Path = PRIVATE_BUILD_ROOT) -> Path:
    if not target.is_absolute() or target.exists() or target.is_symlink():
        raise ValueError("exclusive new absolute PRIVATE output directory required")
    resolved = target.resolve(strict=False)
    boundary = private_root.resolve(strict=False)
    if not private_root.is_dir() or not boundary in resolved.parents:
        raise ValueError("PRIVATE output must be a fresh subdirectory of dedicated private root")
    if not target.parent.is_dir():
        raise ValueError("private output parent directory must already exist")
    for root in owned_roots:
        check = root.resolve(strict=False)
        if resolved == check or check in resolved.parents:
            raise ValueError("private output may not be inside source, repo or model data")
    for ancestor in (target.parent, *target.parents):
        if ancestor.exists():
            attributes = getattr(ancestor.stat(), "st_file_attributes", 0)
            if ancestor.is_symlink() or (attributes & 0x400):
                raise ValueError("private path ancestor must not be a symlink/reparse point")
    return resolved


def create_official_archive(source: Path) -> bytes:
    """Use exact Git-tree bytes; never tar local site packages or loose files."""
    verify_official_checkout(source)
    archive = _git(source, "archive", "--format=tar", "HEAD", timeout=180)
    if not archive or len(archive) > 256 * 1024 * 1024:
        raise ValueError("official tracked source archive unavailable/oversized")
    # Read-only inspect the tree. Reject symlinks and path escapes prospectively.
    import io
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        members = tar.getmembers()
        if any(m.issym() or m.islnk() or m.name.startswith("/") or
               ".." in Path(m.name).parts or not (m.isfile() or m.isdir()) for m in members):
            raise ValueError("official archived Git tree contains symlink/escape/special file")
        files = [member.name for member in members if member.isfile()]
    required = ("pyproject.toml", "uv.lock", "automationbench/scripts/eval.py",
                "automationbench/export.py", "automationbench/task_contract.py",
                "automationbench/rubric/__init__.py", "automationbench/domains/__init__.py")
    if not all(path in files for path in required):
        raise ValueError("official git archive lacks source/strict scorer")
    return archive


def _valid_slug(slug: str) -> str:
    if not re.fullmatch(r"nikmarco/[a-z0-9][a-z0-9-]{5,90}", slug):
        raise ValueError("explicit owned PRIVATE Kaggle dataset slug required")
    return slug


def build(out: Path, *, profile: str, official_dataset: str,
          source: Path = OFFICIAL_SOURCE, q4_root: Path = PRIVATE_Q4_ROOT) -> dict:
    """Write ONLY fresh PRIVATE kernel prep artifacts. Does NOT upload/execute."""
    slug = _valid_slug(official_dataset)
    proof = read_q4_evidence(profile, q4_root)
    output = validate_private_target(out, owned_roots=(source, q4_root, Path(__file__).resolve().parents[1]),
                                     private_root=PRIVATE_BUILD_ROOT)
    archive = create_official_archive(source)
    source_sha = _sha(archive)
    config = {
        "schema": "reap-v2-original-q8-automationbench-matched-public.v1",
        "profile": profile, "source_archive_name": SOURCE_ARCHIVE_NAME,
        "official_tar_sha256": source_sha, "official_commit": OFFICIAL_COMMIT,
        "public_prompt_sha256": PUBLIC_PROMPTS_SHA256,
        "original_q8_filename": Q8_GGUF_NAME, "original_q8_bytes": Q8_GGUF_BYTES,
        "original_q8_sha256": Q8_GGUF_SHA256, "native_server_sha256": LLAMA_SERVER_SHA256,
        "native_server_bytes": LLAMA_SERVER_BYTES, "native_flags": list(PROFILES[profile]["flags"]),
        "expected_context": PROFILES[profile]["context"],
        "matched_q4_official_json_sha256": proof["q4_json_sha256"],
        "matched_q4_native_manifest_sha256": proof["q4_native_manifest_sha256"],
        "tasks": proof["ordered_task_names"], "expected_task_count": proof["task_count"],
        "max_steps": 44, "toolset": "api", "max_concurrent": 1,
        "official_private_leaderboard": False, "gen0_status": "BLOCKED",
    }
    text = KERNEL_TEMPLATE.replace("__LOCKED_CONFIG_LITERAL__", repr(json.dumps(config, sort_keys=True)))
    if "__LOCKED_CONFIG_LITERAL__" in text:
        raise ValueError("private notebook template config replacement failed")
    ast.parse(text, filename="private_q8_automationbench_kernel.py")
    code = compile(text, "private_q8_automationbench_kernel.py", "exec")
    if code is None:
        raise ValueError("notebook source could not compile")
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    (output / SOURCE_ARCHIVE_NAME).write_bytes(archive)
    notebook = output / f"reap128v2-q8-automationbench-{profile}-20261026.py"
    notebook.write_text(text, encoding="utf-8", newline="\n")
    kaggle_id = f"nikmarco/reap128v2-q8-automationbench-{profile}-20261026"
    metadata = {
        "id": kaggle_id, "title": f"reap128v2 q8 automationbench {profile} private 20261026",
        "code_file": notebook.name, "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [Q8_INPUT_DATASET, slug], "kernel_sources": [TOOLCHAIN_KERNEL],
        "competition_sources": [], "model_sources": [],
    }
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {"status": "PRIVATE_Q8_KAGGLE_SOURCE_PREPARED_UNLAUNCHED", "kaggle_id": kaggle_id,
            "mode": profile, "original_q8_sha256": Q8_GGUF_SHA256,
            "official_commit": OFFICIAL_COMMIT, "official_source_archive_sha256": source_sha,
            "notebook_sha256": _sha(notebook.read_bytes()),
            "matching_q4_json_sha256": proof["q4_json_sha256"],
            "matched_count": proof["task_count"], "private_output": str(output),
            "must_upload_official_archive_as_private_dataset": slug,
            "never_claim_private_leaderboard": True, "gen0_status": "BLOCKED"}


# The script is authored as source (never executed by the generator). It runs
# the OFFICIAL evaluator, not a model-created scorer or synthetic responses.
KERNEL_TEMPLATE = r'''"""PRIVATE Q8 AutomationBench official-public research kernel; no Gen-0 release."""
import datetime, gc, hashlib, io, json, os, pathlib, re, shutil, signal
import subprocess, sys, tarfile, threading, time, urllib.request, uuid

CFG = json.loads(__LOCKED_CONFIG_LITERAL__)
ROOT = pathlib.Path('/kaggle/input')
WORK = pathlib.Path('/kaggle/working')
WORK.mkdir(exist_ok=True)
GIB = 1024**3
PORT = 8795
SESSION = uuid.uuid4().hex
R = {'schema': CFG['schema'], 'status': 'NOT_STARTED', 'session_id': SESSION,
     'profile': CFG['profile'], 'official_commit': CFG['official_commit'],
     'q8_sha256': CFG['original_q8_sha256'], 'q4_reference_json_sha256': CFG['matched_q4_official_json_sha256'],
     'gen0_status': 'BLOCKED', 'chosen': None, 'release_approved': False,
     'official_private_leaderboard_comparable': False, 'samples': []}
server = None
evaluator = None
stop = threading.Event()
fatal = []

def save():
    dest = WORK / ('reap-q8-automationbench-' + CFG['profile'] + '-private-status.json')
    tmp = dest.with_suffix('.pending')
    with tmp.open('x', encoding='utf8') as f:
        json.dump(R, f, sort_keys=True, indent=2, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, dest)

def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb', buffering=0) as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
        if not hasattr(os, 'posix_fadvise') or not hasattr(os, 'POSIX_FADV_DONTNEED'):
            raise RuntimeError('clean file-cache advisory unavailable')
        os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return h.hexdigest()

def unique(name):
    matches = [p for p in ROOT.rglob(name) if p.is_file() and not p.is_symlink()]
    if len(matches) != 1: raise RuntimeError('unique mounted source missing/ambiguous: '+name)
    return matches[0]

def snapshot():
    mem = {}
    for line in pathlib.Path('/proc/meminfo').read_text().splitlines():
        key, sep, value = line.partition(':')
        if sep and value.strip().endswith('kB'):
            mem[key] = int(value.strip().split()[0]) / 1048576
    cg = pathlib.Path('/sys/fs/cgroup')
    maximum = (cg/'memory.max').read_text().strip()
    if maximum == 'max': raise RuntimeError('finite cgroup admission unavailable')
    unused = (int(maximum) - int((cg/'memory.current').read_text().strip())) / GIB
    process = subprocess.run(['nvidia-smi','--query-gpu=name,memory.free,memory.total',
                             '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=15)
    devices = [x.split(',') for x in process.stdout.strip().splitlines()]
    if len(devices) != 2 or any('T4' not in x[0] or len(x)!=3 for x in devices):
        raise RuntimeError('exact two NVIDIA T4 cards not present')
    gpu = [int(x[1].strip()) for x in devices]
    return {'ram': mem.get('MemAvailable',0), 'cgroup': unused, 'gpu':gpu,
            'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}

def gpu_compute_pids():
    probe = subprocess.run(['nvidia-smi', '--query-compute-apps=pid',
                            '--format=csv,noheader,nounits'],check=True,
                            capture_output=True,text=True,timeout=15)
    lines = [x.strip() for x in probe.stdout.splitlines() if x.strip()]
    if any(not item.isdigit() for item in lines):
        raise RuntimeError('NVIDIA process census unsupported/ambiguous')
    return {int(x) for x in lines}

def check_resources(s, *, prelaunch=False):
    if s['ram'] < (24 if prelaunch else 7): raise RuntimeError('physical RAM admission/refusal')
    if s['cgroup'] < (29.443 if prelaunch else 10): raise RuntimeError('Q8 finite cgroup headroom refusal')
    floor = 12288 if prelaunch else 2048
    if len(s['gpu']) != 2 or any(v < floor for v in s['gpu']):
        raise RuntimeError('T4 VRAM admission/refusal')
    if shutil.disk_usage(WORK).free < 8 * GIB: raise RuntimeError('private result disk floor breached')

def kill_owned(proc):
    if proc is None or proc.poll() is not None: return
    try:
        os.killpg(proc.pid, signal.SIGTERM); proc.wait(timeout=15)
    except Exception:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL); proc.wait(timeout=10)

def require_offload(raw):
    if not re.search(r'offloaded\s+41/41\s+layers\s+to\s+GPU',raw,re.I):
        raise RuntimeError('native full CUDA 41/41 witness absent')
    if 'successfully fit' not in raw.lower():
        raise RuntimeError('native model/experts GPU fit witness absent')
    cpu_moe = [int(x) for x in re.findall(r'n_cpu_moe\s*[=:]?\s*(\d+)',raw,re.I)]
    overflow = [int(x) for x in re.findall(r'\((\d+) overflowing\)',raw,re.I)]
    if any(cpu_moe) or any(overflow) or re.search(r'blk\.\d+\..*exps.*CPU',raw,re.I):
        raise RuntimeError('observed routed expert CPU offload')
    if not re.search(r'n_ctx\s*=\s*'+str(CFG['expected_context'])+r'\b',raw):
        raise RuntimeError('native context differs from registered Q4 reference')
    if not re.search(r'n_ctx_seq\s*=\s*'+str(CFG['expected_context'])+r'\b',raw):
        raise RuntimeError('native per-sequence context mismatch')
    if not re.search(r'n_seq_max\s*=\s*1\b',raw):
        raise RuntimeError('native concurrency mismatch')

def stage_source(archive):
    source = WORK / 'automationbench-pinned-official-source'
    source.mkdir(mode=0o700, exist_ok=False)
    with tarfile.open(archive, mode='r:') as tf:
        for member in tf:
            relative = pathlib.PurePosixPath(member.name)
            if (relative.is_absolute() or '..' in relative.parts or member.issym() or member.islnk()
                    or not (member.isfile() or member.isdir())):
                raise RuntimeError('official source archive path escape/link')
            path = source.joinpath(*relative.parts)
            if member.isdir(): path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with tf.extractfile(member) as source_file, path.open('xb') as target:
                    shutil.copyfileobj(source_file,target, 1024*1024)
    meta = (source/'pyproject.toml').read_text(encoding='utf8')
    if not all(x in meta for x in ('name = "automation-bench"','version = "1.0.6"',
                                   'requires-python = ">=3.13"',
                                   'auto-bench = "automationbench.scripts.eval:main"')):
        raise RuntimeError('pinned official package identity/version missing')
    return source

def find_offline_python(source):
    # Explicitly refuse if Kaggle does not have a pre-provisioned, OFFLINE
    # Python3.13 runtime with official verifiers/datasets dependencies.
    candidates = [shutil.which('python3.13'),'/opt/python3.13/bin/python3.13']
    for exe in candidates:
        if not exe or not pathlib.Path(exe).is_file(): continue
        v = subprocess.run([exe,'-E','-s','-c',
            'import sys,verifiers,datasets,pydantic,httpx; print(sys.version_info[:2])'],
            cwd=source,capture_output=True,text=True,timeout=50)
        if v.returncode == 0 and '(3, 13)' in v.stdout:
            return exe
    raise RuntimeError('BLOCKED: offline Python3.13 + pinned official dependencies unavailable (do not pip/network install)')

def prove_population(exe, source):
    code = ('from automationbench.domains import get_combined_dataset,DEFAULT_DOMAINS;'
            'from automationbench.task_contract import task_contract_sha256;'
            'import json,hashlib; d=get_combined_dataset(DEFAULT_DOMAINS);'
            'raw=json.dumps([(str(x["example_id"]),str(x["prompt"])) for x in d],'
            'ensure_ascii=False,separators=(",", ":")).encode();'
            'wanted=set(json.loads('+repr(json.dumps(CFG['tasks']))+'));'
            'selected=[x for x in d if json.loads(x["info"])["task_name"] in wanted];'
            'proof=[{"name":json.loads(x["info"])["task_name"],'
            '"example_id":x["example_id"],'
            '"contract":task_contract_sha256(example_id=x["example_id"],'
            'prompt=x["prompt"],info=json.loads(x["info"]))} for x in selected];'
            'print(json.dumps({"n":len(d),"prompt_sha":hashlib.sha256(raw).hexdigest(),'
            '"domains":DEFAULT_DOMAINS,"selected":proof}))')
    output = subprocess.run([exe,'-E','-s','-c',code],cwd=source,env=limited_env(source),
                            capture_output=True,text=True,timeout=900)
    if output.returncode: raise RuntimeError('pinned official full-600 source loader failed before CUDA load')
    result = json.loads(output.stdout)
    if (result.get('n') != 600 or result.get('prompt_sha') != CFG['public_prompt_sha256']
            or result.get('domains') != ['sales','marketing','operations','support','finance','hr']
            or [x['name'] for x in result['selected']] != CFG['tasks']):
        raise RuntimeError('source prompt/population/order does not match Q4 exact selected tasks')
    R['official_task_contracts'] = result['selected']

def limited_env(source):
    allowed = ('PATH','HOME','TMPDIR','TMP','LANG','LC_ALL','LD_LIBRARY_PATH')
    env = {k:os.environ[k] for k in allowed if k in os.environ}
    env.update({'OPENAI_API_KEY':'local-only', 'PYTHON_DOTENV_DISABLED':'1',
                'PYTHONNOUSERSITE':'1','HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1',
                'PYTHONPATH':str(source),'PYTHONDONTWRITEBYTECODE':'1'})
    return env

try:
    R['status'] = 'VALIDATING_UPSTREAM_OFFLINE_BEFORE_GPU'
    save()
    archive = unique(CFG['source_archive_name'])
    if sha_file(archive) != CFG['official_tar_sha256']:
        raise RuntimeError('official source tar source hash mismatch')
    official = stage_source(archive)
    interpreter = find_offline_python(official)
    prove_population(interpreter,official)
    gc.collect()
    initial = snapshot()
    check_resources(initial, prelaunch=True)
    if gpu_compute_pids():
        raise RuntimeError('protected active GPU compute process at prelaunch')
    R['prelaunch'] = initial
    weights = unique(CFG['original_q8_filename'])
    if weights.stat().st_size != CFG['original_q8_bytes'] or sha_file(weights) != CFG['original_q8_sha256']:
        raise RuntimeError('original Q8 GGUF full SHA/byte size mismatch')
    native_candidates = [p for p in ROOT.rglob('llama-server') if p.is_file() and not p.is_symlink()
                         and p.stat().st_size == CFG['native_server_bytes']
                         and sha_file(p) == CFG['native_server_sha256']]
    if len(native_candidates) != 1:
        raise RuntimeError('pinned original native binary is missing or ambiguous')
    local_bin = WORK/'reap-q8-automationbench-native-binary'
    shutil.copytree(native_candidates[0].parent, local_bin)
    executable = local_bin/'llama-server'
    executable.chmod(executable.stat().st_mode | 0o111)
    env = limited_env(official)
    env['LD_LIBRARY_PATH'] = str(local_bin) + ':' + env.get('LD_LIBRARY_PATH','')
    env['LLAMA_ARG_LOG_VERBOSITY'] = '4'
    R['native_argv'] = [str(executable),*CFG['native_flags'],'--host','127.0.0.1',
                        '--port',str(PORT),'-m',str(weights)]
    start_log = WORK / ('reap-q8-automationbench-'+CFG['profile']+'-native.private.log')
    with start_log.open('xb') as stream:
        server = subprocess.Popen(R['native_argv'],stdout=stream,stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL,env=env,cwd=WORK,start_new_session=True)
    R['native_pid'] = server.pid
    R['status'] = 'GPU_STARTING_NOT_SCORED'
    save()
    start = time.monotonic()
    ready = False
    while time.monotonic()-start < 600:
        check_resources(snapshot())
        if server.poll() is not None: raise RuntimeError('owned Q8 server exited during load')
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health',timeout=2) as response:
                if response.status == 200: ready=True; break
        except (OSError,TimeoutError): pass
        time.sleep(3)
    if not ready: raise RuntimeError('Q8 native model never reported healthy')
    # Q8 HLE validated this nonmutating advisory. Preserve cgroup 10-GiB
    # stop and require 14-GiB unused before scoring after reclaim.
    R['before_postload_cache_reclaim'] = snapshot()
    with weights.open('rb',buffering=0) as f:
        os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
    time.sleep(5)
    postload = snapshot()
    check_resources(postload)
    if postload['cgroup'] < 14: raise RuntimeError('postload Q8 <14GiB unused cgroup')
    R['after_postload_cache_reclaim'] = postload
    native_text = start_log.read_bytes().decode('utf8', errors='replace')
    require_offload(native_text)
    if gpu_compute_pids() != {server.pid}:
        raise RuntimeError('GPU compute process is not uniquely owned by Q8 native server')
    R['startup_native_log_sha256'] = sha_file(start_log)
    with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/v1/models',timeout=5) as res:
        model_list=json.loads(res.read(65537))
    objects=model_list.get('data') if isinstance(model_list,dict) else None
    if not isinstance(objects,list) or len(objects)!=1:
        raise RuntimeError('native model inventory is not unique')
    served_id=objects[0].get('id')
    if served_id not in (weights.name,str(weights),weights.stem):
        raise RuntimeError('native served model ID not original Q8 filename/path')
    result = WORK / ('reap-q8-automationbench-'+CFG['profile']+'-official-private.json')
    command=[interpreter,'-E','-s','-m','automationbench.scripts.eval',
             '--model',served_id,'--base-url',f'http://127.0.0.1:{PORT}/v1',
             '--api','chat_completions','--api-key-var','OPENAI_API_KEY',
             '--toolset','api','--domains','all','--tasks',','.join(CFG['tasks']),
             '--num-examples',str(CFG['expected_task_count']),
             '--max-concurrent','1','--max-steps',str(CFG['max_steps']),
             '--no-ensure-complete','--export-json',str(result)]
    R['official_argv'] = command
    R['status'] = 'OFFICIAL_PUBLIC_EVALUATOR_RUNNING_UNVERIFIED'
    save()
    official_log = WORK / ('reap-q8-automationbench-'+CFG['profile']+'-evaluator.private.log')
    with official_log.open('xb') as stream:
        evaluator=subprocess.Popen(command,env=limited_env(official),cwd=official,
                                   stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                                   start_new_session=True)
    end=time.monotonic()+3*3600
    while evaluator.poll() is None:
        state=snapshot()
        R['samples'].append(state)
        check_resources(state)
        if server.poll() is not None: raise RuntimeError('owned native server exited during scoring')
        if time.monotonic() > end: raise RuntimeError('Kaggle evaluator wall-budget exceeded')
        time.sleep(5)
    if evaluator.returncode: raise RuntimeError('OFFICIAL evaluator failed; NO score')
    if not result.is_file() or result.stat().st_size > 256*1024*1024:
        raise RuntimeError('official JSON missing or unbounded')
    parsed=json.loads(result.read_text(encoding='utf8'))
    tasks=parsed.get('tasks')
    meta=parsed.get('meta') or {}
    summary=parsed.get('summary') or {}
    if (not isinstance(tasks,list) or len(tasks)!=len(CFG['tasks'])
            or [x.get('name') for x in tasks]!=CFG['tasks']
            or meta.get('model')!=served_id or meta.get('benchmark_version')!='1.0.6'
            or meta.get('toolset')!='api' or meta.get('max_steps')!=44
            or meta.get('total_tasks')!=len(CFG['tasks']) or summary.get('aborted_tasks')!=[]
            or meta.get('domains')!=list(['sales','marketing','operations','support','finance','hr'])
            or meta.get('partial') is not None
            or not isinstance(parsed.get('usage_by_task'),list)
            or len(parsed['usage_by_task'])!=len(tasks)
            or any(type(x.get('passed')) is not bool or x['passed']!=(x.get('score')==1.0)
                   for x in tasks)):
        raise RuntimeError('official output incomplete, aborted or source identity mismatch')
    if (type(summary.get('passed_count')) is not int
            or summary['passed_count'] != sum(int(x['passed']) for x in tasks)
            or type(summary.get('failed_count')) is not int
            or summary['failed_count'] != len(tasks)-summary['passed_count']):
        raise RuntimeError('official binary pass and denominator summary mismatch')
    for i,(task,usage) in enumerate(zip(tasks,parsed['usage_by_task']),1):
        if (type(task.get('id')) is not int or task['id'] != i
                or type(usage.get('task_id')) is not int or usage['task_id'] != i
                or usage.get('task_name') != task['name']):
            raise RuntimeError('official per-task usage identity mismatch')
        msgs=task.get('messages')
        if (not isinstance(msgs,list) or not msgs or
                (msgs[-1].get('role')=='assistant' and msgs[-1].get('tool_calls')
                 and task.get('steps',0) < 42)):
            raise RuntimeError('official evaluator returned an unscored aborted task')
    R['official_result_sha256'] = sha_file(result)
    R['requested_tasks'] = len(tasks)
    R['raw_binary_passed_count'] = summary['passed_count']
    R['status'] = 'RAW_Q8_OFFICIAL_PUBLIC_TASKS_REQUIRE_INDEPENDENT_ASSERTION_AUDIT'
except Exception as error:
    R['status'] = 'REFUSED_OR_ABORTED_UNSCORED'
    R['error_type'] = type(error).__name__
    R['error'] = str(error)[:500]
finally:
    try: kill_owned(evaluator)
    finally: kill_owned(server)
    R['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    save()
'''


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--profile", choices=tuple(PROFILES), required=True)
    ap.add_argument("--official-dataset", required=True,
                    help="Existing PRIVATE Kaggle dataset slug that will contain the generated source archive")
    ap.add_argument("--official-source", type=Path, default=OFFICIAL_SOURCE)
    opts = ap.parse_args()
    print(json.dumps(build(opts.out, profile=opts.profile, official_dataset=opts.official_dataset,
                           source=opts.official_source), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
