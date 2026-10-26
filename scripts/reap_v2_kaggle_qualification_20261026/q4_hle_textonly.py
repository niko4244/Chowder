"""Frozen-protocol REAP128_v2 Q4_K_M text-only HLE research kernel.

Scope: this is a *research* measurement of a text-only GGUF on the text-only
subset of the official, access-granted `cais/hle` test split. It is NOT an
official HLE score (the official LLM judge is never invoked), NOT Amendment 4
evidence and NOT eligible for the frozen Gen-0 scientific selector.

Guards: pinned dataset bytes/SHA, pinned protocol source SHA, pinned model
bytes/SHA, pinned server binary SHA, frozen serving geometry, host memory and
GPU floors checked before load and every 5 s while serving, single owned
process group torn down on exit, wall-clock budget below the Kaggle limit.
"""

import concurrent.futures as cf
import collections
import datetime
import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import threading
import time
import types
import urllib.request
import uuid

GIB = 1024 ** 3
ROOT = pathlib.Path('/kaggle/input')
WORK = pathlib.Path('/kaggle/working')
WORK.mkdir(parents=True, exist_ok=True)
PORT = 8791

# PROTOCOL_FILE_SHA256 pins the exact imported source bytes;
# PROTOCOL_MANIFEST_SHA256 pins the frozen protocol values it must produce.
PROTOCOL_FILE_SHA256 = 'cbacf0bb2215fce041ca62e991002d2772bee04891bc5e1773d3b0e2c44c49b2'
PROTOCOL_MANIFEST_SHA256 = 'ebdb19e818257c21335889f9bea51c9939293444d315ba0211e21d2c599e70d8'
PROTOCOL_BYTES = 13041
HLE_PARQUET_SHA256 = '6d0ee0602e8aea6b159509577e884f48ecac7b8e3f6822a35f51335a446c726a'
HLE_PARQUET_BYTES = 274276147
HLE_TEXT_ONLY_N = 2158
Q4_GGUF_SHA256 = '2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7'
Q4_GGUF_BYTES = 11563915264
LLAMA_SERVER_SHA256 = 'f10032228f1d80b0f6639cd3bcdcc7194bbc6638368641cac7f1396775b5186d'
LLAMA_SERVER_BYTES = 17888

R = {
    'schema': 'reap-v2-q4-k-m-text-only-hle-research.v1',
    'status': 'NOT_STARTED',
    'scientific_selection': 'BLOCKED',
    'chosen': None,
    'release_approved': False,
    'official_hle_score_claimed': False,
    'amendment4_gate_eligible': False,
    'protocol_sha256': PROTOCOL_MANIFEST_SHA256,
    'protocol_file_sha256': PROTOCOL_FILE_SHA256,
    'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'samples': [],
}
server = None


def save():
    p = WORK / 'reap-v2-q4-hle-textonly-research.json'
    q = p.with_name('.' + p.name + '.pending')
    with q.open('w', encoding='utf-8') as f:
        json.dump(R, f, indent=2, sort_keys=True, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(q, p)


def hash_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
        # Drop read-only page cache after hashing so a constrained cgroup can
        # still admit the CUDA model; the immutable source file is untouched.
        if hasattr(os, 'posix_fadvise') and hasattr(os, 'POSIX_FADV_DONTNEED'):
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return h.hexdigest()


def unique(pattern, label):
    matches = [p for p in ROOT.rglob(pattern) if p.is_file()]
    if len(matches) != 1:
        raise RuntimeError(f'{label}: expected exactly one source, found {len(matches)}')
    return matches[0]


def host():
    mem = {}
    with open('/proc/meminfo') as f:
        for line in f:
            k, _, v = line.partition(':')
            if v.strip().endswith('kB'):
                mem[k] = int(v.strip().split()[0]) / 1048576
    cg = pathlib.Path('/sys/fs/cgroup')
    try:
        limit = (cg / 'memory.max').read_text().strip()
        remain = (int(limit) - int((cg / 'memory.current').read_text().strip())) / GIB if limit != 'max' else None
    except Exception:
        remain = None
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True, timeout=15, check=True)
    free = [int(x.strip()) for x in gpu.stdout.splitlines()]
    return {'physical_free_gib': round(mem.get('MemAvailable', 0), 3),
            'cgroup_unused_gib': round(remain, 3) if remain is not None else None,
            'gpu_free_mib': free}


def check_runtime(h):
    if h['physical_free_gib'] < 7:
        raise RuntimeError('7 GiB free physical RAM floor breached')
    if h['cgroup_unused_gib'] is None or h['cgroup_unused_gib'] < 8:
        raise RuntimeError('8 GiB cgroup headroom floor breached')
    if len(h['gpu_free_mib']) != 2 or any(x < 4096 for x in h['gpu_free_mib']):
        raise RuntimeError('allocated T4 pair lost free VRAM')


def remove_owned():
    global server
    if server is not None and server.poll() is None:
        try:
            os.killpg(server.pid, signal.SIGTERM)
            server.wait(timeout=20)
        except Exception:
            try:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=5)
            except Exception:
                pass


try:
    protocol_path = unique('reap_hle_protocol.py', 'frozen HLE protocol source')
    if protocol_path.stat().st_size != PROTOCOL_BYTES or hash_file(protocol_path) != PROTOCOL_FILE_SHA256:
        raise RuntimeError('frozen HLE protocol source bytes/SHA mismatch')
    spec = importlib.util.spec_from_file_location('reap_hle_protocol_locked', str(protocol_path))
    P = importlib.util.module_from_spec(spec)
    sys.modules['reap_hle_protocol_locked'] = P
    spec.loader.exec_module(P)
    manifest = P.frozen_manifest()
    if manifest['protocol_sha256'] != PROTOCOL_MANIFEST_SHA256:
        raise RuntimeError('embedded and imported protocol digests disagree')
    R['protocol'] = manifest['protocol']
    if P.MAX_COMPLETION_TOKENS != 8192 or P.TEMPERATURE != 0:
        raise RuntimeError('frozen HLE decoding changed')
    flags = list(P.SERVING_FLAGS)

    parquet = unique('*.parquet', 'pinned HLE test split')
    if parquet.stat().st_size != HLE_PARQUET_BYTES or hash_file(parquet) != HLE_PARQUET_SHA256:
        raise RuntimeError('pinned HLE parquet bytes/SHA mismatch')
    import pyarrow.parquet as pq
    rows = pq.read_table(parquet, columns=['id', 'question', 'image', 'answer', 'answer_type']).to_pylist()
    text_rows = P.text_only_rows(rows)
    if len(text_rows) != HLE_TEXT_ONLY_N:
        raise RuntimeError(f'text-only HLE population changed: {len(text_rows)}')
    ordered = P.canonical_order(text_rows)[:P.PLANNED_MAX_SAMPLES]
    R['population'] = {
        'total_rows': len(rows), 'text_only_n': len(text_rows), 'planned_n': len(ordered),
        'order_rule': P.ORDER_RULE, 'wall_budget_seconds': P.WALL_BUDGET_SECONDS,
    }

    weights = unique(P.Q4_GGUF_FILENAME, 'frozen Q4_K_M GGUF')
    if weights.stat().st_size != Q4_GGUF_BYTES or hash_file(weights) != Q4_GGUF_SHA256:
        raise RuntimeError('frozen Q4_K_M GGUF bytes/SHA mismatch')

    servers = [p for p in ROOT.rglob('llama-server')
               if p.is_file() and p.stat().st_size == LLAMA_SERVER_BYTES and hash_file(p) == LLAMA_SERVER_SHA256]
    if len(servers) != 1:
        raise RuntimeError(f'pinned Linux server binary missing/ambiguous: {len(servers)}')
    livebin = WORK / 'reap-q4-hle-pinned-bin'
    if livebin.exists():
        raise RuntimeError('refusing to overwrite existing bin stage')
    shutil.copytree(servers[0].parent, livebin)
    target = livebin / 'llama-server'
    target.chmod(target.stat().st_mode | 0o111)
    env = dict(os.environ, LD_LIBRARY_PATH=str(livebin) + ':' + os.environ.get('LD_LIBRARY_PATH', ''),
               LLAMA_ARG_LOG_VERBOSITY='3')
    version = subprocess.run([str(target), '--version'], env=env, capture_output=True, text=True, timeout=60)
    R['server_version'] = {'code': version.returncode, 'stderr': version.stderr[-800:]}
    if version.returncode != 0:
        raise RuntimeError('pinned CUDA server cannot execute')

    start = host()
    R['initial'] = start
    if len(start['gpu_free_mib']) != 2 or any(x < 12288 for x in start['gpu_free_mib']):
        raise RuntimeError('HLE CUDA admission fails on allocated GPU(s)')
    if start['physical_free_gib'] < 24 or start['cgroup_unused_gib'] is None or start['cgroup_unused_gib'] < 24:
        raise RuntimeError('HLE host admission below 24 GiB')
    if shutil.disk_usage(WORK).free < 4 * GIB:
        raise RuntimeError('working disk reserve below 4 GiB')

    R['data'] = {'q4_gguf_sha256': Q4_GGUF_SHA256, 'hle_parquet_sha256': HLE_PARQUET_SHA256,
                 'text_only_n': len(text_rows), 'server_binary_sha256': LLAMA_SERVER_SHA256}
    R['cmd'] = [str(target), *flags, '--port', str(PORT), '-m', str(weights)]
    log = WORK / 'reap-v2-q4-hle-server.log'
    with log.open('xb') as fp:
        server = subprocess.Popen(R['cmd'], env=env, stdout=fp, stderr=subprocess.STDOUT, start_new_session=True)
        R['server_pid'] = server.pid
        R['status'] = 'STARTED_UNQUALIFIED'
        save()
        boot = time.monotonic()
        ready = False
        while time.monotonic() - boot < 600:
            h = host()
            check_runtime(h)
            if server.poll() is not None:
                raise RuntimeError(f'owned model server exited early: {server.returncode}')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=2) as resp:
                    if resp.status == 200:
                        ready = True
                        break
            except Exception:
                pass
            time.sleep(3)
        if not ready:
            raise RuntimeError('frozen-geometry CUDA model did not become healthy in 600 s')
        R['seconds_until_ready'] = round(time.monotonic() - boot, 1)

        def ask(item):
            body = json.dumps({
                'messages': P.build_messages(item['question']),
                'temperature': P.TEMPERATURE,
                'seed': P.SAMPLER_SEED,
                'max_tokens': P.MAX_COMPLETION_TOKENS,
                'stream': False,
            }).encode('utf-8')
            req = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/chat/completions', data=body,
                                         headers={'Content-Type': 'application/json'})
            t0 = time.monotonic()
            with urllib.request.urlopen(req, timeout=3600) as resp:
                payload = json.loads(resp.read().decode('utf-8'))
            elapsed = time.monotonic() - t0
            text = payload['choices'][0]['message']['content']
            tokens = payload.get('usage', {}).get('completion_tokens')
            if not isinstance(tokens, int) or tokens < 0:
                raise RuntimeError('serving response omitted completion token usage')
            return text, tokens, elapsed

        session = uuid.uuid4().hex
        R['session_id'] = session
        stop = threading.Event()
        fatal = []

        def owner():
            return (server is not None and server.poll() is None)

        def runtime_guard():
            while not stop.is_set():
                try:
                    h = host()
                    h['time_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    R['samples'].append(h)
                    check_runtime(h)
                    if not owner():
                        raise RuntimeError('owned model server exited during cohort')
                except Exception as ex:
                    fatal.append(str(ex))
                    remove_owned()
                    break
                stop.wait(5)

        guard = threading.Thread(target=runtime_guard, name='hle-resource-supervisor', daemon=True)
        guard.start()
        rows_path = WORK / f'reap-q4-hle-textonly-{session}.jsonl'
        summary_path = WORK / f'reap-q4-hle-textonly-{session}-summary.json'
        R['status'] = 'TEXT_ONLY_HLE_COHORT_STARTED'
        save()
        try:
            deadline = time.monotonic() + P.WALL_BUDGET_SECONDS
            scored = []
            item_errors = []
            pending = collections.deque(ordered)
            tail = collections.deque()
            with rows_path.open('x', encoding='utf-8') as handle, cf.ThreadPoolExecutor(max_workers=P.CONCURRENCY) as pool:
                inflight = {}
                while pending or inflight:
                    while pending and len(inflight) < P.CONCURRENCY and time.monotonic() < deadline:
                        item = pending.popleft()
                        inflight[pool.submit(ask, item)] = item
                    if not inflight:
                        tail = pending
                        break
                    done, _ = cf.wait(list(inflight), timeout=60, return_when=cf.FIRST_COMPLETED)
                    for fut in done:
                        if fatal:
                            raise RuntimeError('resource owner abort: ' + fatal[0])
                        item = inflight.pop(fut)
                        try:
                            text, tokens, elapsed = fut.result()
                            verdict = P.grade(text, item['answer'], item['answer_type'])
                        except Exception as ex:
                            # A single malformed item must not destroy a whole
                            # cohort, but a systemic fault still fails closed.
                            item_errors.append(f"{item['id']}: {type(ex).__name__}")
                            if len(item_errors) > 5:
                                raise RuntimeError('too many per-item scoring failures')
                            continue
                        row = {'id': item['id'], 'answer_type': item['answer_type'],
                               'correct': verdict['correct'], 'graded': verdict['graded'],
                               'truncated': tokens >= P.MAX_COMPLETION_TOKENS - 1,
                               'completion_tokens': tokens,
                               'seconds': round(elapsed, 3), 'response_chars': len(text),
                               'eval_session_id': session}
                        scored.append(row)
                        handle.write(json.dumps(row, sort_keys=True) + '\n')
                        handle.flush()
                        if len(scored) % 20 == 0:
                            os.fsync(handle.fileno())
                        if len(scored) % 25 == 0:
                            R['progress'] = {'finished': len(scored), 'planned': len(ordered)}
                            check_runtime(host())
                            save()
                            print('PROGRESS', len(scored), '/', len(ordered), flush=True)
                os.fsync(handle.fileno())
            if fatal:
                raise RuntimeError('resource owner abort: ' + fatal[0])
            if len({r['id'] for r in scored}) != len(scored):
                raise RuntimeError('duplicate completed HLE id')
            summary = P.summarize(scored, len(text_rows))
            summary['planned_n'] = len(ordered)
            summary['not_attempted_n'] = len(tail)
            summary['item_errors'] = item_errors
            summary['completion_tokens'] = sum(r['completion_tokens'] for r in scored)
            summary['generation_wall_seconds'] = round(sum(r['seconds'] for r in scored), 3)
            summary['session_id'] = session
            summary['row_file_sha256'] = hash_file(rows_path)
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8')
            R['summary'] = summary
            R['after_inference_host'] = host()
            check_runtime(R['after_inference_host'])
            R['status'] = ('COMPLETE_Q4_HLE_TEXT_ONLY_RESEARCH_NOT_OFFICIAL_JUDGE'
                           if not tail else 'BUDGET_TRUNCATED_Q4_HLE_TEXT_ONLY_RESEARCH_NOT_OFFICIAL_JUDGE')
            save()
        finally:
            stop.set()
            guard.join(timeout=15)

except Exception as e:
    R['status'] = 'REFUSED_OR_ABORTED'
    R['error'] = str(e)[:800]
    save()
finally:
    remove_owned()
    try:
        logfile = WORK / 'reap-v2-q4-hle-server.log'
        if logfile.is_file():
            data = logfile.read_bytes()
            R['server_log_sha256'] = hashlib.sha256(data).hexdigest()
            R['server_log_bytes'] = len(data)
            R['server_log_excerpt'] = data.decode('utf-8', errors='replace')[-2200:]
    except Exception:
        pass
    R['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save()
