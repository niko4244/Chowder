"""External VM-host witness for ONE fixed benign HumanEval-shaped canary.

No original HumanEval candidates, generated programs, scores or model data.
Only an existing, SHA-pinned rootless Podman image may be used; no image pull,
VM start/stop, system service mutation, GPU, WSL commands, or host bind mounts.
The VM's *separate host Python process* reads the container process's /proc
namespace, network, cgroup, executable and complete process identity; a child
printout and Podman JSON are never sufficient proof. Failure is not approval.

Even a complete benign observation CANNOT set either existing v2 approval pin
or authorize ANY model-generated HumanEval replay. This source deliberately
has no API that accepts a program, image, executable, profile or arbitrary PID.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import uuid


IMAGE_ID = "72036a705a7cff38924efd674e60654666cb283faf7b3fad160b34007f7395bd"
PYTHON_PATH = "/usr/local/bin/python3.13"
PYTHON_SHA256 = "73efccb6aa17a74869ef6f11e0a0118b0f27693ea85f36c7a3758c307fd8914f"
MEMORY_BYTES = 268_435_456
PIDS_LIMIT = 16
NAMESPACES = ("user", "pid", "mnt", "net")
MAX_OUTPUT = 32_768
EXECUTION_APPROVED = False  # HumanEval v2 approval pins remain explicitly unset.

# V37 fixed NEGATIVE tests. Each is a harmless, deterministic constant. The
# sole external address is the explicitly authorized public 1.1.1.1:443 canary
# inside --network=none; no host name, private address or DNS is referenced.
NETWORK_DENIAL_CANDIDATE = '''import errno,json,socket,time
s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.settimeout(1)
try:
 s.connect(('1.1.1.1',443))
 row={'mode':'network','connected':True,'errno':None}
except OSError as exc:
 row={'mode':'network','connected':False,'errno':exc.errno}
finally:s.close()
print(json.dumps(row,sort_keys=True),flush=True)
time.sleep(5)
'''

FILESYSTEM_DENIAL_CANDIDATE = '''import json,pathlib,time
cwd=pathlib.Path.cwd()
expected=pathlib.Path('/work')
owned=cwd/'owned-write.txt'
owned.write_text('fixed-canary',encoding='ascii')
inside=(owned.read_text(encoding='ascii')=='fixed-canary')
owned.unlink()
targets=('/etc/fixed-he-negative.txt','/tmp/fixed-he-negative.txt',
         '/fixed-he-negative.txt','/work/../fixed-he-negative.txt')
denied=[]
for target in targets:
 try:
  with open(target,'x',encoding='ascii') as f:f.write('fixed-canary')
  denied.append(False)
  pathlib.Path(target).unlink(missing_ok=True)
 except (PermissionError,FileNotFoundError,OSError):denied.append(True)
print(json.dumps({'mode':'filesystem','cwd':str(cwd),'inside_writable':inside,
                  'outside_denied':denied,'attempts':list(targets)},sort_keys=True),flush=True)
time.sleep(5)
'''

TIMEOUT_TREE_CANDIDATE = '''import json,subprocess,sys,time
grandchild=subprocess.Popen([sys.executable,'-I','-c','import time;time.sleep(60)'],
                            stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
print(json.dumps({'mode':'timeout','grandchild_spawned':grandchild.pid>0}),flush=True)
time.sleep(60)
'''

NEGATIVE_CANDIDATES = {
    'network': NETWORK_DENIAL_CANDIDATE,
    'filesystem': FILESYSTEM_DENIAL_CANDIDATE,
    'timeout': TIMEOUT_TREE_CANDIDATE,
}
# Populated with verified literal source hashes before a real canary may run.
NEGATIVE_SOURCE_SHA256 = {
    'network': '305515d86f24fb9712dc34691a9b834605208183a088b50be5accdce85dbea5d',
    'filesystem': '1934a317428eadecb085eea643f4dbe0beeb01e6102fea3bfd63f7d31edf4c71',
    'timeout': 'c771533f2c7ccb2d993aaa988d1810b95c96362048c67649744cd5e0a5d4b7f8',
}

# Only the disposable container's own fixed candidate process group is killed
# after the 10-second child wall budget. The Linux VM *host* later independently
# checks every observed PID+starttime and the privately owned cgroup.
NEGATIVE_LAUNCHER = '''import hashlib,json,os,pathlib,signal,subprocess,sys,tempfile
scripts=%s
expected_sha=%s
mode=%r
code=scripts[mode]
if hashlib.sha256(code.encode('utf-8')).hexdigest()!=expected_sha[mode]:raise SystemExit(81)
if sys.version_info[:2]!=(3,13) or os.geteuid()!=65534:raise SystemExit(82)
with tempfile.TemporaryDirectory(prefix='fixed-v37-',dir='/work') as td:
 p=pathlib.Path(td)/'candidate.py'
 p.write_text(code,encoding='utf-8')
 if hashlib.sha256(p.read_bytes()).hexdigest()!=expected_sha[mode]:raise SystemExit(83)
 os.chmod(p,0o600)
 child=subprocess.Popen([sys.executable,'-I','candidate.py'],cwd=td,
                        stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                        text=True,start_new_session=True)
 timed_out=False
 try:stdout,stderr=child.communicate(timeout=10)
 except subprocess.TimeoutExpired:
  timed_out=True
  os.killpg(child.pid,signal.SIGKILL)
  stdout,stderr=child.communicate(timeout=3)
 if len(stdout)>4096 or len(stderr)>4096:raise SystemExit(84)
 rows=[json.loads(line) for line in stdout.splitlines() if line.strip()]
 if len(rows)!=1 or rows[0].get('mode')!=mode:raise SystemExit(85)
 if (mode=='timeout')!=timed_out:raise SystemExit(86)
 if not timed_out and child.returncode!=0:raise SystemExit(87)
 print(json.dumps({'kind':'fixed_negative_v37','mode':mode,'candidate_sha256':expected_sha[mode],
    'python_version':list(sys.version_info[:2]),'candidate_argv':['python','-I','candidate.py'],
    'timeout_budget_s':10,'wall_timeout':timed_out,'process_group_killed':timed_out,
    'observation':rows[0],'generated_programs_executed':0},sort_keys=True),flush=True)
''' % (repr(NEGATIVE_CANDIDATES),repr(NEGATIVE_SOURCE_SHA256),'__MODE__')
NEGATIVE_LAUNCHER_SHA256 = {
    'network': '86fa2fba97b482b542dde83378265c49989c5bb213166717265e32485f3c6e1f',
    'filesystem': '2cb0ce26f13afde831393952864f4f0b16e9cba1bd68cd120807b41a2e7e0ffd',
    'timeout': 'a0891eaeac8cb94e078799935d6f80fa7f430e988f9dd7f25dc55b7c5ee005d5',
}
HOST_WITNESS_SOURCE_PIN = 'e9d3d7e87fb41253df43ba884978b2e4ced835dd2a6bed6aca89ed4d42e0216e'

# This is the ONLY script ever written/executed by the canary.  No import of
# eval/exec/compile, filesystem discovery, generated code or network request.
BENIGN_CANDIDATE = '''import hashlib, json, os, sys, time
print(json.dumps({"marker":"FIXED_BENIGN_HUMANEVAL_CANARY_V1", "version":list(sys.version_info[:2]), "uid":os.geteuid(), "file":"candidate.py"}),flush=True)
time.sleep(8)
'''
CANDIDATE_SHA256 = hashlib.sha256(BENIGN_CANDIDATE.encode("utf-8")).hexdigest()

# Fixed image-only launcher. The candidate file lives in an unmounted, private
# 8 MiB tmpfs; subprocess executes *only* CPython -I candidate.py, timeout=10.
# It leaves the process alive long enough for an outside-VM-host observation.
LAUNCHER = '''import hashlib,json,os,pathlib,subprocess,sys,tempfile
text = %s
if hashlib.sha256(text.encode('utf-8')).hexdigest()!=%r:raise SystemExit(81)
if sys.version_info[:2]!=(3,13) or os.geteuid()!=65534:raise SystemExit(82)
with tempfile.TemporaryDirectory(prefix='fixed-he-',dir='/tmp') as td:
 p=pathlib.Path(td)/'candidate.py'
 p.write_text(text,encoding='utf-8')
 os.chmod(p,0o600)
 result=subprocess.run([sys.executable,'-I','candidate.py'],cwd=td,capture_output=True,text=True,timeout=10,check=False)
 if result.returncode!=0:raise SystemExit(83)
 row=json.loads(result.stdout)
 if row!={'marker':'FIXED_BENIGN_HUMANEVAL_CANARY_V1','version':[3,13],'uid':65534,'file':'candidate.py'}:raise SystemExit(84)
 print(json.dumps({'kind':'fixed_benign_only','candidate_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'executable':sys.executable,'python_version':list(sys.version_info[:2]),'candidate_argv':['python','-I','candidate.py'],'timeout_seconds':10,'generated_programs_executed':0}),flush=True)
''' % (repr(BENIGN_CANDIDATE), CANDIDATE_SHA256)
LAUNCHER_SHA256 = hashlib.sha256(LAUNCHER.encode("utf-8")).hexdigest()

# Executed ONLY with `podman machine ssh -- python3 -c <THIS STATIC SOURCE> PID`.
# Runs in the existing VM's rootless host (NOT in the container). It performs
# exclusively read-only /proc, cgroup and /proc/PID/root observations. The PID
# originates in a strict machine-readable, verified Podman inspect owner.
VM_HOST_OBSERVER = r'''import hashlib,json,os,pathlib,re,sys
pid=int(sys.argv[1]);base=pathlib.Path('/proc')/str(pid)
def read(p,maxlen=32768):
 text=pathlib.Path(p).read_text(encoding='utf-8',errors='strict')
 if len(text)>maxlen:raise ValueError('host observation exceeds bound')
 return text
def status(raw):
 return {a.strip():b.strip() for line in raw.splitlines() if ':' in line for a,b in [line.split(':',1)]}
def stat(p):
 raw=read(pathlib.Path('/proc')/str(p)/'stat',2048);pos=raw.rfind(')')
 if pos<1:raise ValueError('malformed host PID stat')
 cols=raw[pos+2:].split()
 return {'pid':int(p),'ppid':int(cols[1]),'start_ticks':int(cols[19]),'state':cols[0]}
def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for part in iter(lambda:f.read(1048576),b''):h.update(part)
 return h.hexdigest()
if pid<=1:raise ValueError('invalid container owner')
root=stat(pid); children={};allp={}
for entry in pathlib.Path('/proc').iterdir():
 if not entry.name.isdecimal():continue
 try:obj=stat(int(entry.name));allp[obj['pid']]=obj
 except (OSError,ValueError,IndexError):continue
ids={pid};progress=True
while progress:
 progress=False
 for k,item in allp.items():
  if item['ppid'] in ids and k not in ids:ids.add(k);progress=True
children={str(p):allp[p] for p in sorted(ids) if p!=pid and p in allp}
cg=read(base/'cgroup',4096).strip().splitlines()
if len(cg)!=1 or not cg[0].startswith('0::/'):raise ValueError('v2 cgroup unavailable')
rel=cg[0][3:];cgroup=(pathlib.Path('/sys/fs/cgroup')/rel.lstrip('/')).resolve(strict=True)
if not cgroup.is_relative_to(pathlib.Path('/sys/fs/cgroup').resolve(strict=True)):
 raise ValueError('cgroup path escape')
netns=base/'net'
netdev=read(netns/'dev',8192)
ifaces=[line.split(':',1)[0].strip() for line in netdev.splitlines() if ':' in line]
netroute=read(netns/'route',8192)
net6=read(netns/'ipv6_route',8192)
cgprocs=read(cgroup/'cgroup.procs',8192)
mountinfo=read(base/'mountinfo',262144)
hostns={n:os.readlink('/proc/self/ns/'+n) for n in ('user','pid','mnt','net')}
childns={n:os.readlink(str(base/'ns'/n)) for n in ('user','pid','mnt','net')}
out={'host_uid':os.geteuid(),'host_ns':hostns,'child_ns':childns,
 'child':root,'descendants':children,'status':status(read(base/'status',16384)),
 'uid_map':read(base/'uid_map',4096),'gid_map':read(base/'gid_map',4096),
 'cgroup_path':str(cgroup),'cgroup_rel':rel,'memory_max':read(cgroup/'memory.max',1024).strip(),
 'pids_max':read(cgroup/'pids.max',1024).strip(),'cgroup_procs':cgprocs,
 'cgroup_events':read(cgroup/'cgroup.events',1024) if (cgroup/'cgroup.events').is_file() else '',
 'network_interfaces':ifaces,'ipv4_route':netroute,'ipv6_route':net6,
 'root_read_only':bool(os.statvfs(base/'root').f_flag & os.ST_RDONLY),
 'tmp_read_only':bool(os.statvfs(base/'root/tmp').f_flag & os.ST_RDONLY),
 'work_writable':not bool(os.statvfs(base/'root/work').f_flag & os.ST_RDONLY) if (base/'root/work').is_dir() else False,
 'host_mounts_seen':[line.split()[4] for line in mountinfo.splitlines() if len(line.split())>5 and line.split()[4].startswith(('/mnt/','/host/','/run/host'))][:32],
 'python_executable_sha256':digest(base/'root/usr/local/bin/python3.13')}
print(json.dumps(out,sort_keys=True))
'''
VM_HOST_OBSERVER_SHA256 = hashlib.sha256(VM_HOST_OBSERVER.encode("utf-8")).hexdigest()


def checked(argv: list[str], *, timeout: float = 12) -> subprocess.CompletedProcess[str]:
    """No shell and bounded output; no caller-supplied executable/command."""
    result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    if len(result.stdout) > MAX_OUTPUT or len(result.stderr) > MAX_OUTPUT:
        raise RuntimeError("unbounded Podman/host response")
    return result


def _vm_python_static(fixed_source: str, *numeric_or_path_args: str, timeout: float = 15):
    """SSH requires ONE remote command string; encode only fixed in-source code.

    No arbitrary operator Python or shell fragments may enter this function:
    arguments are strictly decimal process identities or VM cgroup paths.
    """
    if not isinstance(fixed_source, str) or fixed_source not in (
            VM_HOST_OBSERVER, VM_PREFLIGHT, VM_POSTCHECK):
        raise ValueError("unapproved outside-VM Python witness source")
    for arg in numeric_or_path_args:
        if not isinstance(arg, str) or not (
            re.fullmatch(r"[0-9]{1,20}", arg)
            or re.fullmatch(r"/sys/fs/cgroup/[A-Za-z0-9_./@-]{1,900}", arg)
        ) or ".." in arg:
            raise ValueError("unsafe outside-VM witness argument")
    token = base64.b64encode(fixed_source.encode("utf-8")).decode("ascii")
    command = ("python3 -c 'import base64;exec(base64.b64decode(\"" + token
               + "\").decode(\"utf-8\"))'" +
               (" " + " ".join(numeric_or_path_args) if numeric_or_path_args else ""))
    return checked(["podman", "machine", "ssh", "--", command], timeout=timeout)


VM_PREFLIGHT = "import os,pathlib;print(os.geteuid());print(pathlib.Path('/sys/fs/cgroup/cgroup.controllers').is_file())"
VM_POSTCHECK = (
    "import json,pathlib,sys;"
    "c=pathlib.Path(sys.argv[1]);a=sys.argv[2:];"
    "pairs=[(int(a[i]),int(a[i+1])) for i in range(0,len(a),2)];"
    "alive=[];"
    "\nfor pid,start in pairs:\n"
    " p=pathlib.Path('/proc')/str(pid)\n"
    " try:\n"
    "  raw=(p/'stat').read_text()\n"
    "  now=int(raw[raw.rfind(')')+2:].split()[19])\n"
    "  if now==start:alive.append(pid)\n"
    " except FileNotFoundError:pass\n"
    "print(json.dumps({'same_owned_processes_live':alive,"
    "'cgroup_procs':(c/'cgroup.procs').read_text().strip().split() if c.exists() else []}))")
VM_PREFLIGHT_SOURCE_PIN = 'ad207be43c99b373d93e16b317d947bde86e49c6311fc475be87038fb029aca2'
VM_POSTCHECK_SOURCE_PIN = '9f3e76fe71f34461c13806a996e1a8124508a244725fd9af38d5cbf98b88c64f'


def _strict_json(raw: str) -> dict:
    def pairs(items: list[tuple[str, object]]) -> dict:
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON evidence field")
            result[key] = value
        return result
    parsed = json.loads(raw, object_pairs_hook=pairs,
                        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    if not isinstance(parsed, dict):
        raise ValueError("host witness is not a JSON object")
    return parsed


def preflight() -> dict:
    image = checked(["podman", "image", "inspect", IMAGE_ID, "--format", "{{.Id}}"])
    if image.returncode != 0 or image.stdout.strip() != IMAGE_ID:
        raise RuntimeError("original preexisting Python 3.13 image ID is not source pinned")
    machine = checked(["podman", "machine", "list", "--format", "json"])
    if machine.returncode != 0:
        raise RuntimeError("existing Podman VM not available; no VM starts permitted")
    vms = json.loads(machine.stdout)
    if not isinstance(vms, list) or not any(item.get("Name") == "podman-machine-default"
                                             and item.get("Running") is True for item in vms):
        raise RuntimeError("existing default Podman VM not already running")
    vm = _vm_python_static(VM_PREFLIGHT)
    if vm.returncode != 0 or vm.stdout.strip().splitlines() != ["1000", "True"]:
        raise RuntimeError("VM independent rootless/cgroup-v2 observer unavailable")
    return {"image_id": IMAGE_ID, "interpreter_sha256_source_pin": PYTHON_SHA256,
            "existing_rootless_vm": True, "host_vm_observer": True,
            "v2_approval_pins_updated": False}


def container_argv(name: str) -> list[str]:
    if not re.fullmatch(r"w10-he-benign-[a-f0-9]{16}", name):
        raise ValueError("not an exclusively owned fixed benign container name")
    return ["podman", "run", "--detach", "--rm", "--pull=never", "--name", name,
            "--network=none", "--cgroupns=private", "--pid=private", "--ipc=private",
            "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
            "--user=65534:65534", "--userns=keep-id", "--pids-limit=16",
            f"--memory={MEMORY_BYTES}", f"--memory-swap={MEMORY_BYTES}",
            "--ulimit=core=0:0", "--tmpfs=/tmp:rw,nosuid,nodev,noexec,size=8m,mode=1777",
            "--tmpfs=/run/secrets:rw,nosuid,nodev,noexec,size=64k,mode=0700",
            "--env=PYTHONDONTWRITEBYTECODE=1", "--workdir=/tmp",
            f"--entrypoint={PYTHON_PATH}", IMAGE_ID, "-I", "-c", LAUNCHER]


def verify_negative_source_pins() -> None:
    """Separate hardcoded hashes: not caller-provided or self-asserted JSON."""
    if (hashlib.sha256(VM_HOST_OBSERVER.encode("utf-8")).hexdigest()
            != HOST_WITNESS_SOURCE_PIN):
        raise ValueError("missing or changed independently pinned host observer source SHA")
    if (hashlib.sha256(VM_PREFLIGHT.encode("utf-8")).hexdigest()
            != VM_PREFLIGHT_SOURCE_PIN
            or hashlib.sha256(VM_POSTCHECK.encode("utf-8")).hexdigest()
            != VM_POSTCHECK_SOURCE_PIN):
        raise ValueError("missing/forged fixed VM preflight/teardown observer source SHA")
    if set(NEGATIVE_CANDIDATES) != {"network", "filesystem", "timeout"}:
        raise ValueError("fixed negative canary inventory changed")
    for mode, source in NEGATIVE_CANDIDATES.items():
        if (not re.fullmatch(r"[a-f0-9]{64}", NEGATIVE_SOURCE_SHA256.get(mode, ""))
                or hashlib.sha256(source.encode("utf-8")).hexdigest() != NEGATIVE_SOURCE_SHA256[mode]):
            raise ValueError("missing/forged fixed negative canary source SHA: " + mode)
        launcher = NEGATIVE_LAUNCHER.replace("'__MODE__'", repr(mode))
        if (not re.fullmatch(r"[a-f0-9]{64}", NEGATIVE_LAUNCHER_SHA256.get(mode, ""))
                or hashlib.sha256(launcher.encode("utf-8")).hexdigest() != NEGATIVE_LAUNCHER_SHA256[mode]):
            raise ValueError("missing/forged fixed negative launcher source SHA: " + mode)


def negative_container_argv(name: str, mode: str) -> list[str]:
    verify_negative_source_pins()
    if mode not in NEGATIVE_CANDIDATES:
        raise ValueError("unapproved negative canary mode")
    cmd = container_argv(name)
    # No --rm: retain exact owned stopped-container logs until independent
    # inspection; only the run's UUID-owned container will be removed later.
    cmd.remove("--rm")
    work = "--tmpfs=/work:rw,nosuid,nodev,noexec,size=8m,mode=1777"
    cmd[cmd.index("--tmpfs=/tmp:rw,nosuid,nodev,noexec,size=8m,mode=1777")] = work
    # Podman enables a writable /tmp for read-only roots by default. Explicit
    # read-only /tmp prevents escape from the sole intended writable /work.
    cmd.insert(cmd.index("--tmpfs=/run/secrets:rw,nosuid,nodev,noexec,size=64k,mode=0700"),
               "--tmpfs=/tmp:ro,nosuid,nodev,noexec,size=64k,mode=0555")
    cmd[cmd.index("--workdir=/tmp")] = "--workdir=/work"
    cmd[-1] = NEGATIVE_LAUNCHER.replace("'__MODE__'", repr(mode))
    return cmd


def verify_external_observation(observed: dict) -> dict:
    if not isinstance(observed, dict):
        raise ValueError("outside-VM observer missing")
    host = observed.get("host_ns")
    child = observed.get("child_ns")
    valid_ns = lambda d: isinstance(d, dict) and all(
        isinstance(d.get(n), str) and re.fullmatch(r"[a-z]+:\[\d+\]", d[n])
        for n in NAMESPACES)
    if not valid_ns(host) or not valid_ns(child) or any(host[n] == child[n] for n in NAMESPACES):
        raise ValueError("independent user/pid/mount/net namespace difference not proven")
    props = observed.get("status")
    if (not isinstance(props, dict) or props.get("CapEff") != "0000000000000000"
            or props.get("NoNewPrivs") != "1" or props.get("Seccomp") != "2"):
        raise ValueError("outside-observer capability, newprivs or seccomp mismatch")
    uids = props.get("Uid", "").split()
    if observed.get("host_uid") != 1000 or len(uids) != 4 or any(int(v) == 0 for v in uids):
        raise ValueError("actual VM/rootless process owner not proven")
    root = observed.get("child")
    if (not isinstance(root, dict) or type(root.get("pid")) is not int
            or root["pid"] <= 1 or type(root.get("start_ticks")) is not int
            or root["start_ticks"] <= 0 or root.get("state") == "Z"):
        raise ValueError("outside-VM process identity/starttime absent")
    descendants = observed.get("descendants")
    if not isinstance(descendants, dict) or not descendants or not any(
        isinstance(item, dict) and item.get("ppid") == root["pid"]
        and type(item.get("start_ticks")) is int and item["start_ticks"] > 0
        for item in descendants.values()
    ):
        raise ValueError("real candidate.py descendant not host-observed")
    if (observed.get("pids_max") != str(PIDS_LIMIT)
            or observed.get("memory_max") != str(MEMORY_BYTES)):
        raise ValueError("external cgroup v2 memory/PID caps unproven")
    cgroup = observed.get("cgroup_path")
    if not isinstance(cgroup, str) or not cgroup.startswith("/sys/fs/cgroup/"):
        raise ValueError("external private cgroup ownership missing")
    procs = observed.get("cgroup_procs", "")
    if str(root["pid"]) not in procs.split():
        raise ValueError("observed container PID missing from own cgroup")
    ipv4_lines = observed.get("ipv4_route", "").splitlines()
    ipv6_lines = observed.get("ipv6_route", "").splitlines()
    # Linux's isolated, interface-less network namespace may report IPv6
    # ::/0 *reject* entries and ::1/128, all carried by local `lo`. These are
    # not proof of outbound routing. Accept only `lo`-owned route entries,
    # refuse any nonloopback NIC, route interface, or default IPv4 gateway.
    if (observed.get("network_interfaces") != ["lo"]
            or not isinstance(observed.get("ipv4_route"), str)
            or not isinstance(observed.get("ipv6_route"), str)
            or not ipv4_lines or not ipv4_lines[0].startswith("Iface")
            or any(len(cols) < 2 or cols[0] != "lo" or cols[1] == "00000000"
                   for row in ipv4_lines[1:] if (cols := row.split()))
            or any(len(cols) < 10 or cols[-1] != "lo"
                   for row in ipv6_lines if (cols := row.split()))):
        raise ValueError("independent no-egress kernel namespace/network route unproven")
    if (observed.get("python_executable_sha256") != PYTHON_SHA256
            or observed.get("root_read_only") is not True
            or observed.get("host_mounts_seen") != []):
        raise ValueError("outside-observed source-pinned interpreter or private root/mount invalid")
    return {"external_namespaces_all_distinct": True, "nonroot_no_caps_seccomp2": True,
            "outside_verified_no_egress_interfaces": True, "cgroup_memory_pids_enforced": True,
            "real_benign_descendants_seen": len(descendants), "source_pinned_interpreter": True}


def _inspect_owned(name: str, owner: str) -> dict:
    raw = checked(["podman", "inspect", name, "--format", "{{json .}}"], timeout=10)
    if raw.returncode != 0:
        raise RuntimeError("owned benign Podman container not inspectable")
    info = _strict_json(raw.stdout)
    if info.get("Id") != owner or info.get("Name", "").lstrip("/") != name:
        raise RuntimeError("Podman container owner ID/name mismatch")
    return info


def _vm_observe(pid: int) -> dict:
    if type(pid) is not int or pid <= 1:
        raise ValueError("outside witness requires verified numeric PID")
    result = _vm_python_static(VM_HOST_OBSERVER, str(pid))
    if result.returncode != 0:
        raise RuntimeError("real outside-container VM-host /proc census unavailable: "
                           + result.stderr[-200:])
    return _strict_json(result.stdout)


def _vm_postcheck(observed: dict) -> dict:
    # External process check, independent of container self-report and Podman.
    child = observed.get("child", {})
    pid, start_ticks = child.get("pid"), child.get("start_ticks")
    cgroup_path = observed.get("cgroup_path")
    if type(pid) is not int or pid <= 1 or type(start_ticks) is not int or start_ticks <= 0:
        raise ValueError("missing external process identity for teardown proof")
    if not isinstance(cgroup_path, str) or not cgroup_path.startswith("/sys/fs/cgroup/"):
        raise ValueError("missing safe external cgroup path")
    pairs = [(pid, start_ticks)]
    for item in observed.get("descendants", {}).values():
        if (not isinstance(item, dict) or type(item.get("pid")) is not int
                or type(item.get("start_ticks")) is not int):
            raise ValueError("external descendant process identity missing")
        pairs.append((item["pid"], item["start_ticks"]))
    if len(pairs) > PIDS_LIMIT:
        raise ValueError("unbounded external owned process tree")
    args = [cgroup_path, *(str(v) for pair in pairs for v in pair)]
    response = _vm_python_static(VM_POSTCHECK, *args)
    if response.returncode != 0:
        raise RuntimeError("independent VM teardown observation failed")
    observation = _strict_json(response.stdout)
    if observation.get("same_owned_processes_live") != [] or observation.get("cgroup_procs") != []:
        raise RuntimeError("owned descendant/cgroup extinction not externally verified")
    return observation


def run_fixed_benign_witness() -> dict:
    """One already-installed container; refusal never authorizes generated code."""
    preflight()
    name = "w10-he-benign-" + uuid.uuid4().hex[:16]
    collision = checked(["podman", "container", "exists", name], timeout=10)
    if collision.returncode != 1:
        raise RuntimeError("exclusive rootless canary name preexisting or unknown")
    started = False
    owner = ""
    pid = None
    observed = None
    proof = None
    error = None
    try:
        launch = checked(container_argv(name), timeout=16)
        if launch.returncode != 0 or not re.fullmatch(r"[a-f0-9]{64}", launch.stdout.strip()):
            raise RuntimeError("fixed no-network canary failed to start: " + launch.stderr[-180:])
        owner = launch.stdout.strip()
        started = True
        info = _inspect_owned(name, owner)
        pid = info.get("State", {}).get("Pid")
        if type(pid) is not int or pid <= 1:
            raise RuntimeError("owned container PID not observed by Podman")
        observed = _vm_observe(pid)
        if observed.get("child", {}).get("pid") != pid:
            raise RuntimeError("VM external PID differs from unique container owner")
        if owner not in observed.get("cgroup_path", ""):
            raise RuntimeError("VM externally observed cgroup does not bind exact owner ID")
        proof = verify_external_observation(observed)
        wait = checked(["podman", "wait", name], timeout=16)
        if wait.returncode != 0 or wait.stdout.strip() != "0":
            raise RuntimeError("benign candidate.py did not finish successfully")
        # Source-bound output must be observed from actual container stdout.
        # With --rm it may be gone after wait: log retrieval is intentionally
        # not accepted as a substitute for host process proof.
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        error = type(exc).__name__ + ": " + str(exc)[:240]
    finally:
        if started:
            try:
                still = checked(["podman", "container", "exists", name], timeout=10)
                if still.returncode == 0:
                    current = _inspect_owned(name, owner)  # refuse collision/reassignment
                    if current["Id"] == owner:
                        checked(["podman", "kill", name], timeout=10)
                        checked(["podman", "rm", "--force", name], timeout=10)
            except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                error = "OWNED_CONTAINER_CLEANUP_UNVERIFIED: " + type(exc).__name__
    post = None
    if observed is not None and pid is not None:
        try:
            post = _vm_postcheck(observed)
        except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as exc:
            error = "EXTERNAL_TEARDOWN_UNVERIFIED: " + type(exc).__name__
    return {"schema": "chowder.reap-v2-external-fixed-benign-host-witness.v1",
            "status": ("BENIGN_OUTSIDE_VM_HOST_OBSERVED_NOT_EXECUTION_APPROVAL"
                       if error is None and proof is not None and post is not None
                       else "REFUSED_INDEPENDENT_BENIGN_HOST_WITNESS"),
            "refusal_reason": error, "image_id": IMAGE_ID,
            "python_executable_sha256": PYTHON_SHA256,
            "candidate_sha256": CANDIDATE_SHA256, "launcher_sha256": LAUNCHER_SHA256,
            "host_observer_source_sha256": VM_HOST_OBSERVER_SHA256,
            "owned_container_id": owner or None, "external_child_pid": pid,
            "external_raw_witness": observed, "verified_benign_properties": proof,
            "teardown": post, "generated_programs_executed": 0,
            "sandbox_qualified_for_generated_code": False,
            "approved_sandbox_sha_written": False, "approved_external_witness_sha_written": False,
            "gen0_status": "BLOCKED", "chosen": None, "release_approved": False}


def verify_negative_log(log: dict, mode: str, host_observed: dict) -> dict:
    """Independent actual Podman logs must match fixed-source outcomes exactly."""
    verify_negative_source_pins()
    if mode not in NEGATIVE_CANDIDATES or not isinstance(log, dict):
        raise ValueError("missing authentic fixed negative canary result")
    if (not isinstance(host_observed, dict)
            or host_observed.get("tmp_read_only") is not True
            or host_observed.get("work_writable") is not True):
        raise ValueError("external VM-host mount observer did not prove /tmp read-only and /work writable")
    expected = {"kind": "fixed_negative_v37", "mode": mode,
                "candidate_sha256": NEGATIVE_SOURCE_SHA256[mode],
                "python_version": [3, 13],
                "candidate_argv": ["python", "-I", "candidate.py"],
                "timeout_budget_s": 10, "generated_programs_executed": 0,
                "wall_timeout": mode == "timeout", "process_group_killed": mode == "timeout"}
    if any(type(log.get(key)) is not type(value) or log[key] != value
           for key, value in expected.items()):
        raise ValueError("forged, incomplete, or wrong-source negative canary result")
    if set(log) != {*expected, "observation"}:
        raise ValueError("unexpected negative evidence output fields")
    row = log["observation"]
    if not isinstance(row, dict) or row.get("mode") != mode:
        raise ValueError("negative child result missing/mismatched")
    if mode == "network":
        # ENETUNREACH/EHOSTUNREACH/EPERM/EACCES: kernel/network enforcement;
        # ECONNREFUSED is *not* sufficient (a real remote listener might reply).
        if (set(row) != {"mode", "connected", "errno"}
                or row["connected"] is not False
                or type(row["errno"]) is not int or row["errno"] not in (101, 113, 1, 13)):
            raise ValueError("active outbound 1.1.1.1:443 was not kernel-denied")
    elif mode == "filesystem":
        protected = ["/etc/fixed-he-negative.txt", "/tmp/fixed-he-negative.txt",
                     "/fixed-he-negative.txt", "/work/../fixed-he-negative.txt"]
        if (set(row) != {"mode", "cwd", "inside_writable", "outside_denied", "attempts"}
                or not isinstance(row["cwd"], str) or
                not re.fullmatch(r"/work/fixed-v37-[a-zA-Z0-9_-]+", row["cwd"])
                or row["inside_writable"] is not True
                or row["attempts"] != protected
                or row["outside_denied"] != [True] * 4):
            raise ValueError("filesystem writable-workdir escape or read-only-root denial unproven")
    elif mode == "timeout":
        descendants = host_observed.get("descendants") if isinstance(host_observed, dict) else None
        if (set(row) != {"mode", "grandchild_spawned"} or row["grandchild_spawned"] is not True
                or not isinstance(descendants, dict) or len(descendants) < 2):
            raise ValueError("inherited grandchild not independently witnessed before timeout")
        root_pid = host_observed.get("child", {}).get("pid")
        if not any(isinstance(item, dict) and item.get("ppid") in {
                int(key) for key in descendants if key.isdecimal()
        } and item.get("pid") != root_pid for item in descendants.values()):
            raise ValueError("host witnessed no true candidate-to-grandchild lineage")
    return {"active_negative_mode": mode, "fixed_source_sha256": NEGATIVE_SOURCE_SHA256[mode],
            "host_observer_required": True, "generated_code_approved": False}


def run_fixed_negative_witness(mode: str) -> dict:
    """One fixed, benign failure canary with actual independent VM-host proof.

    Refuses to infer approval from a Podman self-report. Only UUID-owned
    disposable containers may be terminated; preexisting services untouched.
    """
    if mode not in NEGATIVE_CANDIDATES:
        raise ValueError("unapproved negative canary mode")
    verify_negative_source_pins()
    preflight()
    name = "w10-he-benign-" + uuid.uuid4().hex[:16]
    collision = checked(["podman", "container", "exists", name], timeout=10)
    if collision.returncode != 1:
        raise RuntimeError("exclusive fixed negative container name collided/unknown")
    owned = ""
    root_observation = None
    verdict = None
    log = None
    teardown = None
    refusal = None
    cleanup_verified = False
    try:
        started = checked(negative_container_argv(name, mode), timeout=16)
        if started.returncode != 0 or not re.fullmatch(r"[a-f0-9]{64}", started.stdout.strip()):
            raise RuntimeError("source-pinned fixed negative container launch refused")
        owned = started.stdout.strip()
        info = _inspect_owned(name, owned)
        pid = info.get("State", {}).get("Pid")
        if type(pid) is not int or pid <= 1:
            raise RuntimeError("fixed negative container process PID unavailable")
        if info.get("Image") != IMAGE_ID:
            raise RuntimeError("container image ID does not match approved benign image")
        root_observation = _vm_observe(pid)
        if (root_observation.get("child", {}).get("pid") != pid
                or owned not in root_observation.get("cgroup_path", "")):
            raise RuntimeError("VM-host PID/cgroup not bound to UUID-owned container")
        verify_external_observation(root_observation)
        waited = checked(["podman", "wait", name], timeout=22)
        if waited.returncode != 0 or waited.stdout.strip() != "0":
            raise RuntimeError("fixed negative canary launcher nonzero/timeout")
        raw_log = checked(["podman", "logs", "--tail", "1", name], timeout=25)
        if raw_log.returncode != 0:
            raise RuntimeError("source-bound fixed negative canary log inaccessible")
        log = _strict_json(raw_log.stdout.strip())
        verdict = verify_negative_log(log, mode, root_observation)
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as error:
        refusal = type(error).__name__ + ": " + str(error)[:240]
    finally:
        if owned:
            try:
                existing = checked(["podman", "container", "exists", name], timeout=10)
                if existing.returncode != 0:
                    raise RuntimeError("owned container vanished before verified cleanup")
                _inspect_owned(name, owned)  # refuse reassigned owner
                removed = checked(["podman", "rm", "--force", name], timeout=12)
                gone = checked(["podman", "container", "exists", name], timeout=10)
                cleanup_verified = removed.returncode == 0 and gone.returncode == 1
                if not cleanup_verified:
                    raise RuntimeError("owned container not confirmed removed")
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                refusal = "OWNED_CONTAINER_CLEANUP_UNVERIFIED: " + type(error).__name__
    if root_observation is not None:
        try:
            teardown = _vm_postcheck(root_observation)
        except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as error:
            refusal = "INDEPENDENT_VM_PROCESS_TREE_TEARDOWN_UNVERIFIED: " + type(error).__name__
    # Genuine V37 benign negative observation is not an approval of arbitrary
    # generated Python or of 164 official HumanEval replays.
    passed = (refusal is None and verdict is not None and teardown is not None
              and cleanup_verified)
    return {"schema": "chowder.reap-v2-fixed-negative-external-host-witness.v37",
            "status": ("FIXED_NEGATIVE_EXTERNALLY_OBSERVED_NOT_EXECUTION_APPROVAL"
                       if passed else "REFUSED_FIXED_NEGATIVE_EXTERNAL_HOST_WITNESS"),
            "mode": mode, "refusal_reason": refusal,
            "image_id": IMAGE_ID, "python_executable_sha256": PYTHON_SHA256,
            "candidate_sha256": NEGATIVE_SOURCE_SHA256[mode],
            "launcher_sha256": NEGATIVE_LAUNCHER_SHA256[mode],
            "host_observer_source_sha256": HOST_WITNESS_SOURCE_PIN,
            "owner_container_id": owned or None,
            "raw_host_observation": root_observation, "raw_fixed_canary_result": log,
            "negative_checks": verdict, "independent_vm_teardown": teardown,
            "owned_container_cleanup_verified": cleanup_verified,
            "generated_programs_executed": 0, "sandbox_qualified_for_generated_code": False,
            "approved_sandbox_sha_written": False, "approved_external_witness_sha_written": False,
            "gen0_status": "BLOCKED", "chosen": None, "release_approved": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-fixed-benign", action="store_true",
                        help="Run exactly the built-in harmless candidate.py in existing rootless VM")
    parser.add_argument("--run-negative", choices=sorted(NEGATIVE_CANDIDATES),
                        help="Run exactly one prewritten harmless V37 failure canary")
    args = parser.parse_args()
    if args.run_fixed_benign and args.run_negative:
        parser.error("run one fixed benign canary only per invocation")
    try:
        result = (run_fixed_negative_witness(args.run_negative) if args.run_negative
                  else run_fixed_benign_witness() if args.run_fixed_benign else {
            "status": "READ_ONLY_PRECHECK_FIXED_BENIGN_NOT_LAUNCHED", **preflight(),
            "sandbox_qualified_for_generated_code": False, "gen0_status": "BLOCKED"})
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0 if result["status"] == "READ_ONLY_PRECHECK_FIXED_BENIGN_NOT_LAUNCHED" else 2
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "REFUSED_BENIGN_WITNESS_PREFLIGHT",
                          "error_type": type(exc).__name__,
                          "no_generated_python_executed": True,
                          "sandbox_qualified_for_generated_code": False,
                          "gen0_status": "BLOCKED"}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
