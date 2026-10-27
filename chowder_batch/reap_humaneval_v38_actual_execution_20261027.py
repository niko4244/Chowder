"""Original 164-task Q4/Q8 HumanEval source and generated-code admission boundary.

ACTUAL candidate executions are intentionally impossible while the independent
v2 source-reviewed sandbox capability/witness pins are unset.  Do not convert
the historical Linux HumanEval scores into a newly isolated PASS.  The V37
outside-VM-host evidence contains *fixed benign only* candidates; it cannot
authorize executable model output by substituting a receipt or CLI flag.

This file validates both complete immutable original response populations,
including the original canonical 164 HumanEval tasks, original scorer identity,
each original candidate/session/termination, all 164 exact (source-only)
program SHA256 fingerprints, and independent approval prerequisites without
executing one byte of generated Python.  This is NOT a replacement scorer or
an alternate bypass of the original v2 attestation contract.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time
import uuid

from chowder_batch import reap_humaneval_v2_attestation as original_v2
from chowder_batch import reap_humaneval_independent_witness_20261027 as v37
from chowder_batch.reap_external_full_audit import (
    EVALUATOR_SHA, Q4_SHA, Q8_SHA, SCORER_SHA, _strict_json,
)
from chowder_batch.reap_humaneval_podman_replay import exact_program
from chowder_batch.reap_humaneval_podman_replay import host_safety_guard


CANONICAL = Path("F:/reap-v2-official-humaneval-private-original-inputs-phase4-20261026/humaneval.json")
CANONICAL_SHA256 = "5144d31cf337b25ce262b780ed103b128b1e9d289d3c0d15f4fe34933bb705d5"
Q4_ROWS = Path("F:/reap-v2-kaggle-q4-full-complete-private-evidence-20261026/reap-v2-a2a6b83e345145839cc333561b2f3baf-humaneval.jsonl")
Q8_ROWS = Path("F:/reap-v2-q8-postload-full-complete-original-evidence-20261026-phase4/reap-q8-reclaim-full-95876063255e45159f57e29306e0e885-humaneval.jsonl")
Q4_ROW_SHA256 = "92ec6439be77a7dbeb1a80448a0bb664e3cb6113a9cc0d356f7dbe300db69c43"
Q8_ROW_SHA256 = "cd3c9d1577611a032b36e69a3f96bd272d118819074352503f143bcd68cac301"
Q4_SESSION = "a2a6b83e345145839cc333561b2f3baf"
Q8_SESSION = "95876063255e45159f57e29306e0e885"
ORIGINAL_POPULATION = 164
ROW_LIMIT = 2 * 1024 * 1024
SOURCE_LIMIT = 2 * 1024 * 1024
V37_EXTERNAL_HOST_OBSERVER_SHA256 = "e9d3d7e87fb41253df43ba884978b2e4ced835dd2a6bed6aca89ed4d42e0216e"
V37_IMAGE_ID = "72036a705a7cff38924efd674e60654666cb283faf7b3fad160b34007f7395bd"
V37_PYTHON_EXECUTABLE_SHA256 = "73efccb6aa17a74869ef6f11e0a0118b0f27693ea85f36c7a3758c307fd8914f"

# V38 research-only candidate transport: bytes go through Podman stdin, NEVER
# argv, shell, host mount or environment. The launcher checks SHA before making
# candidate.py, then waits for an OUTSIDE-VM-host source/cgroup/namespace read
# before the host releases the candidate. An independent review must bind
# this new loader + observer source SHA before ANY model response is runnable.
V38_FIXED_HOLDOUT = "print('V38_EXACT_SOURCE_ONLY_HOLDOUT')\n"
V38_FIXED_HOLDOUT_SHA256 = "0e684e1394f9263fb32556f8e1e5dc51c6f9891940a55ec3eae4ec87956fd156"
V38_LOADER = r'''import fcntl,hashlib,json,os,pathlib,resource,signal,subprocess,sys,time
source=sys.stdin.buffer.read(2097153)
if not 0<len(source)<=2097152 or hashlib.sha256(source).hexdigest()!=sys.argv[1]:raise SystemExit(70)
if sys.version_info[:2]!=(3,13) or os.geteuid()!=65534:raise SystemExit(71)
candidate=pathlib.Path('/work/candidate.py')
with candidate.open('xb') as stream:stream.write(source)
os.chmod(candidate,0o644)
# Python executes an immutable kernel-sealed memfd, NEVER the mutable
# candidate.py path. The outside VM host attests BOTH independently before
# release, closing the observed source-path replacement TOCTOU gap.
sealed=os.memfd_create('reap-v38-candidate',os.MFD_CLOEXEC|os.MFD_ALLOW_SEALING)
remaining=memoryview(source)
while remaining:
 written=os.write(sealed,remaining)
 if written<=0:raise RuntimeError('sealed source write failed')
 remaining=remaining[written:]
os.lseek(sealed,0,os.SEEK_SET)
required=fcntl.F_SEAL_SEAL|fcntl.F_SEAL_SHRINK|fcntl.F_SEAL_GROW|fcntl.F_SEAL_WRITE
fcntl.fcntl(sealed,fcntl.F_ADD_SEALS,required)
if fcntl.fcntl(sealed,fcntl.F_GET_SEALS)!=required:raise RuntimeError('candidate memfd not immutable')
print(json.dumps({'stage':'PREEXEC_SOURCE_STAGED','sha256':hashlib.sha256(candidate.read_bytes()).hexdigest(),
 'sealed_fd':sealed,'sealed_sha256':hashlib.sha256(os.pread(sealed,len(source),0)).hexdigest()}),flush=True)
limit=time.monotonic()+12
while time.monotonic()<limit and not pathlib.Path('/work/release').exists():time.sleep(0.05)
if not pathlib.Path('/work/release').exists():raise SystemExit(72)
def limit_resources():
 resource.setrlimit(resource.RLIMIT_AS,(2147483648,2147483648))
 resource.setrlimit(resource.RLIMIT_FSIZE,(1048576,1048576))
 resource.setrlimit(resource.RLIMIT_NOFILE,(64,64))
 resource.setrlimit(resource.RLIMIT_CPU,(10,11))
 resource.setrlimit(resource.RLIMIT_CORE,(0,0))
p=subprocess.Popen([sys.executable,'-I','/proc/self/fd/'+str(sealed)],cwd='/work',stdin=subprocess.DEVNULL,
 stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True,preexec_fn=limit_resources,
 pass_fds=(sealed,))
timed_out=False
try:stdout,stderr=p.communicate(timeout=10)
except subprocess.TimeoutExpired:
 timed_out=True
 os.killpg(p.pid,signal.SIGKILL)
 stdout,stderr=p.communicate(timeout=3)
if len(stdout)>65536 or len(stderr)>65536:raise SystemExit(73)
print(json.dumps({'stage':'ORIGINAL_CANDIDATE_FINISHED','sha256':hashlib.sha256(os.pread(sealed,len(source),0)).hexdigest(),
 'candidate_exit':p.returncode,'timed_out':timed_out,'passed':p.returncode==0 and not timed_out,
 'generated_model_candidate_executed':True},sort_keys=True),flush=True)
'''

V38_VM_SOURCE_OBSERVER = r'''import fcntl,hashlib,json,os,pathlib,stat as statlib,sys
pid=int(sys.argv[1]);sealed_fd=int(sys.argv[2]);base=pathlib.Path('/proc')/str(pid)
if not 3<=sealed_fd<=64:raise RuntimeError('unbounded sealed fd number')
process_stat=(base/'stat').read_text();start_ticks=int(process_stat[process_stat.rfind(')')+2:].split()[19])
source=base/'root/work/candidate.py'
fd=os.open(source,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC)
try:
 file_stat=os.fstat(fd)
 path_stat=os.stat(source,follow_symlinks=False)
 if not statlib.S_ISREG(file_stat.st_mode) or file_stat.st_nlink!=1 or file_stat.st_size<=0:raise RuntimeError('untrusted staged inode')
 if (file_stat.st_dev,file_stat.st_ino)!=(path_stat.st_dev,path_stat.st_ino):raise RuntimeError('staged inode changed')
 if file_stat.st_uid==0 or file_stat.st_mode&0o022:raise RuntimeError('staged inode ownership/mode unsafe')
 raw=b''
 while len(raw)<=2097152:
  part=os.read(fd,min(65536,2097153-len(raw)))
  if not part:break
  raw+=part
 after=os.fstat(fd)
 if (file_stat.st_dev,file_stat.st_ino,file_stat.st_size,file_stat.st_mtime_ns,file_stat.st_ctime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise RuntimeError('staged inode mutated during read')
finally:os.close(fd)
if len(raw)>2097152:raise RuntimeError('unbounded candidate.py staged bytes')
cglines=(base/'cgroup').read_text().strip().splitlines()
if len(cglines)!=1 or not cglines[0].startswith('0::/'):raise RuntimeError('no owner cgroup v2')
cgroup=pathlib.Path('/sys/fs/cgroup')/cglines[0][3:].lstrip('/')
if not cgroup.resolve().is_relative_to(pathlib.Path('/sys/fs/cgroup').resolve()):raise RuntimeError('cgroup escape')
procs=[int(v) for v in (cgroup/'cgroup.procs').read_text().split()]
memfds=[]
for possible in procs:
 handle=pathlib.Path('/proc')/str(possible)/'fd'/str(sealed_fd)
 try:target=os.readlink(handle)
 except (OSError,PermissionError):continue
 if not target.startswith('/memfd:reap-v38-candidate (deleted)'):continue
 opened=os.open(handle,os.O_RDONLY|os.O_CLOEXEC)
 try:
  filestat=os.fstat(opened)
  required=fcntl.F_SEAL_SEAL|fcntl.F_SEAL_SHRINK|fcntl.F_SEAL_GROW|fcntl.F_SEAL_WRITE
  seals=fcntl.fcntl(opened,fcntl.F_GET_SEALS)
  if seals!=required or not statlib.S_ISREG(filestat.st_mode) or filestat.st_size!=len(raw):
   raise RuntimeError('unsealed or malformed candidate memfd')
  sealed_raw=os.pread(opened,len(raw)+1,0)
  if sealed_raw!=raw:raise RuntimeError('memfd payload differs from staged source')
  memfds.append({'pid':possible,'fd':sealed_fd,'sha256':hashlib.sha256(sealed_raw).hexdigest(),
                 'inode':filestat.st_ino,'device':filestat.st_dev,'seals':seals})
 finally:os.close(opened)
if len(memfds)!=1:raise RuntimeError('unique independently readable sealed memfd missing')
status=[]
for p in procs:
 s={k.strip():v.strip() for row in (pathlib.Path('/proc')/str(p)/'status').read_text().splitlines()
       if ':' in row for k,v in [row.split(':',1)]}
 status.append({'pid':p,'nnp':s.get('NoNewPrivs'),'seccomp':s.get('Seccomp'),
                'capeff':s.get('CapEff'),'uid':s.get('Uid')})
ifaces=[row.split(':',1)[0].strip() for row in (base/'net/dev').read_text().splitlines() if ':' in row]
ipv4=(base/'net/route').read_text().splitlines()
if not ipv4 or ipv4[0].split()[:2]!=['Iface','Destination']:raise RuntimeError('malformed IPv4 route table')
for row in ipv4[1:]:
 cols=row.split()
 if len(cols)<11 or cols[0]!='lo' or cols[1]=='00000000':raise RuntimeError('unexpected IPv4 route')
ipv6=(base/'net/ipv6_route').read_text().splitlines()
for row in ipv6:
 cols=row.split()
 if len(cols)<10 or cols[-1]!='lo':raise RuntimeError('unexpected IPv6 route')
mounts=(base/'mountinfo').read_text().splitlines()
if len(mounts)<3:raise RuntimeError('missing mount inventory')
parsed=[]
for row in mounts:
 parts=row.split(' - ',1)
 if len(parts)!=2:raise RuntimeError('bad mountinfo field delimiter')
 lhs,rhs=parts[0].split(),parts[1].split()
 if len(lhs)<6 or len(rhs)<3 or not lhs[0].isdigit():raise RuntimeError('malformed mountinfo record')
 point=lhs[4]
 if '\\040' in point or '\\011' in point or not point.startswith('/'):raise RuntimeError('unsafe escaped mount path')
 parsed.append({'point':point,'source':rhs[1],'fstype':rhs[0],'options':lhs[5].split(',')})
# All mounts below are exact runtime-generated virtual filesystems from the
# independent native Podman-VM capsule census. Unexpected host bind mounts are
# rejected before generated Python is released.
expected={
 '/':('overlay','overlay'),
 '/work':('tmpfs','tmpfs'),'/tmp':('tmpfs','tmpfs'),
 '/run':('tmpfs','tmpfs'),'/var/tmp':('tmpfs','tmpfs'),
 '/run/.containerenv':('tmpfs','tmpfs'),'/run/secrets':('tmpfs','tmpfs'),
 '/etc/hosts':('tmpfs','tmpfs'),'/etc/hostname':('tmpfs','tmpfs'),
 '/etc/resolv.conf':('tmpfs','tmpfs'),
 '/dev':('tmpfs','tmpfs'),'/dev/pts':('devpts','devpts'),
 '/dev/shm':('tmpfs','shm'),'/dev/mqueue':('mqueue','mqueue'),
 '/dev/null':('devtmpfs','none'),'/dev/zero':('devtmpfs','none'),
 '/dev/full':('devtmpfs','none'),'/dev/tty':('devtmpfs','none'),
 '/dev/random':('devtmpfs','none'),'/dev/urandom':('devtmpfs','none'),
 '/proc':('proc','proc'),'/proc/bus':('proc','proc'),'/proc/fs':('proc','proc'),
 '/proc/irq':('proc','proc'),'/proc/sys':('proc','proc'),
 '/proc/sysrq-trigger':('proc','proc'),'/proc/acpi':('tmpfs','tmpfs'),
 '/proc/scsi':('tmpfs','tmpfs'),
 '/proc/kcore':('devtmpfs','none'),'/proc/keys':('devtmpfs','none'),
 '/proc/latency_stats':('devtmpfs','none'),
 '/proc/timer_list':('devtmpfs','none'),'/proc/interrupts':('devtmpfs','none'),
 '/sys':('sysfs','sysfs'),'/sys/fs/cgroup':('cgroup2','cgroup2'),
 '/sys/firmware':('tmpfs','tmpfs'),'/sys/fs/selinux':('tmpfs','tmpfs')}
read_only={'/','/tmp','/run/.containerenv','/run/secrets',
 '/etc/hosts','/etc/hostname','/etc/resolv.conf','/sys','/sys/fs/cgroup',
 '/proc/acpi','/proc/scsi','/proc/kcore','/proc/keys',
 '/proc/latency_stats','/proc/timer_list','/proc/interrupts',
 '/proc/bus','/proc/fs','/proc/irq','/proc/sys','/proc/sysrq-trigger',
 '/sys/firmware','/sys/fs/selinux'}
for m in parsed:
 if (m['point'] not in expected or
    (m['fstype'],m['source'])!=expected[m['point']]):raise RuntimeError('unreviewed container mount: '+repr(m))
 if m['point'] in read_only and 'ro' not in m['options']:raise RuntimeError('runtime mount not read only')
 if m['point'] in ('/work','/tmp','/run','/var/tmp') and 'nodev' not in m['options']:raise RuntimeError('runtime scratch permits devices')
 if m['point']=='/work' and 'noexec' not in m['options']:raise RuntimeError('source staging permits native executables')
 if m['point']=='/run/secrets':
  if list((base/'root/run/secrets').iterdir()):raise RuntimeError('container secret mount contains files')
if len({m['point'] for m in parsed})!=len(parsed):raise RuntimeError('duplicate mount destination')
if not {'/','/work','/tmp','/dev','/proc','/sys','/sys/fs/cgroup','/run/secrets'}.issubset({m['point'] for m in parsed}):raise RuntimeError('missing required sandbox mounts')
names={n:os.readlink(str(base/'ns'/n)) for n in ('user','pid','mnt','net')}
hostnames={n:os.readlink('/proc/self/ns/'+n) for n in ('user','pid','mnt','net')}
result={'staged_program_sha256':hashlib.sha256(raw).hexdigest(),
 'immutable_executable_memfd':memfds[0],
 'staged_inode':{'device':file_stat.st_dev,'inode':file_stat.st_ino,'size':file_stat.st_size,'nlink':file_stat.st_nlink,'uid':file_stat.st_uid,'mode':file_stat.st_mode,'mtime_ns':file_stat.st_mtime_ns,'ctime_ns':file_stat.st_ctime_ns},
 'process_count':len(procs),'process_status':status,'container_pid':pid,
 'container_pid_start_ticks':start_ticks,
 'cgroup_path':str(cgroup.resolve()),'cgroup_procs':procs,
 'pids_max':(cgroup/'pids.max').read_text().strip(),
 'memory_max':(cgroup/'memory.max').read_text().strip(),
 'net_interfaces':ifaces,'ipv4_routes':ipv4,'ipv6_routes':ipv6,'mount_inventory':parsed,'child_ns':names,'observer_ns':hostnames,
 'read_only_root':bool(os.statvfs(base/'root').f_flag&os.ST_RDONLY),
 'read_only_tmp':bool(os.statvfs(base/'root/tmp').f_flag&os.ST_RDONLY),
 'writable_work':not bool(os.statvfs(base/'root/work').f_flag&os.ST_RDONLY),
 'interpreter_sha256':hashlib.sha256((base/'root/usr/local/bin/python3.13').read_bytes()).hexdigest()}
print(json.dumps(result,sort_keys=True))
'''

# 2026-10-27 independent worker-13 source review APPROVED these exact V38
# research-only sources after the actual native sealed-memfd holdout, negative
# controls and post-witness adversarial candidate.py replacement (the original
# immutable sealed source still executed). This authorizes ONLY research
# scoring of source-pinned original model-generated Q4/Q8 Python in an isolated
# Podman VM, never frozen v2 Gate2, Gen-0 selection or release.
APPROVED_V38_RESEARCH_LOADER_SHA256: str | None = "3319565d5d5a6aa210151a32bd367a136a08cfb530402c51f311143de2f08e66"
APPROVED_V38_RESEARCH_HOST_OBSERVER_SHA256: str | None = "c4e5fd10ae77338337357ee250d8f5e19d925f74cf520c42b2e25b94cc519425"


def _research_source_gaps() -> list[str]:
    gaps = []
    if (APPROVED_V38_RESEARCH_LOADER_SHA256 is None
            or APPROVED_V38_RESEARCH_LOADER_SHA256 != _sha(V38_LOADER.encode())):
        gaps.append("INDEPENDENT_V38_GENERATED_CODE_LOADER_SOURCE_REVIEW_REQUIRED")
    if (APPROVED_V38_RESEARCH_HOST_OBSERVER_SHA256 is None
            or APPROVED_V38_RESEARCH_HOST_OBSERVER_SHA256 != _sha(V38_VM_SOURCE_OBSERVER.encode())):
        gaps.append("INDEPENDENT_V38_OUTSIDE_HOST_OBSERVER_SOURCE_REVIEW_REQUIRED")
    return gaps


def original_research_admission() -> None:
    """Does not alter or bypass frozen v2 source-pinned Gate2 approvals."""
    v37.verify_negative_source_pins()
    if (v37.IMAGE_ID != V37_IMAGE_ID or v37.PYTHON_SHA256 != V37_PYTHON_EXECUTABLE_SHA256):
        raise RuntimeError("original research OCI/executable source pins changed")
    gaps = _research_source_gaps()
    if gaps:
        raise RuntimeError("REFUSED unreviewed actual generated-code executor: " + ", ".join(gaps))
    host_safety_guard()
    v37.preflight()


def _bounded_exec(args: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    if len(result.stdout) > 32768 or len(result.stderr) > 32768:
        raise RuntimeError("unbounded original research subprocess output")
    return result


def _source_stage_host_observe(owner_pid: int, expected_sha: str, sealed_fd: int) -> dict:
    if type(owner_pid) is not int or owner_pid <= 1 or not re.fullmatch("[0-9a-f]{64}", expected_sha):
        raise ValueError("invalid independent source owner binding")
    if type(sealed_fd) is not int or not 3 <= sealed_fd <= 64:
        raise ValueError("invalid immutable executable handle")
    token = base64.b64encode(V38_VM_SOURCE_OBSERVER.encode("utf-8")).decode("ascii")
    command = ("sudo -n python3 -c 'import base64;exec(base64.b64decode(\"" + token
               + "\").decode(\"utf-8\"))' " + str(owner_pid) + " " + str(sealed_fd))
    result = _bounded_exec(["podman", "machine", "ssh", "--", command], timeout=15)
    if result.returncode:
        raise RuntimeError("independent VM-host candidate.py source observer unavailable: "
                           + result.stderr[-1600:])
    evidence = v37._strict_json(result.stdout)
    ipv4 = evidence.get("ipv4_routes")
    ipv6 = evidence.get("ipv6_routes")
    mounts = evidence.get("mount_inventory")
    inode = evidence.get("staged_inode")
    sealed = evidence.get("immutable_executable_memfd")
    allowed_mounts = {
        "/": ("overlay", "overlay"), "/work": ("tmpfs", "tmpfs"),
        "/tmp": ("tmpfs", "tmpfs"), "/run": ("tmpfs", "tmpfs"),
        "/var/tmp": ("tmpfs", "tmpfs"), "/run/.containerenv": ("tmpfs", "tmpfs"),
        "/run/secrets": ("tmpfs", "tmpfs"),
        "/etc/hosts": ("tmpfs", "tmpfs"), "/etc/hostname": ("tmpfs", "tmpfs"),
        "/etc/resolv.conf": ("tmpfs", "tmpfs"),
        "/dev": ("tmpfs", "tmpfs"), "/dev/pts": ("devpts", "devpts"),
        "/dev/shm": ("tmpfs", "shm"), "/dev/mqueue": ("mqueue", "mqueue"),
        "/dev/null": ("devtmpfs", "none"), "/dev/zero": ("devtmpfs", "none"),
        "/dev/full": ("devtmpfs", "none"), "/dev/tty": ("devtmpfs", "none"),
        "/dev/random": ("devtmpfs", "none"), "/dev/urandom": ("devtmpfs", "none"),
        "/proc": ("proc", "proc"), "/proc/bus": ("proc", "proc"),
        "/proc/fs": ("proc", "proc"), "/proc/irq": ("proc", "proc"),
        "/proc/sys": ("proc", "proc"), "/proc/sysrq-trigger": ("proc", "proc"),
        "/proc/acpi": ("tmpfs", "tmpfs"), "/proc/scsi": ("tmpfs", "tmpfs"),
        "/proc/kcore": ("devtmpfs", "none"), "/proc/keys": ("devtmpfs", "none"),
        "/proc/latency_stats": ("devtmpfs", "none"),
        "/proc/timer_list": ("devtmpfs", "none"),
        "/proc/interrupts": ("devtmpfs", "none"),
        "/sys": ("sysfs", "sysfs"), "/sys/fs/cgroup": ("cgroup2", "cgroup2"),
        "/sys/firmware": ("tmpfs", "tmpfs"), "/sys/fs/selinux": ("tmpfs", "tmpfs"),
    }
    if (not isinstance(ipv4, list) or not ipv4
            or not isinstance(ipv4[0], str)
            or ipv4[0].split()[:2] != ["Iface", "Destination"]
            or any(not isinstance(row, str) or len(row.split()) < 11
                   or row.split()[0] != "lo" or row.split()[1] == "00000000"
                   for row in ipv4[1:])
            or not isinstance(ipv6, list)
            or any(not isinstance(row, str) or len(row.split()) < 10
                   or row.split()[-1] != "lo" for row in ipv6)
            or not isinstance(mounts, list) or len(mounts) < 3
            or any(not isinstance(mount, dict)
                   or mount.get("point") not in allowed_mounts
                   or (mount.get("fstype"), mount.get("source"))
                       != allowed_mounts[mount["point"]] for mount in mounts)
            or len({m.get("point") for m in mounts}) != len(mounts)
            or not isinstance(inode, dict)
            or any(type(inode.get(k)) is not int for k in
                   ("device", "inode", "size", "nlink", "uid", "mode", "mtime_ns", "ctime_ns"))
            or inode["device"] <= 0 or inode["inode"] <= 0 or inode["size"] <= 0
            or inode["nlink"] != 1 or inode["uid"] == 0 or inode["mode"] & 0o022):
        raise RuntimeError("independent VM-host route/mount/staged-inode witness failed")
    if (not isinstance(sealed, dict)
            or sealed.get("sha256") != expected_sha or sealed.get("fd") != sealed_fd
            or type(sealed.get("pid")) is not int
            or sealed["pid"] not in evidence.get("cgroup_procs", [])
            or type(sealed.get("inode")) is not int or sealed["inode"] <= 0
            or sealed.get("seals") != 15):
        raise RuntimeError("independent VM-host immutable-executable memfd witness failed")
    if (evidence.get("staged_program_sha256") != expected_sha
            or type(evidence.get("container_pid_start_ticks")) is not int
            or evidence["container_pid_start_ticks"] <= 0
            or evidence.get("interpreter_sha256") != V37_PYTHON_EXECUTABLE_SHA256
            or evidence.get("read_only_root") is not True
            or evidence.get("read_only_tmp") is not True
            or evidence.get("writable_work") is not True
            or evidence.get("net_interfaces") != ["lo"]
            or evidence.get("pids_max") != "16"
            or evidence.get("memory_max") != "268435456"
            or type(evidence.get("process_count")) is not int
            or not 2 <= evidence["process_count"] <= 16):
        raise RuntimeError("independent VM-host exact staged program/memory/no-egress witness failed")
    if (not isinstance(evidence.get("process_status"), list)
            or any(record.get("capeff") != "0000000000000000"
                   or record.get("nnp") != "1" or record.get("seccomp") != "2"
                   or any(part == "0" for part in record.get("uid", "").split())
                   for record in evidence["process_status"])):
        raise RuntimeError("independent VM-host per-process rootless/caps/seccomp proof failed")
    for namespace in ("user", "pid", "mnt", "net"):
        if evidence.get("child_ns", {}).get(namespace) == evidence.get("observer_ns", {}).get(namespace):
            raise RuntimeError("VM research original code namespace shares host")
    return evidence


def _new_research_container_argv(name: str) -> list[str]:
    if not re.fullmatch(r"w10-he-v38-[a-f0-9]{16}", name):
        raise ValueError("not exclusive generated-code research owner")
    return ["podman", "run", "--detach", "--pull=never", "--name", name,
            "--network=none", "--cgroupns=private", "--pid=private", "--ipc=private",
            "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
            "--user=65534:65534", "--userns=keep-id", "--passwd=false", "--pids-limit=16",
            "--memory=268435456", "--memory-swap=268435456", "--ulimit=core=0:0",
            "--tmpfs=/work:rw,nosuid,nodev,noexec,size=16m,mode=1777",
            "--tmpfs=/tmp:ro,nosuid,nodev,noexec,size=64k,mode=0555",
            "--tmpfs=/run/secrets:ro,nosuid,nodev,noexec,size=64k,mode=0555,notmpcopyup",
            "--env=PYTHONDONTWRITEBYTECODE=1", "--workdir=/work",
            f"--entrypoint={v37.PYTHON_PATH}", V37_IMAGE_ID,
            "-I", "-c", "import time;time.sleep(45)"]


def _pipe_line_with_timeout(pipe, *, seconds: float = 12) -> bytes:
    received = queue.Queue(maxsize=1)
    threading.Thread(target=lambda: received.put(pipe.readline()), daemon=True).start()
    try:
        line = received.get(timeout=seconds)
    except queue.Empty as exc:
        raise RuntimeError("fixed bounded candidate transport/status watchdog expired") from exc
    if not isinstance(line, bytes) or not 0 < len(line) <= 16_384:
        raise RuntimeError("unbounded/missing isolated candidate status line")
    return line


def _run_research_candidate(program: str, *, fixed_holdout_only: bool = False) -> dict:
    """Execute original program ONLY after independent V38 admission.

    Before release, the outside-container VM host independently reads and
    SHA-verifies the staged candidate.py. Candidate output is not a verifier;
    the original official HumanEval tests are appended by exact_program().
    Any infrastructure/teardown failure aborts the cohort; never score zero.
    """
    if fixed_holdout_only:
        if program != V38_FIXED_HOLDOUT or _sha(program.encode()) != V38_FIXED_HOLDOUT_SHA256:
            raise ValueError("only exact fixed benign holdout before admission")
        v37.verify_negative_source_pins()
        v37.preflight()
        host_safety_guard()
    else:
        original_research_admission()
    if not isinstance(program, str) or not 0 < len(program.encode("utf-8")) <= 2 * 1024 * 1024:
        raise ValueError("candidate bytes unbounded before isolated transport")
    digest = _sha(program.encode("utf-8"))
    name = "w10-he-v38-" + uuid.uuid4().hex[:16]
    exists = _bounded_exec(["podman", "container", "exists", name], timeout=10)
    if exists.returncode != 1:
        raise RuntimeError("untrusted preexisting container owner")
    owner = None
    observed = None
    record = None
    post = None
    refused = None
    owned_removed = False
    transport = None
    try:
        created = _bounded_exec(_new_research_container_argv(name), timeout=16)
        if created.returncode != 0 or not re.fullmatch(r"[0-9a-f]{64}", created.stdout.strip()):
            raise RuntimeError("exclusive no-egress rootless research container refused")
        owner = created.stdout.strip()
        info = v37._inspect_owned(name, owner)
        if info.get("Image") != V37_IMAGE_ID:
            raise RuntimeError("actual research container image ID mismatches source pin")
        vm_pid = info.get("State", {}).get("Pid")
        if type(vm_pid) is not int or vm_pid <= 1:
            raise RuntimeError("unique research container VM root PID missing")
        command = ["podman", "exec", "-i", name, v37.PYTHON_PATH, "-I", "-c",
                   V38_LOADER, digest]
        transport = subprocess.Popen(command, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if transport.stdin is None or transport.stdout is None:
            raise RuntimeError("isolated source transfer is not piped")
        transport.stdin.write(program.encode("utf-8"))
        transport.stdin.flush()
        transport.stdin.close()
        first = v37._strict_json(_pipe_line_with_timeout(transport.stdout).decode("utf-8"))
        sealed_fd = first.get("sealed_fd")
        if (first.get("stage") != "PREEXEC_SOURCE_STAGED"
                or first.get("sha256") != digest or first.get("sealed_sha256") != digest
                or type(sealed_fd) is not int or not 3 <= sealed_fd <= 64
                or len(first) != 4):
            raise RuntimeError("outside-host preexecution candidate SHA does not match staged loader")
        observed = _source_stage_host_observe(vm_pid, digest, sealed_fd)
        if owner not in observed.get("cgroup_path", "") or vm_pid != observed.get("container_pid"):
            raise RuntimeError("real VM-host staged source owner mismatch")
        permission = _bounded_exec(["podman", "exec", name, v37.PYTHON_PATH, "-I", "-c",
                                    "from pathlib import Path;Path('/work/release').write_text('go')"],
                                   timeout=10)
        if permission.returncode != 0:
            raise RuntimeError("source-verified research candidate release refused")
        record = v37._strict_json(_pipe_line_with_timeout(transport.stdout, seconds=19).decode("utf-8"))
        if (record.get("stage") != "ORIGINAL_CANDIDATE_FINISHED"
                or record.get("sha256") != digest
                or record.get("generated_model_candidate_executed") is not True
                or type(record.get("passed")) is not bool
                or type(record.get("timed_out")) is not bool
                or type(record.get("candidate_exit")) is not int):
            raise RuntimeError("original candidate process evidence incomplete/forged")
        if transport.wait(timeout=6) != 0:
            raise RuntimeError("isolated candidate launcher failed after exit evidence")
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        refused = type(error).__name__ + ": " + str(error)[:1600]
    finally:
        if transport is not None and transport.poll() is None:
            transport.kill()
            try:
                transport.wait(timeout=4)
            except subprocess.TimeoutExpired:
                refused = "HOST_TRANSPORT_PROCESS_NOT_REAPED"
        if owner is not None:
            try:
                if _bounded_exec(["podman", "container", "exists", name], timeout=10).returncode != 0:
                    raise RuntimeError("owned container disappeared before audited teardown")
                v37._inspect_owned(name, owner)
                removed = _bounded_exec(["podman", "rm", "--force", name], timeout=12)
                gone = _bounded_exec(["podman", "container", "exists", name], timeout=10)
                owned_removed = removed.returncode == 0 and gone.returncode == 1
                if not owned_removed:
                    raise RuntimeError("original candidate cgroup owner not removed")
            except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
                refused = "OWNED_RUNTIME_TEARDOWN_UNVERIFIED: " + type(error).__name__
    if observed is not None:
        try:
            post = v37._vm_postcheck({"child": {"pid": observed["container_pid"],
                                                "start_ticks": observed["container_pid_start_ticks"]},
                                       "descendants": {}, "cgroup_path": observed["cgroup_path"]})
        except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
            refused = "OUTSIDE_HOST_PROCESS_TREE_EXTINCTION_UNVERIFIED: " + type(error).__name__
    if refused is not None or not owned_removed or post is None or record is None:
        raise RuntimeError("REFUSED original generated candidate execution: " + str(refused))
    return {"source_program_sha256": digest, "source_owner_id": owner,
            "research_only_executable_source_sha256": _sha(V38_LOADER.encode()),
            "outside_vm_host_observer_sha256": _sha(V38_VM_SOURCE_OBSERVER.encode()),
            "outside_host_exact_staged_source_witness": observed,
            "external_owned_process_extinction": post,
            "execution_status": "RESEARCH_ONLY_SOURCE_BOUND_NOT_FROZEN_GATE2",
            "independent_research_correct": record["passed"],
            "candidate_exit": record["candidate_exit"], "timed_out": record["timed_out"],
            "frozen_v2_approved": False, "gen0_status": "BLOCKED"}




@dataclass(frozen=True)
class OriginalArm:
    name: str
    original_model_sha256: str
    source: Path
    source_sha256: str
    session: str
    original_recorded_correct: int


ARMS = {
    "q4": OriginalArm("reap128_v2_q4", Q4_SHA, Q4_ROWS, Q4_ROW_SHA256, Q4_SESSION, 151),
    "q8": OriginalArm("reap128_v2_q8", Q8_SHA, Q8_ROWS, Q8_ROW_SHA256, Q8_SESSION, 152),
}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _bound_source(path: Path, expected_sha: str, byte_limit: int) -> bytes:
    """Only exact original F: source bytes; no data substitution by CLI."""
    if (not isinstance(path, Path) or not path.is_absolute() or path.is_symlink()
            or not path.is_file() or not 0 < path.stat().st_size <= byte_limit):
        raise ValueError("missing, linked, nonoriginal or oversized private HumanEval source")
    raw = path.read_bytes()
    if _sha(raw) != expected_sha:
        raise ValueError("actual original HumanEval bytes differ from source-pinned SHA256")
    return raw


def _read_canonical() -> list[dict]:
    raw = _bound_source(CANONICAL, CANONICAL_SHA256, SOURCE_LIMIT)
    tasks = _strict_json(raw)
    if (not isinstance(tasks, list) or len(tasks) != ORIGINAL_POPULATION
            or any(not isinstance(item, dict) for item in tasks)
            or [item.get("id") for item in tasks] != [f"HumanEval/{i}" for i in range(164)]):
        raise ValueError("canonical original HumanEval 164 IDs/order invalid")
    for item in tasks:
        if (not isinstance(item.get("entry_point"), str) or not isinstance(item.get("test"), str)
                or not isinstance(item.get("header"), str)
                or not isinstance(item.get("prompt"), str)
                or len(item["test"]) > 64_000):
            raise ValueError("original canonical HumanEval task contract invalid")
    return tasks


def original_program_manifest(arm: str) -> dict:
    """Complete original population audit; NO generated program is executed."""
    if arm not in ARMS:
        raise ValueError("only original complete Q4/Q8 arms are eligible")
    spec = ARMS[arm]
    canonical = _read_canonical()
    raw = _bound_source(spec.source, spec.source_sha256, 64 * 1024 * 1024)
    if not raw.endswith(b"\n"):
        raise ValueError("original JSONL output is not complete/flushed")
    rows = [_strict_json(line) for line in raw.splitlines()]
    if (len(rows) != ORIGINAL_POPULATION or any(not isinstance(r, dict) for r in rows)
            or {r.get("id") for r in rows} != {item["id"] for item in canonical}):
        raise ValueError("original 164 response identifiers/population missing")
    lookup: dict[str, dict] = {}
    for row in rows:
        ident = row.get("id")
        if ident in lookup:
            raise ValueError("duplicate original HumanEval candidate ID")
        if (row.get("eval_session_id") != spec.session
                or type(row.get("terminated")) is not bool
                or type(row.get("correct")) is not bool
                or (row["correct"] and not row["terminated"])
                or not isinstance(row.get("text"), str)
                or len(row["text"].encode("utf-8")) > ROW_LIMIT
                or type(row.get("completion_tokens")) is not int
                or not 0 <= row["completion_tokens"] <= 4096):
            raise ValueError("original candidate session/termination/source invariants broken")
        lookup[ident] = row
    if (sum(r["correct"] for r in rows) != spec.original_recorded_correct
            or sum(r["terminated"] for r in rows) != 162):
        raise ValueError("original source-bound recorded baseline does not reconcile")

    transcript = []
    for ordinal, item in enumerate(canonical):
        row = lookup[item["id"]]
        fingerprint = None
        if row["terminated"]:
            # Reproduce original public evaluator's candidate selection, with
            # the existing frozen source helper. Do not evaluate/import/run it.
            program = exact_program(item, row)
            if not 0 < len(program.encode("utf-8")) <= ROW_LIMIT:
                raise ValueError("candidate program text cannot be source bound")
            fingerprint = _sha(program.encode("utf-8"))
        transcript.append({"ordinal": ordinal, "id": item["id"],
                           "terminated": row["terminated"],
                           "original_recorded_correct_unverified": row["correct"],
                           "candidate_program_sha256": fingerprint,
                           "candidate_executed_in_v38": False})
    lineage = json.dumps(transcript, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return {"schema": "chowder.reap-v2-human164-original-program-identity.v38",
            "status": "COMPLETE_ORIGINAL_SOURCE_BINDING_ZERO_GENERATED_PROGRAMS_EXECUTED",
            "arm": spec.name, "original_model_sha256": spec.original_model_sha256,
            "original_serving_session": spec.session, "canonical_raw_sha256": CANONICAL_SHA256,
            "original_rows_raw_sha256": spec.source_sha256,
            "original_frozen_evaluator_sha256": EVALUATOR_SHA,
            "original_frozen_scorer_sha256": SCORER_SHA,
            "source_rows": ORIGINAL_POPULATION, "terminated_original_rows": 162,
            "unterminated_original_rows": 2,
            "original_recorded_research_correct_unverified": spec.original_recorded_correct,
            "v38_independently_executed_programs": 0,
            "v38_independent_verified_correct": None,
            "ordered_program_manifest_sha256": _sha(lineage),
            "source_rows_in_original_order": transcript,
            "v2_approved_execution": False, "gen0_status": "BLOCKED",
            "chosen": None, "release_approved": False}


def independent_execution_gaps() -> list[str]:
    """V37 fixed benign proof does not replace original source-approved pins."""
    gaps: list[str] = []
    gaps.extend(original_v2.execution_admission_gaps())
    if v37.EXECUTION_APPROVED is not False:
        gaps.append("V37_FIXED_BENIGN_ONLY_GATE_CHANGED_NOT_A_REPLAY_APPROVAL")
    if (v37.IMAGE_ID != V37_IMAGE_ID
            or v37.PYTHON_SHA256 != V37_PYTHON_EXECUTABLE_SHA256
            or v37.VM_HOST_OBSERVER_SHA256 != V37_EXTERNAL_HOST_OBSERVER_SHA256):
        gaps.append("ORIGINAL_V37_EXTERNAL_HOST_EXECUTABLE_OR_SOURCE_PIN_MISMATCH")
    try:
        v37.verify_negative_source_pins()
    except ValueError:
        gaps.append("V37_NEGATIVE_CANARY_OR_VM_HOST_SOURCE_SHA_NOT_PINNED")
    if not v37.EXECUTION_APPROVED:
        gaps.append("ONLY_FIXED_BENIGN_CANARIES_VERIFIED_UNTRUSTED_GENERATED_CODE_NOT_ADMITTED")
    return list(dict.fromkeys(gaps))


def require_independent_generated_code_admission() -> None:
    """Hard stop BEFORE opening candidate data for ANY execution route."""
    gaps = independent_execution_gaps()
    if gaps:
        raise RuntimeError("REFUSED original generated HumanEval execution: " + ", ".join(gaps))
    # This is NOT reachable in the frozen current source. A future independently
    # peer-reviewed runner must prove its own candidate-data transfer/profile
    # and source pin; do not silently treat frozen v2 approval as transferable.
    raise RuntimeError("REFUSED V38 original generated-code runner is not independently source-approved")


def execute_original_164(arm: str, *, manifest: dict | None = None) -> dict:
    """Never permit caller-supplied evidence, --force or a benign witness to execute code."""
    if arm not in ARMS or (manifest is not None and not isinstance(manifest, dict)):
        raise ValueError("unsupported original arm or forged execution manifest")
    require_independent_generated_code_admission()
    raise AssertionError("execution admitted without independent source-pinned runner")


def execute_original_research_164(arm: str, *, private_output: Path) -> dict:
    """Full original 164-case research ONLY, if independent V38 source approved.

    A container/cgroup/VM witness failure aborts the cohort immediately;
    unattempted rows are UNKNOWN, never invented failures. Only original
    naturally unterminated cases are source-bound skipped nonpasses.
    """
    if arm not in ARMS:
        raise ValueError("no nonoriginal model is eligible")
    original_research_admission()  # before source and output mutation
    if (not isinstance(private_output, Path) or not private_output.is_absolute()
            or private_output.exists() or private_output.is_symlink()
            or not private_output.name.startswith("reap-v2-human164-v38-private-")
            or not private_output.parent.is_dir()
            or str(private_output.parent).lower() not in ("f:\\", "f:/")):
        raise ValueError("only a fresh exclusive F: original private research output is allowed")
    manifest = original_program_manifest(arm)
    canonical = _read_canonical()
    raw = _bound_source(ARMS[arm].source, ARMS[arm].source_sha256, 64 * 1024 * 1024)
    rows = {_strict_json(row)["id"]: _strict_json(row) for row in raw.splitlines()}
    root = private_output
    root.mkdir(mode=0o700, exist_ok=False)
    (root / "original-input-manifest.json").write_text(
        json.dumps({key: value for key, value in manifest.items()
                    if key != "source_rows_in_original_order"}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    file = root / (arm + "-actual-independent-per-case.private.jsonl")
    source_by_id = {row["id"]: row for row in manifest["source_rows_in_original_order"]}
    correct = 0
    executed = 0
    for index, item in enumerate(canonical):
        original = rows[item["id"]]
        manifest_row = source_by_id[item["id"]]
        row = {"index": index, "id": item["id"], "arm": manifest["arm"],
               "source_session": manifest["original_serving_session"],
               "original_model_sha256": manifest["original_model_sha256"],
               "original_terminated": original["terminated"],
               "original_reported_correct_unverified": original["correct"],
               "program_sha256": manifest_row["candidate_program_sha256"],
               "frozen_v2_gate2_eligible": False}
        if not original["terminated"]:
            row["execution_status"] = "SOURCE_BOUND_UNTERMINATED_NOT_EXECUTED"
            row["independent_research_correct"] = False
        else:
            program = exact_program(item, original)
            if _sha(program.encode("utf-8")) != manifest_row["candidate_program_sha256"]:
                raise RuntimeError("original candidate source drift before isolated execution")
            witness = _run_research_candidate(program)
            row["execution_status"] = witness["execution_status"]
            row["independent_research_correct"] = witness["independent_research_correct"]
            row["live_external_vm_host_witness"] = witness
            executed += 1
        correct += int(row["independent_research_correct"])
        # Durable incremental journal, exclusive fresh directory and append
        # only. Cannot overwrite a previous agent's evidence or claim a score
        # for unexecuted later tasks on interruption.
        with file.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if (index + 1) % 10 == 0:
            print(f"V38_ORIGINAL_RESEARCH_{arm.upper()} {index + 1}/164 "
                  f"executed={executed} correct={correct}", flush=True)
        host_safety_guard()
    if executed != 162 or index != 163:
        raise RuntimeError("full source-bound original 164 population not executed")
    summary = {"schema": "chowder.reap-v2-original-human164-independent-research.v38",
               "status": "COMPLETE_164_ORIGINAL_CODE_ISOLATED_RESEARCH_ONLY_NOT_FROZEN_GATE2",
               "arm": manifest["arm"],
               "original_model_sha256": manifest["original_model_sha256"],
               "original_rows_sha256": manifest["original_rows_raw_sha256"],
               "canonical_sha256": CANONICAL_SHA256,
               "candidate_identity_manifest_sha256": manifest["ordered_program_manifest_sha256"],
               "original_rows": 164, "executed_rows": executed,
               "unterminated_rows_source_skipped": 164 - executed,
               "independently_research_scored_correct": correct,
               "independent_score": correct / 164,
               "per_case_evidence_sha256": _sha(file.read_bytes()),
               "original_v2_attestation_approved": False,
               "windows_native_gate_eligible": False,
               "gen0_status": "BLOCKED", "chosen": None, "release_approved": False}
    (root / (arm + "-complete.private.json")).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=["q4", "q8", "both"])
    parser.add_argument("--execute", action="store_true",
                        help="Only real independently source-approved sandbox may execute")
    parser.add_argument("--research-execute", action="store_true",
                        help="Independent source-approved NEW research runner, NOT frozen v2 Gate2")
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--fixed-holdout", action="store_true",
                        help="Execute only the statically pinned harmless candidate via exact V38 transport")
    args = parser.parse_args(argv)
    try:
        if sum(bool(v) for v in (args.execute, args.research_execute, args.fixed_holdout)) > 1:
            raise ValueError("choose exactly one execution scope")
        if args.fixed_holdout:
            result = _run_research_candidate(V38_FIXED_HOLDOUT, fixed_holdout_only=True)
            print(json.dumps({"status": "FIXED_BENIGN_NATIVE_SOURCE_HOLDOUT_ONLY",
                              "candidate_sha256": result["source_program_sha256"],
                              "host_observer_source_sha256": result["outside_vm_host_observer_sha256"],
                              "pass": result["independent_research_correct"],
                              "frozen_v2_gate2_eligible": False,
                              "gen0_status": "BLOCKED"}, sort_keys=True))
            return 0 if result["independent_research_correct"] else 2
        if args.research_execute:
            if args.private_output is None or args.arm == "both":
                raise ValueError("research execution requires one explicit arm and one fresh private F: output")
            summary = execute_original_research_164(args.arm, private_output=args.private_output)
            print(json.dumps(summary, sort_keys=True))
            return 0
        if args.execute:
            for name in ("q4", "q8") if args.arm == "both" else (args.arm,):
                execute_original_164(name)
        results = [original_program_manifest(name) for name in
                   (("q4", "q8") if args.arm == "both" else (args.arm,))]
        for result in results:
            result.pop("source_rows_in_original_order")  # no 164x2 private detail on stdout
        print(json.dumps({"status": "SOURCE_BOUND_ORIGINAL_164_EXECUTION_REFUSED_PENDING_INDEPENDENT_APPROVAL",
                          "arms": results, "actual_scored_164": False,
                          "execution_gaps": independent_execution_gaps(),
                          "gen0_status": "BLOCKED", "release_approved": False},
                         sort_keys=True))
        return 2
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print(json.dumps({"status": "REFUSED_ACTUAL_ORIGINAL_HUMANEVAL_EXECUTION",
                          "error_type": type(error).__name__,
                          "generated_programs_executed": 0,
                          "independent_score": None,
                          "reason": str(error)[:1800],
                          "gen0_status": "BLOCKED"}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
