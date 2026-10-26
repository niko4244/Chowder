"""CPU-only exact original private Q8 and random128 SHA attestation."""
import hashlib,json,os,pathlib,time,datetime
ROOT=pathlib.Path('/kaggle/input');WORK=pathlib.Path('/kaggle/working')
EXPECTED={
 'reap128_v2-q8_0-20261025.gguf':(19748435968,'89dcaf33b61f59c83a31b718ba98b73c33437d0320ae0cc56737a5d7e43519c6'),
 'random128-q8_0.gguf':(19748435872,'601871e00403658708eb4eebf3b05f9dc54a93b326e31f2f87b48cb59f92241b')
}
R={'schema':'reap-v2-private-original-q8-and-negative-control-remote-sha.v1','status':'REFUSED','without_model_inference':True,'no_scientific_quality_gate':True,'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':{}}
def available():
 with open('/proc/meminfo') as f:
  for s in f:
   if s.startswith('MemAvailable:'):
    return int(s.split()[1])/(1024*1024)
 raise RuntimeError('no physical available RAM')
def cgroup():
 path=pathlib.Path('/sys/fs/cgroup')
 limit=(path/'memory.max').read_text().strip()
 if limit=='max':raise RuntimeError('unbounded/missing container capacity evidence')
 return (int(limit)-int((path/'memory.current').read_text().strip()))/1024**3
try:
 for name,(size,want) in EXPECTED.items():
  matches=[p for p in ROOT.rglob(name) if p.is_file()]
  if len(matches)!=1 or matches[0].stat().st_size!=size:raise RuntimeError('missing/ambiguous/or wrong-size private model '+name)
  path=matches[0];h=hashlib.sha256();measured=0
  with path.open('rb',buffering=0) as f:
   while True:
    b=f.read(8*1024*1024)
    if not b:break
    measured+=len(b);h.update(b)
    if hasattr(os,'posix_fadvise') and hasattr(os,'POSIX_FADV_DONTNEED'):
     os.posix_fadvise(f.fileno(),max(0,measured-16*1024*1024),8*1024*1024,os.POSIX_FADV_DONTNEED)
    if measured%(512*1024*1024)<len(b):
     if available()<7 or cgroup()<8:raise RuntimeError('original host/cloud runtime safety floor approached while hashing')
   if hasattr(os,'posix_fadvise') and hasattr(os,'POSIX_FADV_DONTNEED'):os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
  digest=h.hexdigest()
  R['files'][name]={'bytes':measured,'sha256':digest,'passed':digest==want}
  if digest!=want or measured!=size:raise RuntimeError('private original byte/hash mismatch '+name)
 R['status']='PASS_ORIGINAL_PRIVATE_REMOTE_WEIGHT_INTEGRITY_NO_INFERENCE'
except Exception as exc:
 R['error']=type(exc).__name__+': '+str(exc)[:300]
with (WORK/'reap-v2-private-q8-random128-hash.json').open('x',encoding='utf8') as f:json.dump(R,f,indent=2,sort_keys=True)
print(json.dumps(R,sort_keys=True))