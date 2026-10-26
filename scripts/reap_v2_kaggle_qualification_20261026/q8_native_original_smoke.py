"""Guarded original-flags Q4 Kaggle CUDA smoke. No official gate claimed."""
import datetime, hashlib, json, os, pathlib, shutil, signal, subprocess, time, urllib.request
GIB=1024**3
ROOT=pathlib.Path('/kaggle/input');WORK=pathlib.Path('/kaggle/working')
ORIGINAL_GGUF_SHA='89dcaf33b61f59c83a31b718ba98b73c33437d0320ae0cc56737a5d7e43519c6'
ORIGINAL_RECEIPT_SHA='a1cfe1464df4579ea9d5ae1320efc01c0de5dae1070235caca18c821ce67479d'
PINNED_BUILD_SHA='f10032228f1d80b0f6639cd3bcdcc7194bbc6638368641cac7f1396775b5186d'
FLAGS=['--fit','on','-np','4','-c','16384','--jinja']
R={'schema':'reap-v2-q8-original-kaggle-admission-smoke.v1','status':'NOT_STARTED','scientific_selection':'UNKNOWN','not_official_windows_serving_attestation':True,'flags':FLAGS,'data':{},'samples':[],'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
server=None

def save():
    p=WORK/'reap-v2-kaggle-q8-native-smoke.json';q=p.with_name('.'+p.name+'.pending')
    with q.open('w',encoding='utf-8') as f:json.dump(R,f,indent=2,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(q,p)

def hash_file(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        # Read-only cache eviction after SHA verification: avoid pinning 11.56 GB
        # clean GGUF page-cache in a constrained Kaggle cgroup before CUDA load.
        # It does not modify or rewrite the immutable source model.
        if hasattr(os,'posix_fadvise') and hasattr(os,'POSIX_FADV_DONTNEED'):
            os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
    return h.hexdigest()

def source(name,size,digest):
    matches=[p for p in ROOT.rglob(name) if p.is_file()]
    if len(matches)!=1 or matches[0].stat().st_size!=size or hash_file(matches[0])!=digest:raise RuntimeError(name+' missing/size/hash mismatch')
    return matches[0]

def host():
    mem={}
    with open('/proc/meminfo') as f:
        for line in f:
            k,_,v=line.partition(':')
            if v.strip().endswith('kB'):mem[k]=int(v.strip().split()[0])/1048576
    available=mem.get('MemAvailable',0)
    cg=pathlib.Path('/sys/fs/cgroup')
    try:
        limit=(cg/'memory.max').read_text().strip()
        remain=(int(limit)-int((cg/'memory.current').read_text().strip()))/GIB if limit!='max' else None
    except Exception:remain=None
    gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15,check=True)
    free=[int(x.strip()) for x in gpu.stdout.splitlines()]
    return {'physical_free_gib':round(available,3),'cgroup_unused_gib':round(remain,3) if remain is not None else None,'gpu_free_mib':free}

def check_runtime(h):
    if h['physical_free_gib']<7:raise RuntimeError('runtime original 7 GiB free physical RAM floor breached')
    if h['cgroup_unused_gib'] is None or h['cgroup_unused_gib']<10:raise RuntimeError('conservative Q8 runtime 10 GiB cgroup guard before 8 GiB original floor')

def remove_owned():
    global server
    if server is not None and server.poll() is None:
        try:os.killpg(server.pid,signal.SIGTERM);server.wait(timeout=10)
        except Exception:
            try:os.killpg(server.pid,signal.SIGKILL);server.wait(timeout=5)
            except Exception:pass

try:
    weights=source('reap128_v2-q8_0-20261025.gguf',19748435968,ORIGINAL_GGUF_SHA)
    receipt=source('reap128-v2-q8_0-20261025.gguf.reap-chunked.json',191080,ORIGINAL_RECEIPT_SHA)
    gsm=source('gsm8k.json',484685,'4dae0860ba5b9785343070a4919504f0aa4eccb01a96108b5b851d23e059a522')
    rows=json.loads(gsm.read_text(encoding='utf-8'))
    if len(rows)!=1319:raise RuntimeError('frozen original GSM8K population changed')
    candidates=[x for x in ROOT.rglob('llama-server') if x.is_file()]
    verified=[x for x in candidates if x.stat().st_size==17888 and hash_file(x)==PINNED_BUILD_SHA]
    if len(verified)!=1:raise RuntimeError('original source-pinned Linux executable missing/ambiguous: '+str(len(verified)))
    sourcebin=verified[0].parent;livebin=WORK/'reap-q8-pinned-bin'
    if livebin.exists():raise RuntimeError('refusing overwrite of native bin stage')
    shutil.copytree(sourcebin,livebin)
    target=livebin/'llama-server';target.chmod(target.stat().st_mode | 0o111)
    env=dict(os.environ,LD_LIBRARY_PATH=str(livebin)+':'+os.environ.get('LD_LIBRARY_PATH',''),LLAMA_ARG_LOG_VERBOSITY='4')
    v=subprocess.run([str(target),'--version'],env=env,capture_output=True,text=True,timeout=60)
    R['server_version']={'code':v.returncode,'stdout':v.stdout[-1200:],'stderr':v.stderr[-1200:]}
    if v.returncode!=0 or '7609846' not in (v.stdout+v.stderr):raise RuntimeError('pinned CUDA server cannot execute or original version is not attested')
    start=host();R['initial']=start
    R['immutable_source_cache_evict_hint_used']=hasattr(os,'posix_fadvise') and hasattr(os,'POSIX_FADV_DONTNEED')
    if len(start['gpu_free_mib'])!=2 or any(x<12288 for x in start['gpu_free_mib']):raise RuntimeError('original CUDA admission fails on allocated GPU(s)')
    if start['physical_free_gib']<24 or start['cgroup_unused_gib'] is None or start['cgroup_unused_gib']<24:raise RuntimeError('guarded original Q4 model load host admission below 24 GiB')
    if shutil.disk_usage(WORK).free < 8*GIB:raise RuntimeError('original host runtime disk reserve below 8 GiB')
    R['data']={'q8_sha256':ORIGINAL_GGUF_SHA,'construction_sha256':ORIGINAL_RECEIPT_SHA,'full_official_gsm8k_n':len(rows),'score_scope':'one original Q8 first-item diagnostic only, not an official 1319/164 suite','server_binary_sha256':PINNED_BUILD_SHA}
    R['cmd']=[str(target),*FLAGS,'--port','8790','-m',str(weights)]
    log=WORK/'reap-v2-kaggle-q8-server.log'
    with log.open('xb') as fp:
        server=subprocess.Popen(R['cmd'],env=env,stdout=fp,stderr=subprocess.STDOUT,start_new_session=True)
        R['server_pid']=server.pid;R['status']='STARTED_UNQUALIFIED';save()
        boot=time.monotonic();ready=False;lastgpu=0
        while time.monotonic()-boot<420:
            h=host();h['seconds']=round(time.monotonic()-boot,1);R['samples'].append(h)
            check_runtime(h)
            if server.poll() is not None:raise RuntimeError('owned native model server exited early '+str(server.returncode))
            try:
                with urllib.request.urlopen('http://127.0.0.1:8790/health',timeout=2) as resp:
                    if resp.status==200:ready=True;break
            except Exception:pass
            time.sleep(3)
        if not ready:raise RuntimeError('original flags native GPU model did not become healthy in 420s')
        R['seconds_until_ready']=round(time.monotonic()-boot,1)
        R['before_inference_host']=host();check_runtime(R['before_inference_host'])
        prompt=rows[0]['prompt']
        body={'messages':[{'role':'system','content':'You are a helpful assistant.'},{'role':'user','content':prompt}],'temperature':0.0,'top_k':1,'seed':7,'max_tokens':1024,'chat_template_kwargs':{'enable_thinking':False}}
        req=urllib.request.Request('http://127.0.0.1:8790/v1/chat/completions',data=json.dumps(body).encode('utf-8'),headers={'Content-Type':'application/json'})
        t0=time.monotonic()
        with urllib.request.urlopen(req,timeout=600) as resp: raw=json.loads(resp.read())
        choice=raw['choices'][0]
        R['inference']={'seconds':round(time.monotonic()-t0,3),'finish_reason':choice.get('finish_reason'),'completion_tokens':raw.get('usage',{}).get('completion_tokens'),'timings':raw.get('timings'),'text':choice.get('message',{}).get('content'),'official_item_id':rows[0]['id']}
        R['after_inference_host']=host();check_runtime(R['after_inference_host'])
        R['status']='PROVISIONAL_ORIGINAL_Q8_SMOKE_COMPLETE_NOT_OFFICIAL_SCORE';save()
except Exception as e:
    R['status']='REFUSED_OR_ABORTED';R['error']=str(e)[:500];save()
finally:
    remove_owned()
    try:
        logfile=WORK/'reap-v2-kaggle-q8-server.log'
        if logfile.is_file():
            data=logfile.read_bytes()
            R['server_log_sha256']=hashlib.sha256(data).hexdigest()
            R['server_log_bytes']=len(data)
            R['server_log_excerpt']=data.decode('utf-8',errors='replace')[-2600:]
    except Exception:pass
    save()
    print(json.dumps({k:v for k,v in R.items() if k not in ('samples','inference','server_log_excerpt')},sort_keys=True))