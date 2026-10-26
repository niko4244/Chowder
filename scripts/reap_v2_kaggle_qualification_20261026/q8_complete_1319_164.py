"""Guarded original-flags Q4 Kaggle CUDA smoke. No official gate claimed."""
import concurrent.futures as cf, datetime, hashlib, importlib.util, json, os, pathlib, shutil, signal, subprocess, sys, threading, time, types, urllib.request, uuid
GIB=1024**3
ROOT=pathlib.Path('/kaggle/input');WORK=pathlib.Path('/kaggle/working')
ORIGINAL_GGUF_SHA='89dcaf33b61f59c83a31b718ba98b73c33437d0320ae0cc56737a5d7e43519c6'
ORIGINAL_RECEIPT_SHA='a1cfe1464df4579ea9d5ae1320efc01c0de5dae1070235caca18c821ce67479d'
PINNED_BUILD_SHA='f10032228f1d80b0f6639cd3bcdcc7194bbc6638368641cac7f1396775b5186d'
FLAGS=['--fit','on','-np','4','-c','16384','--jinja']
R={'schema':'reap-v2-q8-original-kaggle-complete-provisional-eval.v1','status':'NOT_STARTED','scientific_selection':'UNKNOWN','not_official_windows_serving_attestation':True,'flags':FLAGS,'data':{},'samples':[],'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
server=None

def save():
    p=WORK/'reap-v2-kaggle-q8-full-native.json';q=p.with_name('.'+p.name+'.pending')
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
    if h['cgroup_unused_gib'] is None or h['cgroup_unused_gib']<10:raise RuntimeError('conservative original Q8 10 GiB cgroup abort before 8 GiB original floor')

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
    he=source('humaneval.json',274061,'5144d31cf337b25ce262b780ed103b128b1e9d289d3c0d15f4fe34933bb705d5')
    human_rows=json.loads(he.read_text(encoding='utf-8'))
    if len(rows)!=1319 or len(human_rows)!=164:raise RuntimeError('frozen original complete suite populations changed')
    evals=[p for p in ROOT.rglob('reap_eval.py') if 'reap128v2-original-scorer-20261026' in str(p)]
    scores=[p for p in ROOT.rglob('scoring.py') if 'reap128v2-original-scorer-20261026' in str(p)]
    if len(evals)!=1 or len(scores)!=1:raise RuntimeError('original evaluator/scorer expanded source absent or ambiguous')
    eval_sha='9736714f21834a45c23599a50fa92ef3bebc6cf70967aca7e2329d26b4d73ee1'
    score_sha='d2e59aca5c0a621bac8d4979d522c9b6107f811610567d21c92a2b53960860eb'
    if hash_file(evals[0])!=eval_sha or hash_file(scores[0])!=score_sha:raise RuntimeError('frozen original scoring source SHA mismatch')
    pkg=types.ModuleType('chowder');pkg.__path__=[];evpkg=types.ModuleType('chowder.evaluators');evpkg.__path__=[]
    sys.modules.update({'chowder':pkg,'chowder.evaluators':evpkg})
    def import_locked(name,path):
        spec=importlib.util.spec_from_file_location(name,str(path))
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
        return module
    scoring=import_locked('chowder.evaluators.scoring',scores[0]);original=import_locked('reap_eval_locked',evals[0])
    if original._digest(rows)!='913ca81f724cba16a07099608047e7522b00910c1c7fe13499c501ac9e4bad89':raise RuntimeError('frozen GSM8K rendered input digest mismatch')
    if original._digest(human_rows)!='49fb688c6788b0531803dfd1cb85a69926ec5a7179e74cdfffc53ba52a92b420':raise RuntimeError('frozen HumanEval rendered input digest mismatch')
    R['source_sha256']={'eval':eval_sha,'score':score_sha,'gsm8k_full':'913ca81f724cba16a07099608047e7522b00910c1c7fe13499c501ac9e4bad89','humaneval_full':'49fb688c6788b0531803dfd1cb85a69926ec5a7179e74cdfffc53ba52a92b420'}

    candidates=[x for x in ROOT.rglob('llama-server') if x.is_file()]
    verified=[x for x in candidates if x.stat().st_size==17888 and hash_file(x)==PINNED_BUILD_SHA]
    if len(verified)!=1:raise RuntimeError('original source-pinned Linux executable missing/ambiguous: '+str(len(verified)))
    sourcebin=verified[0].parent;livebin=WORK/'reap-q8-full-pinned-bin'
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
    R['data']={'q8_sha256':ORIGINAL_GGUF_SHA,'construction_sha256':ORIGINAL_RECEIPT_SHA,'full_official_gsm8k_n':len(rows),'score_scope':'full 1319+164, separately reported PROVISIONAL due no Windows-original attestation','server_binary_sha256':PINNED_BUILD_SHA}
    R['cmd']=[str(target),*FLAGS,'--port','8790','-m',str(weights)]
    log=WORK/'reap-v2-kaggle-q8-full-server.log'
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
        # A new, single native serving session owns all rows; never adopt
        # rows from previous Kaggle notebook versions or aborted Windows tests.
        session=uuid.uuid4().hex
        R['session_id']=session
        import psutil
        def check_owner():
            if server.poll() is not None:raise RuntimeError('owned model server died')
            argv=psutil.Process(server.pid).cmdline()
            if argv!=R['cmd']:raise RuntimeError('native live PID command/flags/model differed')
            listeners={x.pid for x in psutil.net_connections(kind='tcp') if x.status==psutil.CONN_LISTEN and x.laddr and x.laddr.port==8790 and x.laddr.ip in ('127.0.0.1','0.0.0.0','::1','::')}
            if listeners!={server.pid}:raise RuntimeError('native server PID does not own loopback 8790')
        check_owner()
        R['status']='PROVISIONAL_FULL_COHORT_STARTED'
        save()
        stop=threading.Event();fatal=[]
        def runtime_guard():
            while not stop.is_set():
                try:
                    h=host();h['time_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    R['samples'].append(h)
                    check_runtime(h)
                    check_owner()
                except Exception as ex:
                    fatal.append(str(ex))
                    remove_owned()
                    break
                stop.wait(5)
        guard=threading.Thread(target=runtime_guard,name='source-bound-resource-supervisor',daemon=True)
        guard.start()
        try:
            R['suite_results']={}
            for suite,items in [('gsm8k',rows),('humaneval',human_rows)]:
                check_owner(); check_runtime(host())
                path=WORK/('reap-q8-full-'+session+'-'+suite+'.jsonl')
                answers={};suite_t0=time.monotonic()
                with path.open('x',encoding='utf-8') as handle,cf.ThreadPoolExecutor(max_workers=4) as pool:
                    jobs={pool.submit(original._chat,8790,it['prompt'],original.MAX_TOKENS[suite]):it for it in items}
                    for n,fut in enumerate(cf.as_completed(jobs),1):
                        if fatal:raise RuntimeError('resource owner abort: '+fatal[0])
                        item=jobs[fut]
                        row=original._score(suite,item,fut.result())
                        row['eval_session_id']=session
                        answers[row['id']]=row
                        handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                        handle.flush()
                        if n%20==0:os.fsync(handle.fileno())
                        if n%50==0:
                            R['progress']={'suite':suite,'finished':n,'required':len(items)}
                            check_runtime(host());check_owner();save()
                            print('PROGRESS',suite,n,'/',len(items),flush=True)
                    os.fsync(handle.fileno())
                if fatal:raise RuntimeError('resource owner abort: '+fatal[0])
                if len(answers)!=len(items) or set(answers)!={it['id'] for it in items}:raise RuntimeError('incomplete/noncanonical row population')
                check_owner();check_runtime(host())
                ordered=[answers[it['id']] for it in items]
                # Original deterministic scorer summary from the exact source;
                # R remains explicitly Kaggle-provisional, NOT selectable v2.
                result=original._summarize_rows(ordered,path)
                result['suite_elapsed_wall_seconds']=round(time.monotonic()-suite_t0,3)
                result['session_id']=session
                R['suite_results'][suite]=result;save()
            R['after_inference_host']=host();check_runtime(R['after_inference_host'])
            check_owner()
            R['status']='COMPLETE_Q8_FULL_DATA_PROVISIONAL_NOT_FROZEN_WINDOWS_GATE'
            save()
        finally:
            stop.set();guard.join(timeout=12)

except Exception as e:
    R['status']='REFUSED_OR_ABORTED';R['error']=str(e)[:500];save()
finally:
    remove_owned()
    try:
        logfile=WORK/'reap-v2-kaggle-q8-full-server.log'
        if logfile.is_file():
            data=logfile.read_bytes()
            R['server_log_sha256']=hashlib.sha256(data).hexdigest()
            R['server_log_bytes']=len(data)
            R['server_log_excerpt']=data.decode('utf-8',errors='replace')[-2600:]
    except Exception:pass
    save()
    print(json.dumps({k:v for k,v in R.items() if k not in ('samples','inference','server_log_excerpt')},sort_keys=True))