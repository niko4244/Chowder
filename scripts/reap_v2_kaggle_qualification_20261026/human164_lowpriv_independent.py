"""Independent low-privilege CPU replay, never loads a model or external network."""
import datetime,hashlib,json,os,pathlib,re,resource,subprocess,sys,tempfile,time,zipfile
ROOT=pathlib.Path('/kaggle/input'); WORK=pathlib.Path('/kaggle/working')
ROW_HASH='92ec6439be77a7dbeb1a80448a0bb664e3cb6113a9cc0d356f7dbe300db69c43'
SOURCE_FILE_HASH='5144d31cf337b25ce262b780ed103b128b1e9d289d3c0d15f4fe34933bb705d5'
ORIGINAL_EVAL_HASH='9736714f21834a45c23599a50fa92ef3bebc6cf70967aca7e2329d26b4d73ee1'
ORIGINAL_SCORER_HASH='d2e59aca5c0a621bac8d4979d522c9b6107f811610567d21c92a2b53960860eb'
SESSION='a2a6b83e345145839cc333561b2f3baf'
RESULT={'schema':'reap128-v2-humaneval-independent-low-privilege-python3.13-replay.v1','status':'REFUSED','original_windows_gate_eligible':False,'no_model_loaded':True,'no_user_secrets_requested':True,'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def unique(name):
 paths=[p for p in ROOT.rglob(name) if p.is_file()]
 if len(paths)!=1:raise RuntimeError('missing or ambiguous independent input: '+name)
 return paths[0]
def preexec():
 # The model-generated candidate runs with neither Kaggle credentials nor
 # permission to write the root-owned source, rows or final result.
 resource.setrlimit(resource.RLIMIT_CPU,(10,11))
 resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
 resource.setrlimit(resource.RLIMIT_FSIZE,(1024**2,1024**2))
 resource.setrlimit(resource.RLIMIT_CORE,(0,0))
 if os.geteuid()==0:
  os.setgroups([]);os.setgid(65534);os.setuid(65534)
def run(program):
 with tempfile.TemporaryDirectory(prefix='humaneval-lowpriv-') as tmp:
  parent=pathlib.Path(tmp)
  parent.chmod(0o755)
  path=parent/'candidate.py';path.write_text(program,encoding='utf-8');path.chmod(0o644)
  try:
   with open(os.devnull,'wb') as sink:
    finished=subprocess.run([sys.executable,'-I',str(path)],cwd=tmp,stdin=subprocess.DEVNULL,stdout=sink,stderr=sink,timeout=12,preexec_fn=preexec,env={'PATH':'/usr/bin:/bin','PYTHONNOUSERSITE':'1'})
   return finished.returncode==0,'exit'+str(finished.returncode)
  except subprocess.TimeoutExpired:
   return False,'timeout'
try:
 raw=unique('humaneval-original-output.jsonl');src=unique('humaneval.json')
 if digest(raw)!=ROW_HASH or digest(src)!=SOURCE_FILE_HASH:raise RuntimeError('original source/row SHA mismatch')
 data=json.loads(src.read_text(encoding='utf8'));rows=[json.loads(line) for line in raw.read_text(encoding='utf8').splitlines()]
 if len(data)!=164 or len(rows)!=164:raise RuntimeError('original full HumanEval cohort missing')
 by_id={x['id']:x for x in rows}
 if len(by_id)!=164 or {x['id'] for x in data}!=set(by_id):raise RuntimeError('duplicate/missing task ID')
 if any(x['eval_session_id']!=SESSION or type(x['correct']) is not bool or type(x['terminated']) is not bool for x in rows):raise RuntimeError('session/correctness proof corrupted')
 scorer_sources=[p for p in ROOT.rglob('scoring.py') if 'reap128v2-original-scorer-20261026' in str(p)]
 eval_sources=[p for p in ROOT.rglob('reap_eval.py') if 'reap128v2-original-scorer-20261026' in str(p)]
 if len(scorer_sources)!=1 or len(eval_sources)!=1 or digest(scorer_sources[0])!=ORIGINAL_SCORER_HASH or digest(eval_sources[0])!=ORIGINAL_EVAL_HASH:raise RuntimeError('original scorer source not exact')
 # Original scorer's final_answer pure logic: extract text after LAST </think>,
 # or empty when reasoning remains unclosed. This implementation is independent.
 matched=0;correct=0;terminated=0;timeouts=0;errors=0
 examples=[];wall=time.monotonic()
 for index,item in enumerate(data):
  row=by_id[item['id']]; text=row['text'];term=row['terminated']
  if not isinstance(text,str):raise RuntimeError('nonnumeric/nonstr HumanEval output')
  reply=text.rsplit('</think>',1)[1] if '</think>' in text else ('' if '<think>' in text else text)
  blocks=re.findall(r'```(?:python)?\n(.*?)```',reply,flags=re.S)
  code=blocks[-1] if blocks else reply
  if f"def {item['entry_point']}" not in code:code=item['header']+code
  program=code+'\n\n'+item['test']+f"\n\ncheck({item['entry_point']})\n"
  passed,reason=run(program)
  actual=bool(term and passed)
  matched+=int(actual==row['correct']);correct+=actual;terminated+=term;timeouts+=reason=='timeout';errors+=reason.startswith('exit') and reason!='exit0'
  if len(examples)<8 and actual!=row['correct']:examples.append({'id':item['id'],'reason':reason})
  if (index+1)%25==0: print('ISOLATED_PASS_PROGRESS',index+1,'/',len(data),flush=True)
 RESULT.update({'independent_full_rows':164,'matched_recorded_scores':matched,'correct':correct,'terminated':terminated,'timeouts':timeouts,'nonzero_exit_count':errors,'sample_mismatches':examples,'elapsed_seconds':round(time.monotonic()-wall,3),'original_row_sha256':ROW_HASH,'original_scorer_sha256':ORIGINAL_SCORER_HASH,'original_official_input_file_sha256':SOURCE_FILE_HASH,'session_id':SESSION})
 if matched!=164 or correct!=151:raise RuntimeError('independent candidate code replay disagrees with original complete HumanEval scores')
 RESULT['status']='PASS_INDEPENDENT_ISOLATED_HUMANEVAL_FULL_164_PROVISIONAL_LINUX'
except Exception as e:
 RESULT['error']=type(e).__name__+': '+str(e)[:400]
with (WORK/'reap-v2-human164-independent-lowpriv.json').open('x',encoding='utf8') as f:json.dump(RESULT,f,sort_keys=True,indent=2)
print(json.dumps(RESULT,sort_keys=True))