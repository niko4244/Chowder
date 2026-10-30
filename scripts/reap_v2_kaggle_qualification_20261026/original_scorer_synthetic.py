import hashlib, importlib.util, json, pathlib, sys, types, zipfile
ROOT=pathlib.Path('/kaggle/input'); WORK=pathlib.Path('/kaggle/working')
SHA_EVAL='9736714f21834a45c23599a50fa92ef3bebc6cf70967aca7e2329d26b4d73ee1'
SHA_SCORE='d2e59aca5c0a621bac8d4979d522c9b6107f811610567d21c92a2b53960860eb'
R={'schema':'reap-v2-kaggle-original-scorer-compatibility.v1','status':'UNKNOWN','original_windows_equivalence':'NOT_PROVEN'}
try:
    src=list(ROOT.rglob('original-reap-frozen-evaluator-4da43ef.zip'))
    if len(src)==1:
        with zipfile.ZipFile(src[0]) as z:
            data={name:z.read(name) for name in ('chowder_batch/reap_eval.py','src/chowder/evaluators/scoring.py')}
    else:
        # Kaggle may auto-expand uploaded ZIP archives into private input mount.
        R['input_files_seen']=[str(x.relative_to(ROOT)) for x in ROOT.rglob('*') if x.is_file()][:35]
        evals=[x for x in ROOT.rglob('reap_eval.py') if 'reap128v2-original-scorer-20261026' in str(x)]
        scores=[x for x in ROOT.rglob('scoring.py') if 'reap128v2-original-scorer-20261026' in str(x)]
        if len(evals)!=1 or len(scores)!=1:raise ValueError('registered scorer source zip/extraction absent or ambiguous')
        data={'chowder_batch/reap_eval.py':evals[0].read_bytes(),'src/chowder/evaluators/scoring.py':scores[0].read_bytes()}
    if hashlib.sha256(data['chowder_batch/reap_eval.py']).hexdigest()!=SHA_EVAL or hashlib.sha256(data['src/chowder/evaluators/scoring.py']).hexdigest()!=SHA_SCORE:raise ValueError('original scorer/eval source bytes changed')
    temp=WORK/'locked_source';temp.mkdir(exist_ok=False)
    for name,buf in data.items():
        p=temp/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(buf)
    root=types.ModuleType('chowder');root.__path__=[];evals=types.ModuleType('chowder.evaluators');evals.__path__=[]
    sys.modules.update({'chowder':root,'chowder.evaluators':evals})
    def load(name,path):
        spec=importlib.util.spec_from_file_location(name,path)
        mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
        return mod
    score=load('chowder.evaluators.scoring',temp/'src/chowder/evaluators/scoring.py')
    r=load('reap_eval_locked',temp/'chowder_batch/reap_eval.py')
    checks={}
    for name,expected_n,want in [('gsm8k',1319,'913ca81f724cba16a07099608047e7522b00910c1c7fe13499c501ac9e4bad89'),('humaneval',164,'49fb688c6788b0531803dfd1cb85a69926ec5a7179e74cdfffc53ba52a92b420')]:
        f=list(ROOT.rglob(name+'.json'))
        if len(f)!=1:raise ValueError(name+' input path missing')
        data=json.loads(f[0].read_text(encoding='utf-8'))
        if len(data)!=expected_n or r._digest(data)!=want:raise ValueError(name+' frozen full rendered source identity failed')
        checks[name+'_source_pass']=True
    out={'text':'The final answer is 42.','finish_reason':'stop','completion_tokens':8,'seconds':0.1,'timings':None}
    gsm={'id':'synthetic','expected':'42'}
    positive=r._score('gsm8k',gsm,out)
    assert positive['correct'] is True
    rejected=r._score('gsm8k',gsm,{**out,'finish_reason':'length'})
    assert rejected['correct'] is False and rejected['terminated'] is False
    testspec={'id':'synthetic','entry_point':'square','header':'def square(x):\n    pass','test':'def check(candidate):\n    assert candidate(2)==4\n'}
    he=r._score('humaneval',testspec,{**out,'text':'```python\ndef square(x):\n    return x*x\n```'})
    assert he['correct'] is True
    checks.update({'frozen_original_scorer_gsm8k_positive':True,'nonterminating_rejected':True,'human_eval_python_isolated_pass':True})
    R.update({'status':'PASS_CPU_ORIGINAL_SCORER_SYNTHETIC','python_version':sys.version,'score_source_sha256':SHA_SCORE,'eval_source_sha256':SHA_EVAL,'checks':checks})
except Exception as ex:
    R.update({'status':'FAIL','error':repr(ex)[:400]})
(WORK/'reap-v2-original-scorer-crosshost.json').write_text(json.dumps(R,indent=2,sort_keys=True),encoding='utf-8')
print(json.dumps(R,sort_keys=True))