import hashlib, json, pathlib, subprocess, time, os
root=pathlib.Path('/kaggle/input')
expected={'q4':('reap128-v2-q4_k_m-20261025.gguf',11563915264,'2d64b786461b99a335fdbbdf7c4d3a6758b21409bce755751714ab7cd8f63bf7'),'gsm8k':('gsm8k.json',484685,'4dae0860ba5b9785343070a4919504f0aa4eccb01a96108b5b851d23e059a522'),'humaneval':('humaneval.json',274061,'5144d31cf337b25ce262b780ed103b128b1e9d289d3c0d15f4fe34933bb705d5')}
result={'schema':'reap-v2-private-kaggle-original-input-gguf-attest.v1','timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'items':{},'all_pass':False,'no_model_loaded':True}
try:
  for key,(name,size,want) in expected.items():
    paths=list(root.rglob(name))
    if len(paths)!=1: raise RuntimeError(key+' path missing or ambiguous: '+str(len(paths)))
    p=paths[0]
    if p.is_symlink() or not p.is_file() or p.stat().st_size!=size:raise RuntimeError(key+' size/symlink mismatch')
    h=hashlib.sha256()
    with p.open('rb') as f:
      for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    measured=h.hexdigest()
    result['items'][key]={'relative_path':str(p.relative_to(root)),'bytes':size,'sha256':measured,'passed':measured==want}
    if measured!=want:raise RuntimeError(key+' SHA256 does not match frozen exact source')
  gsm=json.loads(next(root.rglob('gsm8k.json')).read_text(encoding='utf-8'))
  he=json.loads(next(root.rglob('humaneval.json')).read_text(encoding='utf-8'))
  if len(gsm)!=1319 or len(he)!=164:raise RuntimeError('frozen original input populations differ')
  result['official_input_counts']={'gsm8k':len(gsm),'humaneval':len(he)}
  result['all_pass']=True
except Exception as e:
  result['error']=str(e)[:400]
with pathlib.Path('/kaggle/working/reap-v2-original-artifact-integrity.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,sort_keys=True)
print(json.dumps(result,sort_keys=True))