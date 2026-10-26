import hashlib, json, os, pathlib, shutil, subprocess, time, traceback
root=pathlib.Path('/kaggle/working')
log=root/'reap128-v2-cuda-build.log'
result={'schema':'reap128-v2-cuda-toolchain.v1','expected_git_prefix':'7609846557c50f9d984719a9e1e8c5f3d02f807b','success':False,'no_model_loaded':True}
def run(cmd,timeout):
    with log.open('a',encoding='utf-8') as f:
        f.write('\nCMD: '+' '.join(cmd)+'\n');f.flush()
        p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=timeout,cwd=str(root))
        f.write('EXIT: '+str(p.returncode)+'\n');f.flush()
    if p.returncode: raise RuntimeError(cmd[0]+' failed with '+str(p.returncode))
try:
    run(['git','clone','--depth','1','--filter=blob:none','--no-checkout','https://github.com/ggml-org/llama.cpp.git','llama.cpp'],300)
    run(['git','-C',str(root/'llama.cpp'),'fetch','--depth','1','origin','7609846557c50f9d984719a9e1e8c5f3d02f807b'],180)
    run(['git','-C',str(root/'llama.cpp'),'checkout','--detach','FETCH_HEAD'],120)
    revision=subprocess.check_output(['git','-C',str(root/'llama.cpp'),'rev-parse','HEAD'],text=True).strip()
    if not revision.startswith('7609846557c50f9d984719a9e1e8c5f3d02f807b'): raise RuntimeError('original Windows source commit mismatch')
    result['source_git_commit']=revision
    result['cuda_vmm_disabled_due_to_missing_driver_link_target']=True
    run(['cmake','-S',str(root/'llama.cpp'),'-B',str(root/'build'),'-DGGML_CUDA=ON','-DGGML_CUDA_NO_VMM=ON','-DLLAMA_CURL=OFF','-DCMAKE_BUILD_TYPE=Release'],300)
    run(['cmake','--build',str(root/'build'),'--config','Release','--target','llama-server','-j','4'],2400)
    src=root/'build'/'bin'
    dst=root/'reap-pinned-linux-bin'
    shutil.copytree(src,dst)
    binaries=list(dst.glob('llama-server*'))
    if not binaries: raise RuntimeError('no server executable found')
    server=binaries[0]
    result['server_sha256']=hashlib.sha256(server.read_bytes()).hexdigest()
    result['server_bytes']=server.stat().st_size
    result['artifact_bin_dir']='reap-pinned-linux-bin'
    result['success']=True
except Exception as ex:
    result['error']=type(ex).__name__+': '+str(ex)[:250]
finally:
    result['log_tail']=log.read_text(encoding='utf-8',errors='replace')[-2500:] if log.exists() else ''
    shutil.rmtree(root/'llama.cpp',ignore_errors=True)
    shutil.rmtree(root/'build',ignore_errors=True)
    with (root/'reap128-v2-cuda-build-result.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,sort_keys=True))