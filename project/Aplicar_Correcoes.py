"""Apply only the reviewed source files; never modifies projects or models."""
from pathlib import Path
import argparse,json,hashlib,datetime,os,time,uuid,shutil
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(path,data):
 temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
 try:
  temp.write_bytes(data)
  for delay in (0,.03,.06,.12,.24,.5):
   if delay:time.sleep(delay)
   try:os.replace(temp,path);return
   except PermissionError:
    if delay==.5:raise
 finally:temp.unlink(missing_ok=True)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--project',required=True);parser.add_argument('--backup-root',required=True);parser.add_argument('--apply',action='store_true');args=parser.parse_args()
 project=Path(args.project).resolve();backup_root=Path(args.backup_root).resolve();payload=Path(__file__).resolve().parent/'atualizacao'
 assert (project/'backend/main.py').is_file(),'Pasta do programa não encontrada'
 plan=json.loads((payload/'manifest.json').read_text(encoding='utf8'));pending=[]
 for item in plan['files']:
  relative=Path(item['path']);target=(project/relative).resolve();source=(payload/relative).resolve()
  assert relative.parts[0]=='backend' and len(relative.parts)==2 and relative.suffix=='.py'
  assert target.is_relative_to(project) and source.is_relative_to(payload.resolve())
  assert sha(source)==item['after_sha256'],'Pacote de atualização alterado'
  current=sha(target)
  if current==item['after_sha256']:continue
  assert current==item['before_sha256'],'O arquivo mudou desde a avaliação; atualização interrompida: '+str(target)
  pending.append((item,target,source))
 result=dict(project=str(project),version=plan['version'],pending_files=[i['path'] for i,_,_ in pending],applied=False)
 if args.apply and pending:
  backup=backup_root/('backup_5_48_0_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));backup.mkdir(parents=True,exist_ok=False)
  for item,target,source in pending:
   saved=backup/item['path'];saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(target,saved)
   assert sha(saved)==item['before_sha256']
  changed=[]
  try:
   for item,target,source in pending:
    assert sha(target)==item['before_sha256'],'Arquivo alterado durante a instalação: '+str(target)
    atomic(target,source.read_bytes());changed.append((item,target))
    assert sha(target)==item['after_sha256']
  except Exception:
   for item,target in reversed(changed):
    if sha(target)==item['after_sha256']:atomic(target,(backup/item['path']).read_bytes())
   raise
  result.update(applied=True,backup=str(backup),changed_files=[i['path'] for i,_ in changed])
  (backup/'installation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
 elif args.apply:result.update(applied=True,already_current=True)
 print(json.dumps(result,ensure_ascii=True,indent=2))
if __name__=='__main__':main()
