from pathlib import Path
import json, hashlib, shutil, random, collections, sys
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parent
SRC=Path('C:/tc2/Circuit_Scanner_Etapa5_Corrigido/project')
SNAP=ROOT/'runtime'
CLASSES=['resistor','capacitor','capacitor_polarized','diode','led','inductor','source_dc','battery','gnd','vcc','transistor_npn','transistor_pnp','mosfet_n','mosfet_p','opamp']
PT=['Resistor','Capacitor','Capacitor polarizado','Diodo','LED','Indutor','Fonte DC','Bateria','GND','VCC','Transistor NPN','Transistor PNP','MOSFET N','MOSFET P','Amplificador operacional']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,obj):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')
def inventory():
    rows=[]
    for m in sorted((SRC/'real_data/meta').glob('*.json')):
        meta=json.loads(m.read_text(encoding='utf-8-sig'))
        image=next((SRC/'real_data/images').glob(m.stem+'.*'),None)
        if not image:continue
        a=SRC/'real_data/annotations'/m.name
        boxes=json.loads(a.read_text(encoding='utf-8-sig')).get('boxes',[]) if a.exists() else []
        rows.append(dict(id=m.stem,meta=meta,path=str(image),sha256=sha(image),boxes=boxes))
    dump(ROOT/'inventory.json',rows)
    summary={c:dict(instances=sum(sum(int(b['class_id'])==i for b in r['boxes']) for r in rows),images=sum(any(int(b['class_id'])==i for b in r['boxes']) for r in rows)) for i,c in enumerate(CLASSES)}
    dump(ROOT/'inventory_summary.json',summary)
    print(json.dumps(summary,indent=2))
    # contact sheets with every source, labels drawn before predictions
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
    for page in range((len(rows)+19)//20):
        canvas=Image.new('RGB',(1400,1160),'white');d=ImageDraw.Draw(canvas)
        for j,r in enumerate(rows[page*20:(page+1)*20]):
            im=Image.open(r['path']).convert('RGB'); im.thumbnail((270,246))
            x=(j%5)*280;y=(j//5)*290
            canvas.paste(im,(x,y+27))
            d.text((x+3,y+3),f"{r['id']}  {len(r['boxes'])}  {r['meta'].get('source_type','?')}",font=font,fill='black')
        out=ROOT/'sheets'/f'sources_{page+1:02d}.jpg';out.parent.mkdir(exist_ok=True);canvas.save(out,quality=91)
def freeze():
    SNAP.mkdir(exist_ok=True)
    names=['backend','models','terminal_keypoints','terminal_training_all_components','ocr_samples','frontend']
    manifest=[]
    for folder in names:
        src=SRC/folder
        if not src.exists():continue
        for p in src.rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts or p.suffix=='.pyc':continue
            rel=p.relative_to(SRC);dest=SNAP/rel;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,dest);manifest.append(dict(path=str(rel).replace('\\','/'),sha256=sha(p),bytes=p.stat().st_size))
    for p in (SRC/'terminal_keypoints/raw/meta').glob('*.json'):
        m=json.loads(p.read_text(encoding='utf-8-sig'));project=m.get('project_id');source=m.get('source_image')
        if project and source:
            raw=SRC/'projects'/project/source;dest=SNAP/'projects'/project/source
            if raw.exists() and not dest.exists():
                dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(raw,dest)
                manifest.append(dict(path=str(dest.relative_to(SNAP)).replace('\\','/'),sha256=sha(raw),bytes=raw.stat().st_size))
    # Source definitions and all support files are preserved; existing reviews/cache/DB are not imported.
    for name in ['requirements.txt','classes.json']:
        p=SRC/name
        if p.exists():shutil.copy2(p,SNAP/name);manifest.append(dict(path=name,sha256=sha(p),bytes=p.stat().st_size))
    dump(ROOT/'runtime_manifest.json',manifest)
    print(f'FROZEN {len(manifest)} files; {sum(x["bytes"] for x in manifest)/1e6:.1f} MB')
if __name__=='__main__':
    inventory()
    if '--freeze' in sys.argv:freeze()
