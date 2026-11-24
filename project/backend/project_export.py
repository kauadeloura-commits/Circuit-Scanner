"""Stage 17: current project package and an explicitly generated share link/QR."""
from pathlib import Path
import hashlib
import ipaddress
import json
import secrets
import sys
from urllib.parse import urlsplit
import zipfile
from backend.circuit_review import revision
from backend.digital_twin import VERSION as LAYOUT_VERSION
from backend.pcb_router import routing_is_current
EXPORT_VERSION="falstad-routed-2"

EXPORT_ARTIFACTS=('project_report.json','project.zip','export.json','share_qr.svg')
SHARE_FILE='share.json'
def digest(path):
    from backend.performance import file_hash
    result=file_hash(path)
    if result is None:raise FileNotFoundError(str(path))
    return result
def base_address(raw):
    if not isinstance(raw,str) or len(raw)>300:raise ValueError('Endereço de compartilhamento inválido.')
    parts=urlsplit(raw.strip())
    if parts.scheme not in {'http','https'} or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment or parts.path not in {'','/'}:
        raise ValueError('Informe somente o endereço do servidor, por exemplo http://192.168.1.10:8010.')
    try:parts.port
    except ValueError:raise ValueError('Porta do endereço inválida.')
    loopback=parts.hostname.lower()=='localhost'
    try:loopback=loopback or ipaddress.ip_address(parts.hostname).is_loopback
    except ValueError:pass
    return raw.strip().rstrip('/'),loopback
def qr_svg(url):
    vendor=str(Path(__file__).parent/'vendor')
    if vendor not in sys.path:sys.path.insert(0,vendor)
    import qrcode
    import qrcode.image.svg
    qr=qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M,box_size=8,border=4)
    qr.add_data(url);qr.make(fit=True)
    return qr.make_image(image_factory=qrcode.image.svg.SvgPathImage).to_string().decode('utf8')
def generate_export(folder,circuit,kicad,*,original_filename=None,base_url=None):
    folder=Path(folder);rev=revision(circuit)
    if kicad.get('source_revision')!=rev or kicad.get('layout_version')!=LAYOUT_VERSION or not routing_is_current(kicad):raise ValueError('Gere o esquemático e o roteamento para a versão atual antes de exportar.')
    existing=folder/SHARE_FILE
    share=json.loads(existing.read_text(encoding='utf8')) if existing.exists() else {}
    token=share.get('token') or secrets.token_urlsafe(24)
    address,loopback=base_address(base_url or 'http://127.0.0.1:8010')
    url=address+'/share/'+token
    allowed={'circuit.json','connectivity.json','components.json','detections.json','terminals.json','wires.json','processing.json',
             'processed.png','gray.png','binary.png','threshold.png','enhanced.png','perspective.png',
             'detections.png','ocr.png','terminals.png','wires.png','junctions.png','connectivity.png',
             'kicad.json','project_report.json','share_qr.svg','LEIA_O_PROJETO.txt'}
    if original_filename and Path(original_filename).name==original_filename:allowed.add(original_filename)
    for name in kicad.get('files',[]):
        path=Path(name)
        if path.is_absolute() or '..' in path.parts:raise ValueError('Arquivo de exportação inválido.')
        allowed.add(name)
    (folder/'share_qr.svg').write_text(qr_svg(url),encoding='utf8')
    report=dict(schema_version='stage17-project-report-1.0',source_revision=rev,
                component_count=len(circuit['components']),net_count=len(circuit['nets']),confirmed=True,
                schematic=True,pcb=kicad['pcb'],pcb_routed=kicad.get("routed",False),pcb_routing=kicad.get("routing"),
                simulator='Falstad/CircuitJS',
                warnings=list(kicad['warnings']),
                share_url=url,local_address=loopback)
    if loopback:report['warnings'].append('Este endereço abre somente neste computador. Para outro dispositivo, gere o link com o IP do servidor acessível na mesma rede.')
    (folder/'project_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    (folder/'LEIA_O_PROJETO.txt').write_text('Abra schematic.kicad_pro no KiCad 7 ou superior.\nO esquemático mantém as redes por rótulos.\nSe houver board.kicad_pcb, a placa contém posicionamento e redes, com roteamento inicial; confira as pendências no relatório.\nConfira pads, encapsulamentos e alimentação dos AOPs antes da montagem.\nAs simulações de transistores/MOSFETs/AOPs são aproximações; consulte os avisos.\nCompartilhamento: '+url+'\n',encoding='utf8')
    files=[];total=0
    for name in sorted(allowed):
        path=folder/name
        if path.is_file() and not path.is_symlink():
            total+=path.stat().st_size;files.append(name)
    if total>200*1024*1024:raise ValueError('Projeto excede 200 MB para exportação.')
    temporary=folder/('export_'+secrets.token_hex(8)+'.tmp')
    try:
        with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
            for name in files:
                encoded=Path(name).suffix.lower() in {'.png','.jpg','.jpeg','.webp'}
                archive.write(folder/name,name,compress_type=zipfile.ZIP_STORED if encoded else zipfile.ZIP_DEFLATED)
        temporary.replace(folder/'project.zip')
    finally:
        if temporary.exists():temporary.unlink()
    manifest=dict(source_revision=rev,layout_version=LAYOUT_VERSION,export_version=EXPORT_VERSION,share_url=url,token=token,local_address=loopback,files=files,
                  hashes={name:digest(folder/name) for name in files},zip_sha256=digest(folder/'project.zip'),report=report)
    (folder/'export.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    (folder/SHARE_FILE).write_text(json.dumps(dict(token=token,enabled=True),indent=2),encoding='utf8')
    return manifest
def export_is_current(folder,circuit):
    folder=Path(folder);path=folder/'export.json'
    if not path.is_file():return None
    try:
        manifest=json.loads(path.read_text(encoding='utf8'))
        if (manifest.get('source_revision')!=revision(circuit) or manifest.get('layout_version')!=LAYOUT_VERSION
            or manifest.get('export_version')!=EXPORT_VERSION or not (folder/'project.zip').is_file()):return None
        report=manifest.get('report') or {}
        if not routing_is_current(dict(pcb=report.get('pcb'),routing=report.get('pcb_routing'))):return None
        if manifest.get('zip_sha256')!=digest(folder/'project.zip'):return None
        for name,value in manifest.get('hashes',{}).items():
            path=Path(name)
            if path.is_absolute() or '..' in path.parts or not (folder/name).is_file() or digest(folder/name)!=value:return None
        return manifest
    except (OSError,ValueError,TypeError):return None
