from datetime import datetime
from pathlib import Path
import copy
from uuid import uuid4
import json
import shutil
import os
from contextlib import asynccontextmanager
from backend.cpu_inference import start_model_warmup

import cv2

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse

from backend.database import (
    atualizar_deteccoes,
    atualizar_ocr,
    atualizar_processamento,
    atualizar_terminais,
    atualizar_terminais_pos_fios,
    atualizar_fios,
    buscar_projeto,
    excluir_projeto_por_id_interno,
    finalizar_projeto,
    inicializar_banco,
    inserir_projeto_temporario,
    listar_projetos,
    listar_resumos_projetos,
    registrar_projeto_recuperado,
    limpar_nomes_recuperados,
    contar_projetos,
    DATABASE_PATH,
)
from backend.image_processing import processar_imagem
from backend.resistor_geometry import symbol_analysis_session
from backend.dataset_generator import (
    DATASET_DIR,
    dataset_info,
    generate_dataset,
    generate_previews,
)
from backend.recognition import (
    detectar_componentes,
    model_status,
    salvar_imagem_deteccoes,
)
from backend.detection_boxes import validate_no_nested_boxes
from backend.performance import (cached_project_stage,time_project_operation,DEFER_INITIAL_GEOMETRY,geometry_key,cache_valid,cache_store,GEOMETRY_OUTPUTS,read_json as read_performance_json)
from backend.terminal_detection import executar_deteccao_terminais, salvar_imagem_terminais
from backend.wire_detection import detectar_fios, salvar_imagem_fios, salvar_mascara
from backend.junction_detection import (
    analisar_clique_juncao,
    aplicar_correcoes_juncoes,
    resolver_juncoes,
    salvar_imagem_juncoes,
)
from backend.circuit_connectivity import construir_conectividade, salvar_imagem_redes, source_signature
from backend.circuit_document import gerar_circuito, validar_circuito
from backend.circuit_page import circuit_page
from backend.circuit_review import apply_review, confirm_review, revision, review_issues, review_component_models
from backend.circuit_review_page import review_page
from backend.kicad_export import build_kicad, ARTIFACTS as KICAD_ARTIFACTS
from backend.project_export import EXPORT_ARTIFACTS
from backend.component_models import catalog_summary, resolve_parameters, preserve_manual_models
from backend.model_ocr_page import model_ocr_page
from tempfile import TemporaryDirectory
from threading import RLock

_circuit_review_lock=RLock()
from backend.transistor_wire_refinement import (
    refine_transistor_terminals_from_wires,
    save_transistor_wire_refinement_preview,
)

from backend.ocr import (
    aplicar_correcoes_manuais,
    executar_ocr_componentes,
    ocr_status,
    salvar_amostras_ocr,
    salvar_imagem_componentes,
)

from backend.terminal_learning import (
    KEYPOINT_ORDER,
    TRANSISTOR_CLASSES,
    save_transistor_training_sample,
    terminal_learning_status,
    terminal_model_runtime_status,
    diagnose_transistor_keypoints,
    repair_training_samples_from_metadata,
    train_terminal_keypoint_model,
    lookup_known_training_ground_truth,
    predict_transistor_visual_prior,
)
from backend.generic_terminal_learning import (
    generic_terminal_learning_status,
    save_generic_terminal_training_sample,
)
from backend.all_component_terminal_learning import (
    model_runtime_status as all_component_terminal_model_status,
)

from backend.real_dataset import (
    REAL_DATA_DIR,
    build_mixed_dataset,
    get_real_image,
    list_real_images,
    load_classes as load_real_classes,
    real_dataset_info,
    save_annotations,
    upload_real_image,
)
from backend.component_learning import (
    component_training_status,
    rollback_component_model,
    request_component_training_stop,
    set_foundation_from_active_model,
    resume_component_training,
    start_component_training,
)
from backend.component_trainer_page import component_trainer_html
from backend.component_diagnostics import (
    audit_real_annotations,
    component_diagnostic_status,
    start_component_diagnostic,
)

MANUAL_PIN_SCHEMAS = {
    "vcc": [("VCC", "power")],
    "gnd": [("GND", "ground")],
    "resistor": [("1", "terminal_1"), ("2", "terminal_2")],
    "capacitor": [("1", "terminal_1"), ("2", "terminal_2")],
    "capacitor_polarized": [("+", "positive"), ("-", "negative")],
    "diode": [("A", "anode"), ("K", "cathode")],
    "led": [("A", "anode"), ("K", "cathode")],
    "inductor": [("1", "terminal_1"), ("2", "terminal_2")],
    "source_dc": [("+", "positive"), ("-", "negative")],
    "battery": [("+", "positive"), ("-", "negative")],
    "transistor_npn": [("B", "base"), ("C", "collector"), ("E", "emitter")],
    "transistor_pnp": [("B", "base"), ("C", "collector"), ("E", "emitter")],
    "mosfet_n": [("G", "gate"), ("D", "drain"), ("S", "source")],
    "mosfet_p": [("G", "gate"), ("D", "drain"), ("S", "source")],
    "opamp": [("+", "non_inverting_input"), ("-", "inverting_input"), ("OUT", "output")],
}

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECTS_DIR = BASE_DIR / "projects"
PROJECTS_DIR.mkdir(exist_ok=True)

EXTENSOES_PERMITIDAS = {".jpg", ".jpeg", ".png"}

inicializar_banco()
LEGACY_RECOVERED_NAMES_CLEANED = limpar_nomes_recuperados()
generate_previews()

@asynccontextmanager
async def scanner_lifespan(app):
    start_model_warmup()
    yield

app = FastAPI(title="Circuit Scanner - Etapas 1 a 18", lifespan=scanner_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/projects-files",
    StaticFiles(directory=PROJECTS_DIR),
    name="projects-files",
)

app.mount(
    "/dataset-files",
    StaticFiles(directory=DATASET_DIR),
    name="dataset-files",
)

app.mount(
    "/real-data-files",
    StaticFiles(directory=REAL_DATA_DIR),
    name="real-data-files",
)


def _carregar_json_texto(texto):
    if not texto:
        return None

    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        return None


def _texto_json_se_existir(caminho: Path):
    if not caminho.exists():
        return None
    try:
        # Valida o JSON, mas preserva o texto completo no SQLite.
        texto = caminho.read_text(encoding="utf-8")
        json.loads(texto)
        return texto
    except Exception:
        return None


def _recuperar_indice_projetos_do_disco() -> dict:
    """Reconstrói o índice SQLite usando as pastas projects/ existentes.

    O banco é apenas o índice/metadado; os arquivos do projeto continuam sendo
    a fonte de recuperação. Isso evita que uma atualização do backend faça a
    lista do front-end parecer vazia caso circuit_scanner.db seja substituído
    ou removido acidentalmente.
    """
    antes = contar_projetos()
    recuperados = 0
    ignorados = 0

    for project_dir in sorted(PROJECTS_DIR.glob("PRJ_*")):
        if not project_dir.is_dir():
            continue
        project_id = project_dir.name
        if buscar_projeto(project_id) is not None:
            ignorados += 1
            continue

        originals = []
        for ext in (".png", ".jpg", ".jpeg"):
            originals.extend(project_dir.glob(f"original{ext}"))
        if not originals:
            # Sem original não há como reabrir o projeto de forma segura.
            continue
        original_path = sorted(originals)[0]

        processing_info = _texto_json_se_existir(project_dir / "processing.json")
        detections_info = _texto_json_se_existir(project_dir / "detections.json")
        ocr_info = _texto_json_se_existir(project_dir / "components.json")
        terminals_info = _texto_json_se_existir(project_dir / "terminals.json")
        wires_info = _texto_json_se_existir(project_dir / "wires.json")

        processed_filename = "processed.png" if (project_dir / "processed.png").exists() else None
        detection_filename = "detections.png" if (project_dir / "detections.png").exists() else None
        ocr_filename = "ocr.png" if (project_dir / "ocr.png").exists() else None
        terminals_filename = "terminals.png" if (project_dir / "terminals.png").exists() else None
        wires_filename = "wires.png" if (project_dir / "wires.png").exists() else None

        if wires_info and wires_filename:
            status = "fios_detectados"
        elif terminals_info and terminals_filename:
            status = "terminais_detectados"
        elif ocr_info:
            status = "ocr_concluido"
        elif detections_info:
            status = "componentes_detectados"
        elif processed_filename:
            status = "processado"
        else:
            status = "criado"

        created_at = datetime.fromtimestamp(
            project_dir.stat().st_mtime
        ).astimezone().isoformat(timespec="seconds")

        registrar_projeto_recuperado(
            project_id=project_id,
            name=project_id,
            created_at=created_at,
            status=status,
            path=f"projects/{project_id}",
            original_filename=original_path.name,
            stored_filename=original_path.name,
            processed_filename=processed_filename,
            processing_info=processing_info,
            detection_filename=detection_filename,
            detections_info=detections_info,
            ocr_filename=ocr_filename,
            ocr_info=ocr_info,
            terminals_filename=terminals_filename,
            terminals_info=terminals_info,
            wires_filename=wires_filename,
            wires_info=wires_info,
        )
        if buscar_projeto(project_id) is not None:
            recuperados += 1

    nomes_limpos = limpar_nomes_recuperados()
    return {
        "database_path": str(DATABASE_PATH),
        "projects_before": antes,
        "projects_after": contar_projetos(),
        "recovered": recuperados,
        "already_indexed": ignorados,
        "legacy_recovered_names_cleaned": nomes_limpos,
    }


STORAGE_RECOVERY = _recuperar_indice_projetos_do_disco()


def _components_info_from_detections(detections_info: dict) -> dict:
    """Converte detecções visuais em componentes provisórios para a Etapa 7.

    A posição dos terminais depende da classe + bbox, não do OCR. Isso permite
    gerar terminals.png logo após o YOLO e, principalmente, regenerá-lo assim
    que o usuário adiciona/remove/corrige uma detecção manualmente.
    """
    detections = (detections_info or {}).get("detections") or []
    components = []
    for index, det in enumerate(detections, start=1):
        tipo = str(det.get("class") or det.get("type") or "").strip()
        bbox = det.get("bbox")
        if not tipo or not isinstance(bbox, dict):
            continue
        detection_id = str(det.get("detection_id") or det.get("component_id") or f"DET_{index:03d}")
        components.append(
            {
                "component_id": detection_id,
                "detection_id": detection_id,
                "type": tipo,
                "bbox": bbox,
                "visual_confidence": det.get("confidence"),
                "reference": {
                    "text": detection_id,
                    "source": "provisional_detection_id",
                    "confidence": None,
                },
                "value": {
                    "text": "",
                    "normalized": None,
                    "unit": None,
                    "source": "pending_ocr",
                    "confidence": None,
                },
                "review_required": bool(det.get("review_required", False)),
                "review_reasons": list(det.get("review_reasons") or []),
                "provisional_from_detection": True,
            }
        )

    return {
        "schema_version": "stage7-from-detections-1.0",
        "source": "detections",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "components": components,
    }


OLD_RESULT_ARTIFACTS=('spice.json','simulation.cir','simulation.json','simulation.csv','simulation.raw','simulation.log')

def _remover_artefatos_desatualizados(project_dir: Path, nomes: tuple[str, ...]) -> None:
    """Remove arquivos antigos para nunca exibir um resultado obsoleto."""
    if set(nomes) & {'terminals.json','wires.json','junctions.json','detections.json','components.json'}:
        nomes=tuple(dict.fromkeys(nomes+('connectivity.json','connectivity.png','circuit.json')))
    if 'circuit.json' in nomes:
        nomes=tuple(dict.fromkeys(nomes+OLD_RESULT_ARTIFACTS+KICAD_ARTIFACTS+EXPORT_ARTIFACTS))
    elif set(nomes)&set(OLD_RESULT_ARTIFACTS):
        nomes=tuple(dict.fromkeys(nomes+EXPORT_ARTIFACTS))
    for nome in nomes:
        caminho = project_dir / nome
        if caminho.exists():
            if nome=='circuit.json':
                previous=_carregar_json_texto(caminho.read_text(encoding='utf8'))
                if previous and previous.get('review'):
                    history=project_dir/'circuit_review_history';history.mkdir(exist_ok=True)
                    shutil.copy2(caminho,history/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:8]+'.json'))
            caminho.unlink()


def _continuar_automaticamente_para_linhas(project_id: str, trigger: str) -> dict:
    """Executa a Etapa 8 sem exigir uso manual do Swagger (/docs).

    A falha da Etapa 8 nao invalida reconhecimento/OCR/terminais ja salvos; ela
    retorna um aviso explicito para a tela de fluxo e pode ser repetida por um
    unico botao.
    """
    projeto_atual = buscar_projeto(project_id)
    if projeto_atual is None:
        return {"executed": False, "trigger": trigger, "warning": "Projeto nao encontrado apos salvar terminais."}
    try:
        folder=PROJECTS_DIR/project_id
        wires=_json_projeto(folder,'wires.json')
        if trigger=='after_manual_terminal_correction' and wires and wires.get('stage9'):
            image=cv2.imread(str(folder/projeto_atual['stored_filename']))
            terminals=_json_projeto(folder,'terminals.json')
            previous=wires['stage9']
            junctions=resolver_juncoes(image,wires,terminals_info=terminals)
            manual={n['node_id']:n for n in previous.get('nodes',[]) if n.get('manual_override')}
            junctions['nodes']=[copy.deepcopy(manual.pop(n['node_id'],n)) for n in junctions.get('nodes',[])]+list(manual.values())
            # Retain explicit junction decisions and only reattach the edited
            # pin names. Wire pixels, segments and crossing groups are reused.
            from backend.junction_detection import _summary
            junctions.update(_summary(junctions['nodes']))
            wires['stage9']=junctions
            (folder/'junctions.json').write_text(json.dumps(junctions,ensure_ascii=False,indent=2),encoding='utf8')
            salvar_imagem_juncoes(image,junctions,folder/'junctions.png')
            _gerar_resultados_eletricos(project_id,projeto_atual,wires,terminals,image)
            _salvar_estado_eletrico(project_id,projeto_atual,wires)
            info=_latest_components_for_project(buscar_projeto(project_id))
            cache_store(folder,'geometry',geometry_key(BASE_DIR,folder,buscar_projeto(project_id),info),GEOMETRY_OUTPUTS)
            return dict(executed=True,trigger=trigger,segment_count=int(wires.get('segment_count') or 0),wire_processing_reused=True)
        wires = _executar_etapa8_fios(project_id, projeto_atual)
        return {
            "executed": True,
            "trigger": trigger,
            "segment_count": int(wires.get("segment_count") or 0),
            "transistor_terminal_refinement": wires.get("transistor_terminal_refinement"),
        }
    except Exception as exc:
        return {"executed": False, "trigger": trigger, "warning": str(exc)}


@time_project_operation('geometry')
@symbol_analysis_session
def _gerar_terminais_automaticamente(
    project_id: str,
    projeto: dict,
    components_info: dict,
    trigger: str = "manual",
):
    """Components -> independent wires -> bounded terminals -> junctions."""
    if not components_info or not components_info.get("components"):
        raise ValueError("Nao ha componentes para detectar fios e terminais.")
    project_dir = PROJECTS_DIR / project_id
    input_key=geometry_key(BASE_DIR,project_dir,projeto,components_info)
    from backend.performance import shared_restore,shared_store
    previous_terms=read_performance_json(project_dir/'terminals.json')
    shareable=(os.environ.get('SCANNER_SHARED_ANALYSIS_CACHE','1')!='0' and not(project_dir/'terminal_corrections.json').exists()
               and not any(c.get('terminal_detection',{}).get('status')=='manual_ground_truth' for c in previous_terms.get('components',[])))
    if not previous_terms and not cache_valid(project_dir,'geometry',input_key,GEOMETRY_OUTPUTS) and shareable and shared_restore(BASE_DIR,'geometry',input_key,project_dir,GEOMETRY_OUTPUTS,project_id=project_id,stored_filename=projeto['stored_filename']):
        saved=read_performance_json(project_dir/'terminals.json');wires=read_performance_json(project_dir/'wires.json')
        wires.pop('stage10',None);wires.pop('stage11',None)
        (project_dir/'wires.json').write_text(json.dumps(wires,ensure_ascii=False,indent=2),encoding='utf8')
        atualizar_fios(project_id=project_id,status='juncoes_resolvidas' if not wires.get('stage9',{}).get('review_count') else 'juncoes_aguardando_revisao',wires_filename='wires.png',wires_info=json.dumps(wires,ensure_ascii=False))
        atualizar_terminais(project_id=project_id,status='terminais_detectados_apos_fios',terminals_filename='terminals.png',terminals_info=json.dumps(saved,ensure_ascii=False))
        cache_store(project_dir,'geometry',input_key,GEOMETRY_OUTPUTS)
    if cache_valid(project_dir,'geometry',input_key,GEOMETRY_OUTPUTS):
        saved=read_performance_json(project_dir/'terminals.json')
        originals={c.get('detection_id'):c for c in components_info['components']}
        fields=('reference','value','model','electrical_parameters','parameter_policy','parameter_pending')
        changed=any(any(c.get(k)!=originals.get(c.get('detection_id'),{}).get(k) for k in fields)
                    for c in saved.get('components',[]))
        if changed:
            _sincronizar_modelos_terminais(project_id,projeto,components_info)
            saved=read_performance_json(project_dir/'terminals.json')
            salvar_imagem_terminais(project_dir/str(projeto['stored_filename']),saved,project_dir/'terminals.png')
            cache_store(project_dir,'geometry',geometry_key(BASE_DIR,project_dir,projeto,components_info),GEOMETRY_OUTPUTS)
        saved['processing_reused']=True
        return saved
    _remover_artefatos_desatualizados(project_dir, (
        "terminals.png", "terminals.json", "wires.png", "wires.json", "wires_mask.png",
        "wires_skeleton.png", "wires_erased_mask.png", "transistor_wire_refinement.png",
        "junctions.json", "junctions.png"))
    wires = _executar_etapa8_fios(project_id, projeto, components_override=components_info,
                                  generation_trigger=trigger)
    resultado = json.loads((project_dir / "terminals.json").read_text(encoding="utf-8"))
    resultado["automatic_stage8"] = {
        "executed": True, "trigger": f"before_terminals_{trigger}",
        "segment_count": int(wires.get("segment_count") or 0),
        "transistor_terminal_refinement": wires.get("transistor_terminal_refinement"),
        "terminal_detection_after_wires": True,
    }
    (project_dir / "terminals.json").write_text(json.dumps(resultado,ensure_ascii=False,indent=2),encoding="utf-8")
    atualizar_terminais(project_id=project_id,status="terminais_detectados_apos_fios",
        terminals_filename="terminals.png",terminals_info=json.dumps(resultado,ensure_ascii=False))
    cache_store(project_dir,'geometry',geometry_key(BASE_DIR,project_dir,projeto,components_info),GEOMETRY_OUTPUTS)
    if shareable:shared_store(BASE_DIR,'geometry',input_key,project_dir,GEOMETRY_OUTPUTS,stored_filename=projeto['stored_filename'])
    return resultado



def _latest_components_for_project(projeto: dict) -> dict:
    ocr_info = _carregar_json_texto(projeto.get("ocr_info"))
    if ocr_info and isinstance(ocr_info.get("components"), list):
        return ocr_info
    detections_info = _carregar_json_texto(projeto.get("detections_info"))
    return _components_info_from_detections(detections_info or {})


def _terminal_result_for_manual_edit(project_id: str, projeto: dict) -> dict:
    current = _carregar_json_texto(projeto.get("terminals_info"))
    if current and isinstance(current.get("components"), list):
        return copy.deepcopy(current)
    source = _latest_components_for_project(projeto)
    try:
        project_dir = PROJECTS_DIR / project_id
        source_path = project_dir / projeto["stored_filename"]
        return _gerar_terminais_automaticamente(project_id,projeto,source,trigger='manual_editor_initialization')
    except Exception:
        components = copy.deepcopy(source.get("components") or [])
        for comp in components:
            comp.setdefault("terminals", [])
            comp.setdefault("terminal_detection", {
                "status": "pending_manual_training",
                "review_required": True,
                "review_reasons": ["sem_geometria_automatica_confiavel"],
            })
        return {
            "schema_version": "stage7-manual-training-1.0",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "count": len(components),
            "terminal_count": sum(len(c.get("terminals") or []) for c in components),
            "review_count": sum(1 for c in components if (c.get("terminal_detection") or {}).get("review_required")),
            "components": components,
        }


def _find_component(components: list[dict], component_id: str) -> dict | None:
    target = str(component_id)
    for comp in components:
        ids = {
            str(comp.get("component_id") or ""),
            str(comp.get("detection_id") or ""),
            str((comp.get("reference") or {}).get("text") or ""),
        }
        if target in ids:
            return comp
    return None


def _bbox_iou(a: dict | None, b: dict | None) -> float:
    if not isinstance(a, dict) or not isinstance(b, dict):
        return 0.0
    try:
        ax1, ay1, ax2, ay2 = map(float, (a["x1"], a["y1"], a["x2"], a["y2"]))
        bx1, by1, bx2, by2 = map(float, (b["x1"], b["y1"], b["x2"], b["y2"]))
    except Exception:
        return 0.0
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
    inter = iw * ih
    area_a = max(0.0, ax2-ax1) * max(0.0, ay2-ay1)
    area_b = max(0.0, bx2-bx1) * max(0.0, by2-by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def _terminal_corrections_path(project_id: str) -> Path:
    return PROJECTS_DIR / project_id / "terminal_corrections.json"


def _load_terminal_corrections(project_id: str) -> dict:
    path = _terminal_corrections_path(project_id)
    if not path.exists():
        return {"components": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {"components": []}
    except Exception:
        return {"components": []}


def _clean_pin_schema(items) -> list[tuple[str, str]]:
    clean: list[tuple[str, str]] = []
    seen = set()
    for item in items or []:
        if isinstance(item, dict):
            pin = str(item.get("pin") or "").strip()
            role = str(item.get("role") or item.get("electrical_role") or pin).strip()
        elif isinstance(item, (list, tuple)) and item:
            pin = str(item[0]).strip()
            role = str(item[1] if len(item) > 1 else pin).strip()
        else:
            pin = str(item or "").strip()
            role = pin
        if not pin or len(pin) > 32 or pin in seen:
            continue
        seen.add(pin)
        clean.append((pin, role[:64] or pin))
    return clean


def _schema_for_component(component: dict, requested_schema=None, points: dict | None = None) -> list[tuple[str, str]]:
    """Retorna pinos editaveis para classes atuais e futuras."""
    component_type = str(component.get("type") or "")
    fixed = MANUAL_PIN_SCHEMAS.get(component_type)
    if fixed:
        return list(fixed)
    requested = _clean_pin_schema(requested_schema)
    if requested:
        return requested
    existing = _clean_pin_schema([
        {"pin": t.get("pin"), "role": t.get("electrical_role")}
        for t in (component.get("terminals") or [])
    ])
    if existing:
        return existing
    if isinstance(points, dict):
        return _clean_pin_schema([{"pin": pin, "role": pin} for pin in points])
    return []


def _save_terminal_correction_record(
    project_id: str,
    component: dict,
    points: dict,
    pin_schema: list[tuple[str, str]],
) -> None:
    data = _load_terminal_corrections(project_id)
    items = data.setdefault("components", [])
    component_id = str(component.get("component_id") or component.get("detection_id") or "")
    record = {
        "component_id": component_id,
        "type": component.get("type"),
        "bbox": component.get("bbox"),
        "points": points,
        "pin_schema": [{"pin": pin, "role": role} for pin, role in pin_schema],
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    replaced = False
    for i, item in enumerate(items):
        if str(item.get("component_id")) == component_id:
            items[i] = record
            replaced = True
            break
    if not replaced:
        items.append(record)
    data["schema_version"] = "terminal-corrections-2.0"
    _terminal_corrections_path(project_id).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _apply_saved_terminal_corrections(project_id: str, result: dict) -> int:
    data = _load_terminal_corrections(project_id)
    components = result.get("components") or []
    applied = 0
    used_component_indexes: set[int] = set()
    for record in data.get("components") or []:
        record_id = str(record.get("component_id") or "")
        comp = _find_component(components, record_id)
        comp_index = components.index(comp) if comp in components else None
        # O ID DET_xxx pode mudar após nova detecção/correção. Nesse caso,
        # reaplica o ground truth pelo mesmo tipo + sobreposição forte da bbox.
        if comp is None:
            rtype = str(record.get("type") or "")
            rbbox = record.get("bbox")
            candidates = []
            for idx, candidate in enumerate(components):
                if idx in used_component_indexes or str(candidate.get("type") or "") != rtype:
                    continue
                score = _bbox_iou(rbbox, candidate.get("bbox"))
                if score >= 0.55:
                    candidates.append((score, idx, candidate))
            if candidates:
                _score, comp_index, comp = max(candidates, key=lambda x: x[0])
        if comp is None:
            continue
        component_type = str(comp.get("type") or "")
        schema = _schema_for_component(
            comp,
            requested_schema=record.get("pin_schema"),
            points=record.get("points"),
        )
        if not schema:
            continue
        if comp_index is not None:
            used_component_indexes.add(comp_index)
        points = record.get("points") or {}
        pin_names = [pin for pin, _role in schema]
        if not all(pin in points for pin in pin_names):
            continue
        terminals = []
        for idx, (pin, role) in enumerate(schema, start=1):
            item = points[pin]
            terminals.append({
                "pin_id": f"{comp.get('component_id', 'COMP')}.{pin}",
                "pin": pin,
                "electrical_role": role,
                "terminal_side": "manual_ground_truth",
                "index": idx,
                "position": {"x": round(float(item["x"]), 2), "y": round(float(item["y"]), 2)},
                "search_radius_px": 10,
                "source": "manual_ground_truth_persisted",
                "identity_uncertain": False,
                "confidence": 1.0,
            })
        comp["terminals"] = terminals
        comp['review_required']=False
        comp['review_reasons']=[]
        comp["terminal_detection"] = {
            "status": "manual_ground_truth",
            "pin_count": len(terminals),
            "review_required": False,
            "review_reasons": [],
            "method": "persisted_user_terminal_correction",
        }
        if component_type in TRANSISTOR_CLASSES:
            comp["transistor_geometry"] = {
                "success": True,
                "reliable": True,
                "source": "manual_ground_truth_persisted",
                "manual_correction_record_id": record_id,
            }
        applied += 1
    if applied:
        result["terminal_count"] = sum(len(c.get("terminals") or []) for c in components)
        result["review_count"] = sum(1 for c in components if (c.get("terminal_detection") or {}).get("review_required"))
        result["persisted_manual_corrections_applied"] = applied
    return applied


def preparar_resposta_projeto(projeto):
    base_url = (
        f"http://127.0.0.1:8010/projects-files/"
        f"{projeto['project_id']}"
    )

    processed_filename = projeto.get("processed_filename")
    detection_filename = projeto.get("detection_filename")
    ocr_filename = projeto.get("ocr_filename")
    terminals_filename = projeto.get("terminals_filename")
    wires_filename = projeto.get("wires_filename")

    detections_info = _carregar_json_texto(
        projeto.get("detections_info")
    )
    ocr_info = _carregar_json_texto(projeto.get("ocr_info"))
    terminals_info = _carregar_json_texto(projeto.get("terminals_info"))
    wires_info = _carregar_json_texto(projeto.get("wires_info"))
    junctions_info = (wires_info or {}).get("stage9") or None
    connectivity_info, circuit_info = _resultados_eletricos_atuais(projeto, wires_info, terminals_info)

    return {
        "project_id": projeto["project_id"],
        "name": projeto["name"],
        "created_at": projeto["created_at"],
        "status": projeto["status"],
        "path": projeto["path"],
        "original_filename": projeto["original_filename"],
        "stored_filename": projeto["stored_filename"],
        "processed_filename": processed_filename,
        "detection_filename": detection_filename,
        "ocr_filename": ocr_filename,
        "terminals_filename": terminals_filename,
        "wires_filename": wires_filename,
        "image_url": f"{base_url}/{projeto['stored_filename']}",
        "processed_url": (
            f"{base_url}/{processed_filename}"
            if processed_filename
            else None
        ),
        "detections_url": (
            f"{base_url}/{detection_filename}"
            if detection_filename
            else None
        ),
        "ocr_url": (
            f"{base_url}/{ocr_filename}"
            if ocr_filename
            else None
        ),
        "components_url": (
            f"{base_url}/components.json"
            if ocr_info
            else None
        ),
        "terminals_url": (
            f"{base_url}/{terminals_filename}"
            if terminals_filename
            else None
        ),
        "terminals_json_url": (
            f"{base_url}/terminals.json"
            if terminals_info
            else None
        ),
        "wires_url": (f"{base_url}/{wires_filename}" if wires_filename else None),
        "wires_json_url": (f"{base_url}/wires.json" if wires_info else None),
        "wires_mask_url": (f"{base_url}/wires_mask.png" if wires_info else None),
        "wires_skeleton_url": (f"{base_url}/wires_skeleton.png" if wires_info else None),
        "transistor_wire_refinement_url": (f"{base_url}/transistor_wire_refinement.png" if wires_info and (PROJECTS_DIR / projeto['project_id'] / "transistor_wire_refinement.png").exists() else None),
        "junctions_url": (f"{base_url}/junctions.png" if junctions_info else None),
        "junctions_json_url": (f"{base_url}/junctions.json" if junctions_info else None),
        "junction_review_url": f"http://127.0.0.1:8010/projects/{projeto['project_id']}/junction-review",
        "workflow_url": f"http://127.0.0.1:8010/projects/{projeto['project_id']}/workflow",
        "stage8_status_url": f"http://127.0.0.1:8010/projects/{projeto['project_id']}/stage8-status",
        "terminal_trainer_url": f"http://127.0.0.1:8010/projects/{projeto['project_id']}/terminal-trainer",
        "stage7_readiness_url": f"http://127.0.0.1:8010/projects/{projeto['project_id']}/stage7-readiness",
        "processing_info": _carregar_json_texto(
            projeto.get("processing_info")
        ),
        "detections_info": detections_info,
        "ocr_info": ocr_info,
        "terminals_info": terminals_info,
        "wires_info": wires_info,
        "junctions_info": junctions_info,
        "connectivity_info": connectivity_info,
        "circuit_info": circuit_info,
        "connectivity_url": f"{base_url}/connectivity.png" if connectivity_info else None,
        "connectivity_json_url": f"{base_url}/connectivity.json" if connectivity_info else None,
        "circuit_json_url": f"http://127.0.0.1:8010/projects/{projeto['project_id']}/circuit/download" if circuit_info else None,
        "circuit_view_url": f"http://127.0.0.1:8010/projects/{projeto['project_id']}/circuit-view",
        "simulation_view_url": f"http://127.0.0.1:8010/projects/{projeto['project_id']}/simulation-view",
        "net_count": (connectivity_info or {}).get('net_count',0),
        "detection_count": (
            detections_info.get("count", 0)
            if detections_info
            else 0
        ),
        "logical_component_count": (
            ocr_info.get("count", 0)
            if ocr_info
            else 0
        ),
        "terminal_count": (
            terminals_info.get("terminal_count", 0)
            if terminals_info
            else 0
        ),
        "wire_segment_count": (wires_info.get("segment_count", 0) if wires_info else 0),
        "wire_endpoint_candidate_count": (wires_info.get("endpoint_candidate_count", 0) if wires_info else 0),
        "junction_connected_count": (junctions_info.get("connected_junction_count", 0) if junctions_info else 0),
        "junction_crossing_count": (junctions_info.get("unconnected_crossing_count", 0) if junctions_info else 0),
        "junction_review_count": (junctions_info.get("review_count", 0) if junctions_info else 0),
    }


async def salvar_novo_projeto(file: UploadFile, name: str = ""):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Nenhum arquivo foi enviado.",
        )

    nome_arquivo_recebido = Path(file.filename).name
    extensao = Path(nome_arquivo_recebido).suffix.lower()

    if extensao not in EXTENSOES_PERMITIDAS:
        raise HTTPException(
            status_code=400,
            detail="Formato inválido. Envie uma imagem JPG, JPEG ou PNG.",
        )

    nome_projeto = name.strip() or Path(nome_arquivo_recebido).stem
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    status = "criado"
    stored_filename = f"original{extensao}"

    while True:
        temporario = f"PENDING_{uuid4().hex}"

        id_interno = inserir_projeto_temporario(
            temporario,
            nome_projeto,
            created_at,
            status,
            nome_arquivo_recebido,
            stored_filename,
        )

        project_id = f"PRJ_{id_interno:04d}"
        project_dir = PROJECTS_DIR / project_id

        if not project_dir.exists():
            break

        excluir_projeto_por_id_interno(id_interno)

    try:
        project_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        caminho_imagem = project_dir / stored_filename

        with caminho_imagem.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        finalizar_projeto(
            id_interno,
            project_id,
            f"projects/{project_id}",
        )

    except Exception as erro:
        excluir_projeto_por_id_interno(id_interno)

        if project_dir.exists():
            shutil.rmtree(project_dir, ignore_errors=True)

        raise HTTPException(
            status_code=500,
            detail=f"Não foi possível criar o projeto: {erro}",
        )

    projeto = buscar_projeto(project_id)
    resposta = preparar_resposta_projeto(projeto)
    resposta["mensagem"] = "Projeto criado com sucesso."
    return resposta


@app.get("/")
def inicio():
    return {
        "status": "ok",
        "mensagem": "Circuit Scanner - reconhecimento assistido, simulação e exportação",
        "interface": "http://127.0.0.1:8010/workflow",
    }


@app.get("/workflow", response_class=HTMLResponse)
def abrir_fluxo_de_projetos():
    """Entrada única: envio, reconhecimento, OCR e revisões no mesmo fluxo."""
    return HTMLResponse(content=_guided_page())


@app.post("/projects")
async def criar_projeto(
    file: UploadFile = File(...),
    name: str = Form(""),
):
    return await salvar_novo_projeto(
        file=file,
        name=name,
    )


@app.post("/upload")
async def upload_imagem(
    file: UploadFile = File(...),
    name: str = Form(""),
):
    return await salvar_novo_projeto(
        file=file,
        name=name,
    )


@app.get("/projects/summary")
def obter_resumos_projetos():
    return listar_resumos_projetos()


@app.get("/projects")
def obter_lista_de_projetos():
    # Se arquivos de projetos existem no disco mas o índice SQLite foi trocado
    # ou apagado, recupera os registros antes de responder ao front-end.
    _recuperar_indice_projetos_do_disco()
    return [
        preparar_resposta_projeto(projeto)
        for projeto in listar_projetos()
    ]


@app.get("/projects/{project_id}")
def obter_projeto(project_id: str):
    projeto = buscar_projeto(project_id)
    if projeto is None:
        _recuperar_indice_projetos_do_disco()
        projeto = buscar_projeto(project_id)

    if projeto is None:
        raise HTTPException(
            status_code=404,
            detail="Projeto não encontrado.",
        )

    caminho_imagem = (
        PROJECTS_DIR
        / projeto["project_id"]
        / projeto["stored_filename"]
    )

    if not caminho_imagem.exists():
        raise HTTPException(
            status_code=404,
            detail="A imagem original do projeto não foi encontrada.",
        )

    return preparar_resposta_projeto(projeto)


@app.get("/projects/{project_id}/workflow-advanced", response_class=HTMLResponse)
def abrir_fluxo_do_projeto(project_id: str):
    """Painel de continuidade: reconhecer, revisar e seguir para linhas."""
    projeto = buscar_projeto(project_id)
    if projeto is None:
        raise HTTPException(status_code=404, detail="Projeto nao encontrado.")
    payload = json.dumps({"project_id": project_id}, ensure_ascii=False)
    html = """<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'><title>Fluxo do projeto</title>
<style>
body{font-family:Arial,sans-serif;margin:18px;background:#f4f6f8;color:#17202a}.bar{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin:12px 0}button,a.btn{border:0;border-radius:6px;padding:10px 13px;background:#1769aa;color:#fff;text-decoration:none;font-size:14px;cursor:pointer}button.secondary,a.secondary{background:#465a65}button.review,a.review{background:#b45309}.status{background:#fff;border:1px solid #d5dde5;border-radius:8px;padding:12px;line-height:1.45;white-space:pre-wrap}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:12px;margin-top:14px}.panel{background:#fff;border:1px solid #d5dde5;border-radius:8px;padding:10px}.panel img{width:100%;max-height:420px;object-fit:contain;background:#fafafa}.panel h3{margin:2px 0 8px}.ok{color:#087a2f}.warn{color:#b00020}
</style></head><body><a href='/workflow'>Voltar aos projetos</a><h2 id='title'>Fluxo do projeto</h2><p>Primeiro componentes, depois fios e por ultimo terminais dentro das caixas. Ao refazer os fios, os contatos dos terminais tambem sao atualizados. As correcoes manuais dentro das caixas sao preservadas.</p>
<div class='bar'><button id='detect'>Reconhecer e gerar linhas</button><button id='ocr' class='secondary'>Ler OCR e atualizar linhas</button><button id='wires' class='secondary'>Refazer fios e terminais</button><button id='junctions' class='secondary'>Resolver juncoes</button><a id='edit' class='btn review'>Corrigir terminais</a><a id='junctionReview' class='btn review'>Revisar juncoes</a><a id='modelView' class='btn review'>Valores e modelos OCR</a><a id='circuitView' class='btn secondary'>Redes e circuit.json</a><a id='circuitReview' class='btn review'>Revisar circuito — etapa 12</a><a id='simulationView' class='btn secondary'>Simular — etapas 13 e 14</a><a id='productionView' class='btn secondary'>Esquemático, PCB e ZIP — etapas 15 a 18</a><a href='/component-trainer' class='btn secondary'>Treinar componentes</a><button id='refresh' class='secondary'>Atualizar tela</button></div><div id='status' class='status'>Carregando...</div><div id='images' class='grid'></div>
<script id='payload' type='application/json'>__PAYLOAD__</script><script>
const cfg=JSON.parse(document.getElementById('payload').textContent),id=cfg.project_id,statusBox=document.getElementById('status'),images=document.getElementById('images');document.getElementById('edit').href='/projects/'+id+'/terminal-trainer';document.getElementById('junctionReview').href='/projects/'+id+'/junction-review';document.getElementById('modelView').href='/projects/'+id+'/ocr-models-view';document.getElementById('circuitView').href='/projects/'+id+'/circuit-view';document.getElementById('circuitReview').href='/projects/'+id+'/circuit-review-view';document.getElementById('simulationView').href='/projects/'+id+'/simulation-view';document.getElementById('productionView').href='/projects/'+id+'/production-view';
function panel(title,url){if(!url)return '';return `<div class="panel"><h3>${title}</h3><img src="${url}?t=${Date.now()}" alt="${title}"></div>`}
async function refresh(message=''){const r=await fetch('/projects/'+id);const p=await r.json();if(!r.ok)throw new Error(p.detail||'Erro');document.getElementById('title').textContent=p.name+' — '+id;const tr=(p.wires_info||{}).transistor_terminal_refinement||{};const reviews=(p.terminals_info||{}).review_count||0,jr=p.junction_review_count||0;statusBox.className='status '+(reviews||jr?'warn':'ok');statusBox.textContent=(message?message+'\\n':'')+'Estado: '+p.status+'\\nTerminais: '+(p.terminal_count||0)+' | Linhas: '+(p.wire_segment_count||0)+' | Revisoes de terminais: '+reviews+'\\nJuncoes conectadas: '+(p.junction_connected_count||0)+' | Cruzamentos separados: '+(p.junction_crossing_count||0)+' | Revisoes de juncao: '+jr+'\\nTransistores refinados: '+(tr.refined_count||0)+'/'+(tr.transistor_count||0);images.innerHTML=panel('Original',p.image_url)+panel('Componentes',p.detections_url)+panel('Fios entre componentes',p.wires_url)+panel('Terminais dentro das caixas',p.terminals_url)+panel('Juncoes e cruzamentos',p.junctions_url)+panel('Redes eletricas',p.connectivity_url)+panel('Refinamento do transistor',p.transistor_wire_refinement_url)}
async function run(path,label){statusBox.className='status';statusBox.textContent=label+'...';try{const r=await fetch('/projects/'+id+'/'+path,{method:'POST'});const d=await r.json();if(!r.ok)throw new Error(d.detail||'Erro');await refresh(d.mensagem||label+' concluido.')}catch(e){statusBox.className='status warn';statusBox.textContent='Erro: '+e.message}}
document.getElementById('detect').onclick=()=>run('detect','Reconhecendo e gerando linhas');document.getElementById('ocr').onclick=()=>run('ocr','Lendo OCR e atualizando linhas');document.getElementById('wires').onclick=()=>run('wires','Refazendo fios e terminais');document.getElementById('junctions').onclick=()=>run('junctions','Resolvendo juncoes e cruzamentos');document.getElementById('refresh').onclick=()=>refresh();refresh().catch(e=>statusBox.textContent='Erro: '+e.message);
</script></body></html>""".replace("__PAYLOAD__", payload.replace("</", "<\\/"))
    return HTMLResponse(content=html)


@app.get("/projects/{project_id}/junction-review", response_class=HTMLResponse)
def abrir_revisao_juncoes(project_id: str):
    """Visual junction review in the current guided workflow."""
    if buscar_projeto(project_id) is None:
        raise HTTPException(status_code=404, detail="Projeto não encontrado.")
    payload = json.dumps({"project_id": project_id}, ensure_ascii=False)
    html = """<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'><title>Revisao de juncoes</title>
<style>body{font-family:Arial,sans-serif;margin:18px;background:#f4f6f8;color:#17202a}.layout{display:grid;grid-template-columns:minmax(360px,1.2fr) minmax(340px,1fr);gap:14px}.panel{background:#fff;border:1px solid #d5dde5;border-radius:8px;padding:12px}.panel img{width:100%;max-height:75vh;object-fit:contain;cursor:crosshair}.row{display:grid;grid-template-columns:110px 1fr;gap:8px;align-items:center;padding:8px 0;border-bottom:1px solid #e5e7eb}select,button{padding:8px;border-radius:6px;border:1px solid #b9c3cc}button{background:#1769aa;color:#fff;cursor:pointer;margin-top:12px}button:disabled{background:#94a3b8;cursor:not-allowed}.msg{white-space:pre-wrap;margin:10px 0}.legend{font-size:13px;line-height:1.5;color:#52606d}@media(max-width:850px){.layout{grid-template-columns:1fr}}</style></head><body><a id='back'>Voltar ao fluxo</a><h2>Etapa 9 — Juncoes e cruzamentos</h2><p class='legend'>Verde: conectado. X roxo: cruzamento sem conexao. Laranja: revisar. A regra automatica conecta T; cruzamento de quatro vias so conecta com ponto visivel. Se um T nao aparecer, clique exatamente sobre ele e use Adicionar juncao.</p><div class='layout'><div class='panel'><img id='preview'><button id='add' disabled>Adicionar juncao no ponto clicado</button><div id='clickMsg' class='legend'>Clique na imagem para marcar uma juncao ausente.</div><div id='msg' class='msg'>Carregando...</div></div><div class='panel'><div id='rows'></div><button id='save'>Salvar correcoes</button></div></div>
<script id='payload' type='application/json'>__PAYLOAD__</script><script>
</script><script src='/junction-review.js?v=5.40.0' onerror="document.getElementById('msg').textContent='Não foi possível carregar a revisão. Atualize a página.'"></script></body></html>""".replace("__PAYLOAD__", payload.replace("</", "<\\/"))
    html=html.replace('<h2>Etapa 9 — Juncoes e cruzamentos</h2>','<h2>Junções e cruzamentos</h2>').replace("<img id='preview'>","<img id='preview' alt='Circuito com junções e cruzamentos' hidden>")
    html=html.replace("<button id='save'>Salvar correcoes</button>","<button id='save' disabled>Salvar correcoes</button><button id='retry' hidden>Tentar novamente</button>")
    return HTMLResponse(content=html)


@app.get('/junction-review.js')
def script_revisao_juncoes():
    return FileResponse(BASE_DIR/'frontend/junction-review.js',media_type='application/javascript')


@app.post("/projects/{project_id}/process")
@cached_project_stage('process')
def processar_projeto(project_id: str, force: bool = False):
    projeto = buscar_projeto(project_id)

    if projeto is None:
        raise HTTPException(
            status_code=404,
            detail="Projeto não encontrado.",
        )

    project_dir = PROJECTS_DIR / project_id
    caminho_original = project_dir / projeto["stored_filename"]

    if not caminho_original.exists():
        raise HTTPException(
            status_code=404,
            detail="A imagem original do projeto não foi encontrada.",
        )

    caminho_processado = project_dir / "processed.png"
    caminho_json = project_dir / "processing.json"

    try:
        informacoes = processar_imagem(
            caminho_original=caminho_original,
            caminho_processado=caminho_processado,
        )

        caminho_json.write_text(
            json.dumps(
                informacoes,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        atualizar_processamento(
            project_id=project_id,
            status="processado",
            processed_filename="processed.png",
            processing_info=json.dumps(
                informacoes,
                ensure_ascii=False,
            ),
        )

        # Se havia arquivos de uma detecção antiga, removemos porque
        # a imagem processada acabou de mudar.
        for nome in (
            "detections.png",
            "detections.json",
            "ocr.png",
            "components.json",
            "terminals.png",
            "terminals.json",
        ):
            antigo = project_dir / nome
            if antigo.exists():
                antigo.unlink()

    except Exception as erro:
        raise HTTPException(
            status_code=500,
            detail=f"Erro durante o pré-processamento: {erro}",
        )

    projeto_atualizado = buscar_projeto(project_id)
    resposta = preparar_resposta_projeto(projeto_atualizado)
    resposta["mensagem"] = "Imagem processada com sucesso."
    return resposta


@app.get("/dataset/info")
def obter_dataset_info():
    return dataset_info()


@app.post("/dataset/generate")
def gerar_dataset(per_class: int = 60, seed: int = 4411):
    try:
        manifest = generate_dataset(
            per_class=per_class,
            seed=seed,
            clear=True,
        )
        generate_previews()

        return {
            "mensagem": "Dataset sintético gerado com sucesso.",
            "manifest": manifest,
            "info": dataset_info(),
        }

    except ValueError as erro:
        raise HTTPException(status_code=400, detail=str(erro))

    except Exception as erro:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao gerar dataset: {erro}",
        )


@app.get("/recognition/status")
def obter_status_reconhecimento():
    return model_status()


@app.get("/system/version")
def obter_versao_sistema():
    return {
        "build": "2026-10-19-evaluation-regressions-v5.49.0",
        "mixed_ink_text_preserved": True,
        "ocr_confidence_probability_boundary": True,
        "optional_local_print_reader": True,
        "saved_wires_junction_recovery": True,
        "observed_external_lead_recovery": True,
        "native_export_connections_verified": True,
        "compact_falstad_wires": True,
        "compact_native_graph_verified": True,
        "supervised_ocr_font_specialist": True,
        "supervised_text_component_guard": True,
        "supervised_notebook_text_context": True,
        "atomic_windows_save_retry": True,
        "optional_timing_never_blocks_step": True,
        "neighbor_bjt_consensus_separation": True,
        "observed_short_external_leads": True,
        "scaled_one_lead_power_head": True,
        "separate_plates_polarity_recovery": True,
        "cpu_worker_budget_restored": True,
        "complete_numeric_prefix_unit_reading": True,
        "opamp_original_sign_ink": True,
        "ground_upright_photo_layout": True,
        "terminal_save_reuses_wire_graph": True,
        "trained_handwriting_glyphs": True,
        "complete_component_reference_grammar": True,
        "shared_photo_analysis_cache": True,
        "metadata_edits_preserve_geometry": True,
        "cpu_model_background_preparation": True,
        "junction_review_prepares_preview_before_clicks": True,
        "native_autosave_vcc_voltage_supported": True,
        "dc_source_hollow_ring_topology": True,
        "local_falstad_api": True,
        "spatial_connected_schematic": True,
        "processing_target_seconds": 30,
        "local_ground_drawing_forest": True,
        "native_falstad_grid_pixels": 8,
        "resistor_terminals_outside_body": True,
        "opamp_local_sign_ocr": True,
        "photo_wire_bends_preserved": True,
        "ocr_component_neighborhood_priority": True,
        "ocr_contextual_beta_and_equal_correction": True,
        "ocr_gain_numeric_crop_after_visible_equal": True,
        "cpu_exact_neighborhood_table_thinning": True,
        "ocr_text_lines_recognition_only": True,
        "ocr_overlapping_text_boxes_consolidated": True,
        "ocr_B_equals_transistor_beta": True,
        "cpu_bjt_quality_upper_bound_pruning": True,
        "cpu_connection_spatial_prefilter": True,
        "cpu_pure_vision_session_cache": True,
        "visible_arrows_and_plus_define_component_family": True,
        "component_boxes_include_semantic_marks": True,
        "cpu_inference_threads_bounded": True,
        "cpu_graph_cluster_scan_once": True,
        "ocr_bounded_global_canvas": True,
        "processing_timings_exclude_nested_stages": True,
        "polarized_opposed_lead_contacts": True,
        "polarity_plus_requires_four_visible_arms": True,
        "polarity_body_excludes_detached_plus": True,
        "terminal_autosave_before_navigation": True,
        "capacitor_open_plate_text_guard": True,
        "ocr_models_and_electrical_parameters": True,
        "datasheet_parameters_opt_in": True,
        "new_model_simulation_integration": True,
        "stage15_editable_kicad_schematic": True,
        "stage16_optional_unrouted_pcb": True,
        "stage17_zip_share_qr": True,
        "user_ready_download_screen": True,
        "falstad_only": True,
        "reference_image_ocr": True,
        "simulation_requires_confirmed_circuit": True,
        "text_groups_require_connected_symbol_evidence": True,
        "inductor_continuous_curve_two_lead_metric": True,
        "source_ring_requires_connected_leads": True,
        "ocr_single_letters_and_numbers_with_physical_symbol_protection": True,
        "opamp_original_signs_on_final_contacts": True,
        "bjt_exclusive_branch_arrow_head_metric": True,
        "resistor_closed_body_or_connected_zigzag_with_opposed_leads": True,
        "vcc_original_image_single_lead_validation": True,
        "notebook_colors_with_repeated_line_validation": True,
        "notebook_filter_shared_by_wires_terminals_junctions": True,
        "notebook_rule_filter_version": "5.14.5",
        "notebook_dark_rules_tracked_under_shadows": True,
        "stage10_electrical_nets": True,
        "stage11_image_independent_circuit_json": True,
        "stage12_human_circuit_review": True,
        "circuit_schema_version": "circuit-1.0",
        "junction_branches_verified_on_original_ink": True,
        "junction_dot_candidates_merged": True,
        "polarity_glyphs_excluded_from_contacts": True,
        "polarity_on_final_wire_contacts": True,
        "battery_long_plate_without_plus": True,
        "wire_detection_restored_version": "5.13.6",
        "terminal_border_ink_paths": True,
        "npn_original_ink_contacts_after_stage8": True,
        "npn_rotated_base_bar_and_arrow_contacts": True,
        "visible_polarity_sign_terminal_identity": True,
        "opamp_visible_triangle_and_input_signs": True,
        "mosfet_visible_gate_source_connector": True,
        "wire_component_ink_color_consensus": True,
        "junction_component_ink_color_filter": True,
        "api_port": 8010,
        "manual_detection_review": True,
        "filled_diode_geometry": True,
        "led_emission_arrow_resolution": True,
        "paper_background_robust_geometry": True,
        "geometry_multiview_safe_rescue": True,
        "rotated_gnd_recognition": True,
        "strict_iec_resistor_recovery": True,
        "terminal_detection": True,
        "terminal_schema_version": "stage7-4.1-visible-terminal-identity",
        "automatic_terminals_after_ocr": True,
        "automatic_terminals_after_detection": True,
        "automatic_terminals_after_detection_edit": True,
        "resistor_body_axis": True,
        "resistor_axis_trace": True,
        "capacitor_plate_center_trace": True,
        "bjt_type_first_arrow_validation": True,
        "uncertain_terminal_review": True,
        "terminal_local_connection_snap": True,
        "capacitor_parallel_plate_axis": True,
        "project_index_recovery": True,
        "led_body_axis": True,
        "pnp_constrained_fork_snap": True,
        "terminal_manual_training": True,
        "terminal_keypoint_model": True,
        "terminal_trainer_page": True,
        "terminal_keypoint_model_first": False,
        "physical_transistor_evidence_first": True,
        "npn_lower_emitter_mirror_rule": True,
        "npn_corrected_points_model": True,
        "terminal_keypoint_runtime_diagnostics": True,
        "terminal_keypoint_dataset_audit": True,
        "transistor_snap_protected": True,
        "pose_class_does_not_override_detector": True,
        "validation_split_grouped_by_project": True,
        "training_visual_prior": True,
        "generic_terminal_manual_review": True,
        "stage7_readiness_gate": True,
        "wire_detection": True,
        "wire_schema_version": "stage8-wires-1.0",
        "wire_component_masking": True,
        "wire_text_masking": True,
        "wire_skeletonization": True,
        "wire_graph_segments": True,
        "notebook_ruling_suppression": True,
        "automatic_stage8_after_terminals": False,
        "pipeline_order": ["components", "wires", "terminals", "junctions"],
        "terminals_strictly_inside_bbox": True,
        "all_classes_wire_contacts_after_wires": True,
        "all_classes_rotational_recognition": True,
        "component_ink_view_in_recognition": True,
        "manual_BCE_correction_rebuilds_wires": True,
        "workflow_page_without_docs": True,
        "guided_workflow": True,
        "transistor_multi_pass_symbol_analysis": True,
        "all_dataset_classes_terminal_editor": True,
        "all_component_terminal_training_dataset": True,
        "seven_class_terminal_pose_model": True,
        "seven_class_terminal_pose_safe_fallback": True,
        "dynamic_pin_schema_for_future_classes": True,
        "component_trainer_page": True,
        "component_training_background_job": True,
        "component_model_backup_before_training": True,
        "real_mixed_dataset_training": True,
        "versioned_mixed_datasets": True,
        "windows_locked_dataset_fix": True,
        "component_training_resume_checkpoint": True,
        "ruled_paper_led_rejection": True,
        "correlated_view_consensus_damping": True,
        "mirrored_bjt_geometry": True,
        "opamp_geometry_validation": True,
        "reviewed_empty_real_negatives": True,
        "real_validation_metrics_by_class": True,
        "small_monochrome_detail_tiles": False,
        "class_safe_line_art_rescue": True,
        "low_confidence_resistor_text_rejection": True,
        "cardinal_rotation_synthetic_training": True,
        "schematic_label_clutter_training": True,
        "geometric_resistor_proposals": False,
        "geometric_bjt_proposals": False,
        "yolo_independent_symbol_fallback": False,
        "universal_monochrome_dataset_classes": True,
        "monochrome_multiscale_pyramid": False,
        "mandatory_real_monochrome_training_variants": True,
        "dynamic_dataset_class_rescue": True,
        "pyramid_requires_full_view_corroboration": True,
        "monochrome_cross_variant_consensus": True,
        "pyramid_only_false_positive_rejection": True,
        "opamp_triangle_geometry_proposal": True,
        "diode_physical_geometry_validation": True,
        "ruled_line_diode_rejection": True,
        "low_contrast_opamp_geometry": True,
        "opamp_polarity_anchor": True,
        "iec_rectangular_resistor_geometry": True,
        "parallel_plate_capacitor_geometry": True,
        "edge_clipped_capacitor_geometry": True,
        "edge_base_bjt_geometry": True,
        "stage9_electrical_junction_resolution": True,
        "t_junction_connects": True,
        "four_way_requires_visible_dot": True,
        "unconnected_crossing_opposite_pairing": True,
        "manual_junction_review_page": True,
        "automatic_stage9_after_wires": True,
        "stage9_endpoint_to_wire_t_recovery": True,
        "manual_click_add_junction": True,
        "terminal_plus_passing_wire_forms_t": True,
        "manual_junction_uses_terminal_context": True,
        "junction_click_coordinate_letterbox_fix": True,
        "junction_click_diagnostics": True,
        "manual_two_endpoint_junction_confirmation": True,
        "adaptive_manual_junction_snap": True,
        "cad_training_upload_area": True,
        "dataset_source_type_tracking": True,
        "cad_source_group_split_protection": True,
        "cad_and_handwritten_combined_training": True,
        "restore_previous_component_model_button": True,
        "candidate_model_quality_gate": True,
        "negative_image_false_positive_audit": True,
        "high_confidence_geometry_validation": True,
        "inductor_battery_gnd_mosfet_geometry_filters": True,
        "capacitor_and_resistor_strict_geometry_filters": True,
        "ai_first_component_detection": True,
        "confidence_never_inflated_by_duplicate_views": True,
        "stable_foundation_component_training": True,
        "user_confirmed_foundation_model": True,
        "focused_real_component_training_tiles": True,
        "real_page_hard_negative_tiles": True,
        "geometry_validation_does_not_invent_components": True,
        "dataset_annotation_quality_audit": True,
        "pen_color_precision_recall_evaluation": True,
        "anchor_based_multiview_grouping": True,
        "oversized_multi_component_box_rejection": True,
        "cross_class_annotation_overlap_report": True,
        "graceful_training_stop_and_best_validation": True,
    }


@app.get("/system/storage")
def obter_status_armazenamento():
    return {
        "database_path": str(DATABASE_PATH),
        "database_exists": DATABASE_PATH.exists(),
        "database_project_count": contar_projetos(),
        "projects_directory": str(PROJECTS_DIR),
        "project_folders": len([p for p in PROJECTS_DIR.glob("PRJ_*") if p.is_dir()]),
        "legacy_recovered_names_cleaned": LEGACY_RECOVERED_NAMES_CLEANED,
        "startup_recovery": STORAGE_RECOVERY,
    }


@app.post("/system/storage/reindex")
def reindexar_armazenamento():
    return _recuperar_indice_projetos_do_disco()


@app.get("/terminal-learning/status")
def obter_status_treino_terminais():
    return terminal_learning_status()


@app.get("/terminal-learning/all-components/status")
def obter_status_treino_terminais_todos_componentes():
    status = generic_terminal_learning_status()
    status["trained_model"] = all_component_terminal_model_status(load_model=False)
    return status


@app.get("/terminal-learning/all-components/model-runtime")
def obter_runtime_modelo_terminais_todos_componentes():
    return all_component_terminal_model_status(load_model=True)


@app.get("/terminal-learning/model-runtime")
def obter_runtime_modelo_terminais():
    """Confirma se o .pt existe E se realmente pode ser carregado como YOLO-Pose."""
    return terminal_model_runtime_status(load_model=True)


@app.get("/projects/{project_id}/terminal-learning/diagnostics")
def diagnosticar_keypoints_do_projeto(project_id: str):
    projeto = buscar_projeto(project_id)
    if projeto is None:
        raise HTTPException(status_code=404, detail="Projeto não encontrado.")
    project_dir = PROJECTS_DIR / project_id
    source_path = project_dir / projeto["stored_filename"]
    image = cv2.imread(str(source_path))
    if image is None:
        raise HTTPException(status_code=404, detail="Imagem original do projeto não encontrada.")
    components_info = _latest_components_for_project(projeto)
    diagnostics = []
    corrections = _load_terminal_corrections(project_id)
    corrected_ids = {str(r.get("component_id") or "") for r in corrections.get("components") or []}
    for comp in components_info.get("components") or []:
        ctype = str(comp.get("type") or "")
        if ctype not in TRANSISTOR_CLASSES:
            continue
        cid = str(comp.get("component_id") or comp.get("detection_id") or "")
        bbox = comp.get("bbox")
        if not isinstance(bbox, dict):
            continue
        known_match = lookup_known_training_ground_truth(image=image, bbox=bbox, component_type=ctype)
        diag = diagnose_transistor_keypoints(image=image, bbox=bbox, component_type=ctype)
        visual_prior = None
        try:
            visual_prior = predict_transistor_visual_prior(image=image, bbox=bbox, component_type=ctype)
        except Exception:
            visual_prior = None
        if known_match:
            effective = "known_training_ground_truth"
        elif diag.get("accepted") and not diag.get("review_required", True):
            effective = "trained_terminal_keypoint_model"
        elif visual_prior and visual_prior.get("accepted"):
            effective = "training_visual_prior"
        elif diag.get("accepted"):
            effective = "trained_terminal_keypoint_model_review"
        else:
            effective = "review_required"
        diagnostics.append({
            "component_id": cid,
            "type": ctype,
            "bbox": bbox,
            "manual_correction_saved": cid in corrected_ids,
            "known_training_match": known_match,
            "visual_prior": visual_prior,
            "effective_source": effective,
            "diagnostic": diag,
        })
    return {
        "project_id": project_id,
        "model_runtime": terminal_model_runtime_status(load_model=True),
        "transistor_count": len(diagnostics),
        "transistors": diagnostics,
    }


@app.get("/projects/{project_id}/stage7-readiness")
def verificar_prontidao_etapa7(project_id: str):
    """Report terminal review independently of readiness to detect wires."""
    projeto = buscar_projeto(project_id)
    if projeto is None:
        raise HTTPException(status_code=404, detail="Projeto não encontrado.")
    result = _terminal_result_for_manual_edit(project_id, projeto)
    _apply_saved_terminal_corrections(project_id, result)
    unresolved = []
    valid = []
    deferred_to_stage8 = []
    unsupported = []
    for comp in result.get("components") or []:
        cid = str(comp.get("component_id") or comp.get("detection_id") or "")
        ctype = str(comp.get("type") or "")
        schema = _schema_for_component(comp)
        if not schema:
            unsupported.append({"component_id": cid, "type": ctype})
            continue
        expected = [pin for pin, _role in schema]
        terminals = comp.get("terminals") or []
        terminal_detection = comp.get("terminal_detection") or {}
        deferred_transistor = (
            ctype in {"transistor_npn", "transistor_pnp"}
            and terminal_detection.get("status") == "awaiting_stage8_wires"
            and terminal_detection.get("method") == "transistor_points_after_stage8_wires"
        )
        if deferred_transistor:
            deferred_to_stage8.append({
                "component_id": cid,
                "type": ctype,
                "expected_pins": expected,
                "status": "awaiting_stage8_wires",
                "source": (comp.get("transistor_geometry") or {}).get("source"),
            })
            continue
        found = {str(t.get("pin")): t for t in terminals if isinstance(t.get("position"), dict)}
        missing = [pin for pin in expected if pin not in found]
        review = bool(terminal_detection.get("review_required"))
        invalid_pos = []
        for pin in expected:
            if pin not in found:
                continue
            pos = found[pin].get("position") or {}
            try:
                x,y=float(pos["x"]),float(pos["y"])
                box=comp['bbox']
                if not (float(box['x1'])<x<float(box['x2']) and float(box['y1'])<y<float(box['y2'])):
                    invalid_pos.append(pin)
            except Exception:
                invalid_pos.append(pin)
        if missing or invalid_pos or review:
            unresolved.append({
                "component_id": cid,
                "type": ctype,
                "missing_pins": missing,
                "invalid_position_pins": invalid_pos,
                "review_required": review,
                "review_reasons": terminal_detection.get("review_reasons", []),
                "source": (comp.get("transistor_geometry") or {}).get("source") or (terminals[0].get("source") if terminals else None),
            })
        else:
            valid.append({
                "component_id": cid,
                "type": ctype,
                "pins": expected,
                "source": terminals[0].get("source") if terminals else None,
            })
    # Classes fora do escopo básico não bloqueiam automaticamente a Etapa 8,
    # mas ficam visíveis para o usuário. O gate exige todos os componentes que
    # possuem esquema de terminais na versão atual.
    # Transistores automáticos são uma exceção intencional: os três pontos B/C/E
    # só podem ser definidos com segurança depois que o skeleton de fios existe.
    # Eles contam como contexto válido para iniciar a Etapa 8, mas continuam
    # explícitos em deferred_to_stage8 até o refinamento posterior.
    ready = len(unresolved) == 0 and (len(valid) + len(deferred_to_stage8)) > 0
    return {
        "project_id": project_id,
        "ready_for_stage8": bool(result.get('components')),
        "terminals_ready_for_connectivity": ready and not deferred_to_stage8,
        "pipeline_order": ["components", "wires", "terminals", "junctions"],
        "supported_component_count": len(valid) + len(unresolved),
        "valid_component_count": len(valid),
        "deferred_transistor_count": len(deferred_to_stage8),
        "unresolved_count": len(unresolved),
        "unsupported_count": len(unsupported),
        "unresolved": unresolved,
        "deferred_to_stage8": deferred_to_stage8,
        "unsupported": unsupported,
        "terminal_review_url": f"http://127.0.0.1:8010/projects/{project_id}/terminal-trainer",
        "terminal_preview_url": f"http://127.0.0.1:8010/projects-files/{project_id}/terminals.png",
        "next_step": "Revisar terminais e juncoes" if ready else "Refazer fios e terminais ou revisar os contatos indicados",
    }


@app.post("/terminal-learning/repair-samples")
def reparar_amostras_antigas_de_terminais():
    try:
        return repair_training_samples_from_metadata()
    except Exception as erro:
        raise HTTPException(status_code=500, detail=f"Erro ao reparar amostras de terminais: {erro}")


@app.post("/terminal-learning/train")
def treinar_modelo_terminais(
    epochs: int = 80,
    imgsz: int = 320,
    batch: int = 8,
    base_model: str = "yolo11n-pose.pt",
    device: str | None = None,
):
    try:
        return train_terminal_keypoint_model(
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            base_model=base_model,
            device=device,
        )
    except ValueError as erro:
        raise HTTPException(status_code=409, detail=str(erro))
    except Exception as erro:
        raise HTTPException(status_code=500, detail=f"Erro no treino de terminais: {erro}")


@app.put("/projects/{project_id}/terminals/draft")
@time_project_operation('terminal_draft')
def salvar_rascunho_terminais(project_id: str, payload: dict):
    from backend.terminal_review_draft import save
    projeto=buscar_projeto(project_id)
    if projeto is None:raise HTTPException(404,'Projeto não encontrado.')
    folder=PROJECTS_DIR/project_id;image=folder/projeto['stored_filename']
    if not image.is_file():raise HTTPException(404,'Imagem original não encontrada.')
    components=_terminal_result_for_manual_edit(project_id,projeto).get('components',[])
    try:
        count=save(folder,image,components,payload.get('components'),_find_component,_schema_for_component,_clean_pin_schema,MANUAL_PIN_SCHEMAS)
    except ValueError as error:raise HTTPException(400,str(error)) from error
    return {'saved':True,'draft_components':count,'mensagem':'Marcações salvas automaticamente. Complete os pinos ao voltar aos terminais.'}


@app.put("/projects/{project_id}/terminals/corrections")
@time_project_operation('terminal_review')
@symbol_analysis_session
def corrigir_terminais_e_salvar_treino(project_id: str, payload: dict):
    projeto = buscar_projeto(project_id)
    if projeto is None:
        raise HTTPException(status_code=404, detail="Projeto não encontrado.")
    edits = payload.get("components")
    if not isinstance(edits, list) or not edits:
        raise HTTPException(status_code=400, detail="components deve ser uma lista não vazia.")

    project_dir = PROJECTS_DIR / project_id
    source_path = project_dir / projeto["stored_filename"]
    image = cv2.imread(str(source_path))
    if image is None:
        raise HTTPException(status_code=404, detail="Imagem original do projeto não encontrada.")
    h, w = image.shape[:2]
    result = _terminal_result_for_manual_edit(project_id, projeto)
    components = result.get("components") or []
    training_samples = []
    all_component_training_samples = []
    corrected_count = 0

    for edit in edits:
        component_id = str(edit.get("component_id") or "").strip()
        if not component_id:
            raise HTTPException(status_code=400, detail="Cada correção precisa de component_id.")
        component = _find_component(components, component_id)
        if component is None:
            raise HTTPException(status_code=404, detail=f"Componente {component_id} não encontrado.")
        component_type = str(component.get("type") or "")
        points = edit.get("points")
        schema = _schema_for_component(
            component,
            requested_schema=edit.get("pin_schema"),
            points=points if isinstance(points, dict) else None,
        )
        if not schema:
            raise HTTPException(status_code=400, detail=f"{component_id}: informe ao menos um nome de pino antes de salvar.")
        if not isinstance(points, dict):
            raise HTTPException(status_code=400, detail=f"{component_id}: points deve ser um objeto com os pinos da classe.")
        required_pins = [pin for pin, _role in schema]
        missing = [pin for pin in required_pins if pin not in points]
        if missing:
            raise HTTPException(status_code=400, detail=f"{component_id}: faltam pontos {', '.join(missing)}.")

        bbox=component.get("bbox") or {}
        for pin in required_pins:
            try:
                x,y=round(float(points[pin]["x"]),2),round(float(points[pin]["y"]),2)
                inside=(0<=x<w and 0<=y<h and float(bbox["x1"])<x<float(bbox["x2"])
                        and float(bbox["y1"])<y<float(bbox["y2"]))
            except (KeyError,TypeError,ValueError):
                inside=False
            if not inside:
                raise HTTPException(status_code=400,detail=f"{component_id}.{pin}: o terminal deve ficar dentro da caixa do componente.")

        try:
            generic_sample = save_generic_terminal_training_sample(
                image_path=source_path,
                project_id=project_id,
                component=component,
                points=points,
                pin_schema=[{"pin": pin, "role": role} for pin, role in schema],
            )
        except (ValueError, FileNotFoundError, OSError) as erro:
            raise HTTPException(status_code=400, detail=str(erro))
        all_component_training_samples.append(generic_sample["sample_id"])

        sample = None
        if component_type in TRANSISTOR_CLASSES:
            try:
                sample = save_transistor_training_sample(
                    image_path=source_path,
                    project_id=project_id,
                    component=component,
                    points=points,
                )
            except (ValueError, FileNotFoundError) as erro:
                raise HTTPException(status_code=400, detail=str(erro))
            training_samples.append(sample["sample_id"])

        terminals = []
        for idx, (pin, role) in enumerate(schema, start=1):
            item = points[pin]
            x, y = float(item["x"]), float(item["y"])
            if not (0 <= x < w and 0 <= y < h):
                raise HTTPException(status_code=400, detail=f"{component_id}.{pin} está fora da imagem.")
            terminals.append({
                "pin_id": f"{component_id}.{pin}",
                "pin": pin,
                "electrical_role": role,
                "terminal_side": "manual_ground_truth",
                "index": idx,
                "position": {"x": round(x, 2), "y": round(y, 2)},
                "search_radius_px": 10,
                "source": "manual_ground_truth",
                "identity_uncertain": False,
                "confidence": 1.0,
            })
        component["terminals"] = terminals
        component['review_required']=False
        component['review_reasons']=[]
        component["terminal_detection"] = {
            "status": "manual_ground_truth",
            "pin_count": len(terminals),
            "review_required": False,
            "review_reasons": [],
            "method": "user_terminal_correction",
        }
        if sample is not None:
            component["terminal_training"] = {
                "saved": True,
                "sample_id": sample["sample_id"],
                "keypoint_order": list(KEYPOINT_ORDER),
            }
        component["generic_terminal_training"] = {
            "saved": True,
            "sample_id": generic_sample["sample_id"],
            "pin_schema": [{"pin": pin, "role": role} for pin, role in schema],
        }
        _save_terminal_correction_record(project_id, component, points, schema)
        corrected_count += 1

    result["count"] = len(components)
    result["terminal_count"] = sum(len(c.get("terminals") or []) for c in components)
    result["review_count"] = sum(1 for c in components if (c.get("terminal_detection") or {}).get("review_required"))
    result["manual_terminal_corrections_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    result["manual_training_samples_added"] = len(training_samples)
    result["all_component_training_samples_saved"] = len(all_component_training_samples)
    result["schema_version"] = "stage7-terminal-learning-2.0"
    output_json = project_dir / "terminals.json"
    output_image = project_dir / "terminals.png"
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    salvar_imagem_terminais(source_path, result, output_image)
    _remover_artefatos_desatualizados(
        project_dir,
        ("connectivity.json", "connectivity.png", "circuit.json"),
    )
    atualizar_terminais(
        project_id=project_id,
        status="terminais_revisados",
        terminals_filename="terminals.png",
        terminals_info=json.dumps(result, ensure_ascii=False),
    )
    automatic_stage8 = _continuar_automaticamente_para_linhas(
        project_id,
        trigger="after_manual_terminal_correction",
    )
    from backend.terminal_review_draft import discard
    discard(project_dir,{str(e['component_id']) for e in edits})
    response = preparar_resposta_projeto(buscar_projeto(project_id))
    response["corrected_components"] = corrected_count
    response["training_samples_added"] = training_samples
    response["all_component_training_samples_saved"] = all_component_training_samples
    response["terminal_learning"] = terminal_learning_status()
    response["all_component_terminal_learning"] = generic_terminal_learning_status()
    response["automatic_stage8"] = automatic_stage8
    response["mensagem"] = (
        f"{corrected_count} componente(s) corrigido(s); {len(all_component_training_samples)} amostra(s) salva(s) para treino de todos os componentes. "
        "O terminals.png foi atualizado com os pontos manuais e as linhas foram recalculadas automaticamente."
    )
    return response


@app.get("/projects/{project_id}/terminal-trainer", response_class=HTMLResponse)
def abrir_treinador_de_terminais(project_id: str):
    """Editor e coleta de ground truth para todas as classes do dataset."""
    projeto = buscar_projeto(project_id)
    if projeto is None:
        raise HTTPException(status_code=404, detail="Projeto não encontrado.")
    result = _terminal_result_for_manual_edit(project_id, projeto)
    editable = []
    for comp in result.get("components") or []:
        ctype = str(comp.get("type") or "")
        schema = _schema_for_component(comp)
        existing = {
            str(t.get("pin")): t.get("position")
            for t in (comp.get("terminals") or [])
            if isinstance(t.get("position"), dict)
        }
        editable.append({
            "component_id": comp.get("component_id") or comp.get("detection_id"),
            "type": ctype,
            "bbox": comp.get("bbox"),
            "pins": [pin for pin, _role in schema],
            "pin_schema": [{"pin": pin, "role": role} for pin, role in schema],
            "fixed_schema": ctype in MANUAL_PIN_SCHEMAS,
            "points": existing,
            "review_required": bool((comp.get("terminal_detection") or {}).get("review_required")),
        })
    if not editable:
        raise HTTPException(status_code=409, detail="Nenhum componente foi encontrado para corrigir terminais.")
    image_url = f"/projects-files/{project_id}/{projeto['stored_filename']}"
    from backend.terminal_review_draft import restore
    restore(PROJECTS_DIR/project_id,PROJECTS_DIR/project_id/projeto['stored_filename'],editable,result.get('components') or [])
    payload = json.dumps({
        "project_id": project_id,
        "image_url": image_url,
        "components": editable,
        "status": generic_terminal_learning_status(),
        "transistor_status": terminal_learning_status(),
    }, ensure_ascii=False)
    html = """<!doctype html>
<html lang='pt-BR'><head><meta charset='utf-8'><title>Revisão de terminais</title>
<style>
body{font-family:Arial,sans-serif;margin:18px;background:#f5f5f5;color:#222}.bar{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin-bottom:12px}
button,select,input{padding:8px 10px;font-size:14px}button.active{outline:3px solid #333}button.primary{background:#1769aa;color:#fff;border:0;border-radius:5px}#wrap{overflow:auto;border:1px solid #aaa;background:white;max-height:70vh}canvas{display:block}#msg{margin-top:10px;font-weight:600}.hint{max-width:1150px;line-height:1.45}code{background:#eee;padding:2px 4px}.warn{color:#b00020;font-weight:700}.ok{color:#087a2f;font-weight:700}#dynamicPins{display:none}
</style></head><body>
<h2>Correção e treinamento de terminais — todos os componentes</h2>
<p class='hint'>Selecione o componente e depois o pino. Clique sobre o fio, dentro da caixa azul do componente. Toda correção — resistor, capacitor, diodo, LED, transistor, MOSFET, amplificador operacional e demais classes — fica persistida e vira uma amostra de treinamento.</p>
<div class='bar'><button id='prev'>Anterior</button><label>Componente <select id='component'></select></label><button id='next'>Próximo</button><button id='center'>Centralizar componente</button><span id='state'></span></div>
<div class='bar'><strong>Pino:</strong><span id='pins'></span><span id='dynamicPins'><input id='newPin' size='10' placeholder='nome do pino'><button id='addPin'>Adicionar pino</button><button id='removePin'>Remover pino selecionado</button></span></div>
<div class='bar'><button id='save' class='primary'>Salvar este componente</button><button id='saveAll' class='primary'>Salvar todos os alterados</button><button id='clear'>Limpar pontos deste componente</button><a id='preview' target='_blank'>Abrir terminals.png</a><a id='workflow'>Voltar ao fluxo</a></div>
<div id='wrap'><canvas id='canvas'></canvas></div><div id='msg'></div><p class='hint'>Dataset geral: <span id='dataset'></span>. A pasta <code>terminal_training_all_components</code> guarda os recortes e as anotações por classe. Corrigir novamente o mesmo componente atualiza a amostra, sem duplicá-la.</p>
<script id='payload' type='application/json'>__PAYLOAD__</script>
<script>
const cfg=JSON.parse(document.getElementById('payload').textContent);const projectId=cfg.project_id,comps=cfg.components,trainStatus=cfg.status;
const img=new Image();img.src=cfg.image_url;const canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d'),sel=document.getElementById('component'),msg=document.getElementById('msg'),pinsBox=document.getElementById('pins'),state=document.getElementById('state');let pin=null,current=0;
const palette=['#ff9800','#2196f3','#e91e63','#00a152','#7b1fa2','#d32f2f'];
comps.forEach((c,i)=>{c.dirty=!!c.draft_restored;const o=document.createElement('option');o.value=i;o.textContent=c.component_id+' — '+c.type+(c.review_required?' — REVISAR':'');sel.appendChild(o)});document.getElementById('dataset').textContent=trainStatus.sample_count+' amostra(s) em '+trainStatus.class_count+' classe(s)';document.getElementById('preview').href='/projects-files/'+projectId+'/terminals.png';document.getElementById('workflow').href='/projects/'+projectId+'/workflow';
function buildPins(){pinsBox.innerHTML='';const c=comps[current];if(!c.pins.length)pin=null;else if(!c.pins.includes(pin))pin=c.pins[0];c.pins.forEach((p,i)=>{const b=document.createElement('button');b.textContent=p;b.dataset.pin=p;b.onclick=()=>{pin=p;[...pinsBox.querySelectorAll('button')].forEach(x=>x.classList.toggle('active',x.dataset.pin===pin))};if(p===pin)b.classList.add('active');pinsBox.appendChild(b)});document.getElementById('dynamicPins').style.display=c.fixed_schema?'none':'inline-flex';state.className=c.review_required?'warn':'ok';state.textContent=(c.review_required?'REVISÃO PENDENTE':'automático disponível')+(c.dirty?' — ALTERADO':'')}
function draw(){if(!img.complete)return;ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(img,0,0);const c=comps[current],b=c.bbox;if(b){ctx.strokeStyle='#00a0ff';ctx.lineWidth=2;ctx.strokeRect(b.x1,b.y1,b.x2-b.x1,b.y2-b.y1)}ctx.save();if(b){ctx.beginPath();ctx.rect(b.x1,b.y1,b.x2-b.x1,b.y2-b.y1);ctx.clip()}c.pins.forEach((p,i)=>{const q=c.points[p];if(!q)return;ctx.beginPath();ctx.arc(q.x,q.y,7,0,Math.PI*2);ctx.fillStyle=palette[i%palette.length];ctx.fill();ctx.font='bold 18px Arial';ctx.fillText(p,q.x+9,q.y-7)});ctx.restore()}
function centerComponent(){const b=comps[current].bbox,wrap=document.getElementById('wrap');if(!b)return;wrap.scrollLeft=Math.max(0,(b.x1+b.x2)/2-wrap.clientWidth/2);wrap.scrollTop=Math.max(0,(b.y1+b.y2)/2-wrap.clientHeight/2)}
function show(i){current=(i+comps.length)%comps.length;sel.value=current;pin=comps[current].pins[0]||null;buildPins();draw();setTimeout(centerComponent,0)}
img.onload=()=>{canvas.width=img.naturalWidth;canvas.height=img.naturalHeight;show(0)};sel.onchange=()=>show(Number(sel.value));document.getElementById('prev').onclick=()=>show(current-1);document.getElementById('next').onclick=()=>show(current+1);document.getElementById('center').onclick=centerComponent;
canvas.addEventListener('click',e=>{if(!pin){msg.textContent='Selecione ou adicione um pino primeiro.';return}const r=canvas.getBoundingClientRect(),x=(e.clientX-r.left)*canvas.width/r.width,y=(e.clientY-r.top)*canvas.height/r.height;const b=comps[current].bbox;if(!b||x<=b.x1||x>=b.x2||y<=b.y1||y>=b.y2){msg.textContent='O terminal deve ficar dentro da caixa azul do componente selecionado.';return}comps[current].points[pin]={x,y};comps[current].dirty=true;msg.textContent='';buildPins();draw()});document.getElementById('clear').onclick=()=>{comps[current].points={};comps[current].dirty=true;buildPins();draw()};
document.getElementById('addPin').onclick=()=>{const c=comps[current],v=document.getElementById('newPin').value.trim();if(!v||c.pins.includes(v)){msg.textContent='Digite um nome de pino novo.';return}c.pins.push(v);c.pin_schema.push({pin:v,role:v});c.dirty=true;pin=v;document.getElementById('newPin').value='';buildPins();draw()};document.getElementById('removePin').onclick=()=>{const c=comps[current];if(!pin)return;c.pins=c.pins.filter(p=>p!==pin);c.pin_schema=c.pin_schema.filter(p=>p.pin!==pin);delete c.points[pin];c.dirty=true;pin=c.pins[0]||null;buildPins();draw()};
function editFor(c){if(!c.pins.length)throw new Error(c.component_id+': adicione ao menos um pino.');for(const p of c.pins)if(!c.points[p])throw new Error(c.component_id+': marque '+p+'.');return {component_id:c.component_id,points:c.points,pin_schema:c.pin_schema}}
async function saveList(list){msg.textContent='Salvando terminais e atualizando ligações…';try{const res=await fetch('/projects/'+projectId+'/terminals/corrections',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({components:list.map(editFor)})});const data=await res.json();if(!res.ok)throw new Error(data.detail||'Erro');for(const c of comps)if(list.includes(c)){c.review_required=false;c.dirty=false}buildPins();document.getElementById('preview').href='/projects-files/'+projectId+'/terminals.png?t='+Date.now();const st=data.all_component_terminal_learning||{};document.getElementById('dataset').textContent=(st.sample_count||0)+' amostra(s) em '+(st.class_count||0)+' classe(s)';msg.textContent=data.mensagem;return true}catch(err){msg.textContent='Erro: '+err.message;return false}}
document.getElementById('save').onclick=()=>saveList([comps[current]]);document.getElementById('saveAll').onclick=()=>{const changed=comps.filter(c=>c.dirty);if(!changed.length){msg.textContent='Nenhum componente foi alterado.';return}saveList(changed)};
</script></body></html>""".replace("__PAYLOAD__", payload.replace("</", "<\\/"))
    return HTMLResponse(content=html)


@app.get("/ocr/status")
def obter_status_ocr():
    return ocr_status()


@app.get("/component-models")
def obter_catalogo_modelos():
    return catalog_summary()


@app.post("/component-models/resolve")
def consultar_parametros_modelo(payload: dict):
    try:
        return resolve_parameters(payload.get('type'),payload.get('model',''),
                                  payload.get('electrical_parameters'),payload.get('parameter_policy'))
    except (ValueError,TypeError) as error:
        raise HTTPException(status_code=422,detail=str(error))


@app.get("/model-editor.js")
def abrir_editor_modelos_js():
    return FileResponse(BASE_DIR/'frontend/js/model_editor.js',media_type='application/javascript')

@app.get('/visual-editor.js')
def visual_editor_script():
    return FileResponse(BASE_DIR/'frontend/visual-editor.js',media_type='application/javascript')


@app.get("/projects/{project_id}/ocr-models-view",response_class=HTMLResponse)
def abrir_modelos_ocr(project_id: str):
    if buscar_projeto(project_id) is None:raise HTTPException(status_code=404,detail='Projeto não encontrado.')
    return model_ocr_page(project_id)


@app.post("/projects/{project_id}/detect")
@cached_project_stage('detect')
def reconhecer_componentes(
    project_id: str,
    confidence: float = 0.15,
    imgsz: int = 640,
    source: str = "original",
    strategy: str = "auto",
    adaptive: bool = True,
    force: bool = False,
):
    if confidence < 0.01 or confidence > 0.99:
        raise HTTPException(
            status_code=400,
            detail="confidence deve ficar entre 0.01 e 0.99.",
        )

    if imgsz < 320 or imgsz > 1280:
        raise HTTPException(
            status_code=400,
            detail="imgsz deve ficar entre 320 e 1280.",
        )

    source = source.strip().lower()
    if source not in {"original", "processed"}:
        raise HTTPException(
            status_code=400,
            detail="source deve ser original ou processed.",
        )

    strategy = strategy.strip().lower()
    if strategy not in {"auto", "color", "grayscale"}:
        raise HTTPException(
            status_code=400,
            detail="strategy deve ser auto, color ou grayscale.",
        )

    projeto = buscar_projeto(project_id)

    if projeto is None:
        raise HTTPException(
            status_code=404,
            detail="Projeto não encontrado.",
        )

    project_dir = PROJECTS_DIR / project_id

    # O modelo foi validado com a imagem original. O processamento binário
    # continua disponível para inspeção, mas não é mais escolhido sem o
    # usuário pedir explicitamente source=processed.
    if source == "processed":
        processed_filename = projeto.get("processed_filename")
        if not processed_filename:
            raise HTTPException(
                status_code=409,
                detail="Este projeto ainda não possui imagem processada.",
            )
        source_path = project_dir / processed_filename
    else:
        source_path = project_dir / projeto["stored_filename"]

    output_image = project_dir / "detections.png"
    output_json = project_dir / "detections.json"

    try:
        resultado = detectar_componentes(
            image_path=source_path,
            output_image_path=output_image,
            output_json_path=output_json,
            confidence=confidence,
            imgsz=imgsz,
            strategy=strategy,
            adaptive=adaptive,
        )

        resultado["requested_source"] = source
        output_json.write_text(
            json.dumps(resultado, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        atualizar_deteccoes(
            project_id=project_id,
            status="reconhecido",
            detection_filename="detections.png",
            detections_info=json.dumps(
                resultado,
                ensure_ascii=False,
            ),
        )

        _remover_artefatos_desatualizados(
            project_dir,
            ("ocr.png", "components.json", "terminals.png", "terminals.json", "wires.png", "wires.json", "wires_mask.png", "wires_skeleton.png", "wires_erased_mask.png", "transistor_wire_refinement.png"),
        )

    except FileNotFoundError as erro:
        raise HTTPException(
            status_code=409,
            detail=str(erro),
        )

    except Exception as erro:
        raise HTTPException(
            status_code=500,
            detail=f"Erro durante o reconhecimento YOLO: {erro}",
        )

    terminal_resultado = None
    terminal_warning = None
    if resultado.get("detections") and not DEFER_INITIAL_GEOMETRY.get():
        try:
            terminal_resultado = _gerar_terminais_automaticamente(
                project_id=project_id,
                projeto=buscar_projeto(project_id),
                components_info=_components_info_from_detections(resultado),
                trigger="automatic_detection",
            )
        except Exception as erro:
            terminal_warning = str(erro)

    projeto_atualizado = buscar_projeto(project_id)
    resposta = preparar_resposta_projeto(projeto_atualizado)
    resposta["geometry_deferred_until_ocr"] = DEFER_INITIAL_GEOMETRY.get()
    resposta["terminals_generated_automatically"] = terminal_resultado is not None
    resposta["automatic_stage8"] = (terminal_resultado or {}).get("automatic_stage8")
    resposta["terminal_preview_url"] = resposta.get("terminals_url")
    if terminal_warning:
        resposta["terminal_detection_warning"] = terminal_warning
    resposta["mensagem"] = (
        f"Reconhecimento concluído: {resultado['count']} componente(s) detectado(s). "
        + (
            f"Terminais gerados automaticamente: {terminal_resultado['terminal_count']} ponto(s); "
            "a detecção de linhas também foi executada."
            if terminal_resultado
            else "Fios e terminais serão calculados após a leitura do OCR." if DEFER_INITIAL_GEOMETRY.get()
            else "Não foi possível gerar os terminais provisórios; veja terminal_detection_warning."
        )
    )
    return resposta


@app.put("/projects/{project_id}/detections")
@time_project_operation('detection_review')
def salvar_reconhecimento_corrigido(project_id: str, payload: dict):
    projeto = buscar_projeto(project_id)
    if projeto is None:
        raise HTTPException(status_code=404, detail="Projeto não encontrado.")

    detections_info = _carregar_json_texto(projeto.get("detections_info"))
    if not detections_info:
        raise HTTPException(
            status_code=409,
            detail="Execute o reconhecimento antes de fazer correções.",
        )

    recebidas = payload.get("detections")
    if not isinstance(recebidas, list) or len(recebidas) > 200:
        raise HTTPException(
            status_code=400,
            detail="detections deve ser uma lista com até 200 componentes.",
        )

    classes = load_real_classes()
    classes_por_id = {int(item["id"]): item for item in classes}
    largura = float(detections_info.get("image_width") or 0)
    altura = float(detections_info.get("image_height") or 0)
    corrigidas = []

    for item in recebidas:
        try:
            class_id = int(item["class_id"])
            classe = classes_por_id[class_id]
            caixa = item["bbox"]
            x1 = max(0.0, float(caixa["x1"]))
            y1 = max(0.0, float(caixa["y1"]))
            x2 = min(largura, float(caixa["x2"])) if largura else float(caixa["x2"])
            y2 = min(altura, float(caixa["y2"])) if altura else float(caixa["y2"])
        except (KeyError, TypeError, ValueError):
            raise HTTPException(
                status_code=400,
                detail="Existe uma classe ou caixa inválida na correção.",
            )

        if x2 - x1 < 5 or y2 - y1 < 5:
            raise HTTPException(
                status_code=400,
                detail="Toda caixa precisa ter pelo menos 5 × 5 pixels.",
            )

        corrigidas.append(
            {
                "class_id": class_id,
                "class": str(classe["key"]),
                "confidence": round(float(item.get("confidence", 1.0)), 6),
                "bbox": {
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                    "width": round(x2 - x1, 2),
                    "height": round(y2 - y1, 2),
                    "center_x": round((x1 + x2) / 2, 2),
                    "center_y": round((y1 + y2) / 2, 2),
                },
                "inference_sources": item.get("inference_sources", ["manual_review"]),
                "alternatives": [],
                "manual_reviewed": True,
                "review_required": False,
                "review_reasons": [],
            }
        )

    try:
        validate_no_nested_boxes(corrigidas)
    except ValueError as erro:
        raise HTTPException(status_code=400, detail=str(erro))

    corrigidas.sort(key=lambda item: (item["bbox"]["y1"], item["bbox"]["x1"]))
    def box_signature(items):
        return sorted((item.get('class'),tuple(round(float(item['bbox'][k]),2) for k in ('x1','y1','x2','y2'))) for item in items)
    if box_signature(corrigidas)==box_signature(detections_info.get('detections') or []):
        # Continuing an unchanged review must preserve OCR and manual contacts.
        resposta=preparar_resposta_projeto(projeto)
        resposta.update(processing_reused=True,ocr_invalidated_by_detection_edit=False,
                        geometry_deferred_until_ocr=payload.get('defer_geometry') is True,
                        mensagem='Reconhecimento salvo. Suas leituras e correções foram preservadas.')
        return resposta
    for indice, item in enumerate(corrigidas, start=1):
        item["detection_id"] = f"DET_{indice:03d}"

    detections_info["detections"] = corrigidas
    detections_info["count"] = len(corrigidas)
    detections_info["review_count"] = 0
    detections_info["manual_reviewed_at"] = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )

    project_dir = PROJECTS_DIR / project_id
    source_path = project_dir / projeto["stored_filename"]
    output_json = project_dir / "detections.json"
    output_image = project_dir / "detections.png"
    output_json.write_text(
        json.dumps(detections_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    salvar_imagem_deteccoes(source_path, corrigidas, output_image)
    atualizar_deteccoes(
        project_id=project_id,
        status="reconhecido",
        detection_filename="detections.png",
        detections_info=json.dumps(detections_info, ensure_ascii=False),
    )

    # A edição da lista/caixa/classe invalida OCR e terminais anteriores.
    # Apague os artefatos velhos e gere imediatamente novos terminais usando
    # as detecções corrigidas. Assim componentes adicionados pelo usuário
    # aparecem no terminals.png sem depender de um arquivo antigo.
    _remover_artefatos_desatualizados(
        project_dir,
        ("ocr.png", "components.json", "terminals.png", "terminals.json", "wires.png", "wires.json", "wires_mask.png", "wires_skeleton.png", "wires_erased_mask.png", "transistor_wire_refinement.png"),
    )
    terminal_resultado = None
    terminal_warning = None
    defer_geometry = payload.get('defer_geometry') is True
    if corrigidas and not defer_geometry:
        try:
            terminal_resultado = _gerar_terminais_automaticamente(
                project_id=project_id,
                projeto=projeto,
                components_info=_components_info_from_detections(detections_info),
                trigger="manual_detection_edit",
            )
        except Exception as erro:
            terminal_warning = str(erro)

    projeto_atualizado = buscar_projeto(project_id)
    resposta = preparar_resposta_projeto(projeto_atualizado)
    resposta["terminals_generated_automatically"] = terminal_resultado is not None
    resposta["automatic_stage8"] = (terminal_resultado or {}).get("automatic_stage8")
    resposta["terminal_preview_url"] = resposta.get("terminals_url")
    resposta["ocr_invalidated_by_detection_edit"] = True
    resposta["geometry_deferred_until_ocr"] = defer_geometry
    if terminal_warning:
        resposta["terminal_detection_warning"] = terminal_warning
    resposta["mensagem"] = (
        f"Reconhecimento corrigido: {len(corrigidas)} componente(s). "
        + (
            f"Terminais refeitos automaticamente: {terminal_resultado['terminal_count']} ponto(s) "
            f"em {terminal_resultado['count']} componente(s), seguidos pela detecção de linhas. "
            if terminal_resultado
            else "Fios e terminais serão atualizados após a leitura dos textos. " if defer_geometry
            else "Os terminais não puderam ser refeitos; veja terminal_detection_warning. "
        )
        + "Como a lista de componentes mudou, execute o OCR novamente para atualizar referências e valores."
    )
    return resposta


@app.post("/projects/{project_id}/ocr")
@cached_project_stage('ocr')
def ler_referencias_e_valores(project_id: str, force: bool = False):
    projeto = buscar_projeto(project_id)
    if projeto is None:
        raise HTTPException(status_code=404, detail="Projeto não encontrado.")

    detections_info = _carregar_json_texto(projeto.get("detections_info"))
    if not detections_info or not detections_info.get("detections"):
        raise HTTPException(
            status_code=409,
            detail="Execute o reconhecimento de componentes antes do OCR.",
        )

    project_dir = PROJECTS_DIR / project_id
    source_path = project_dir / projeto["stored_filename"]
    output_json = project_dir / "components.json"
    output_image = project_dir / "ocr.png"

    try:
        resultado = executar_ocr_componentes(
            image_path=source_path,
            detections_info=detections_info,
            output_json_path=output_json,
            output_image_path=output_image,
        )
        resultado = preserve_manual_models(resultado, _carregar_json_texto(projeto.get('ocr_info')))
        output_json.write_text(json.dumps(resultado,ensure_ascii=False,indent=2),encoding='utf8')
        deteccoes_filtradas = resultado.get("retained_detections", [])
        detections_info["detections"] = deteccoes_filtradas
        detections_info["count"] = len(deteccoes_filtradas)
        detections_info["review_count"] = sum(
            1 for item in deteccoes_filtradas if item.get("review_required")
        )
        detections_info["text_false_positive_count"] = resultado.get(
            "text_false_positive_count", 0
        )
        detections_info["text_false_positives"] = resultado.get(
            "text_false_positives", []
        )
        detections_info["text_overlap_review_count"] = resultado.get(
            "text_overlap_review_count", 0
        )
        detections_info["text_overlap_reviews"] = resultado.get(
            "text_overlap_reviews", []
        )
        detections_info["bbox_adjustment_count"] = resultado.get(
            "bbox_adjustment_count", 0
        )
        detections_info["bbox_adjustments"] = resultado.get(
            "bbox_adjustments", []
        )
        (project_dir / "detections.json").write_text(
            json.dumps(detections_info, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        salvar_imagem_deteccoes(
            source_path,
            deteccoes_filtradas,
            project_dir / "detections.png",
        )
        atualizar_deteccoes(
            project_id=project_id,
            status="reconhecido",
            detection_filename="detections.png",
            detections_info=json.dumps(detections_info, ensure_ascii=False),
        )
        atualizar_ocr(
            project_id=project_id,
            status="ocr_concluido",
            ocr_filename="ocr.png",
            ocr_info=json.dumps(resultado, ensure_ascii=False),
        )
    except FileNotFoundError as erro:
        raise HTTPException(status_code=503, detail=str(erro))
    except ValueError as erro:
        raise HTTPException(status_code=409, detail=str(erro))
    except Exception as erro:
        raise HTTPException(
            status_code=500,
            detail=f"Erro durante a leitura OCR: {erro}",
        )

    terminal_resultado = None
    terminal_warning = None
    try:
        # Etapa 7 é executada automaticamente assim que OCR/componentes termina.
        terminal_resultado = _gerar_terminais_automaticamente(
            project_id=project_id,
            projeto=buscar_projeto(project_id),
            components_info=resultado,
            trigger="ocr",
        )
    except Exception as erro:
        # Não apaga um OCR válido caso somente a visualização dos terminais falhe.
        terminal_warning = str(erro)

    projeto_atualizado = buscar_projeto(project_id)
    resposta = preparar_resposta_projeto(projeto_atualizado)
    resposta["terminals_generated_automatically"] = terminal_resultado is not None
    resposta["automatic_stage8"] = (terminal_resultado or {}).get("automatic_stage8")
    resposta["terminal_preview_url"] = resposta.get("terminals_url")
    if terminal_warning:
        resposta["terminal_detection_warning"] = terminal_warning
    resposta["mensagem"] = (
        f"OCR concluído em {resultado.get('elapsed_seconds', 0):.1f}s: "
        f"{resultado['reference_ocr_count']} referência(s) "
        f"{resultado['value_ocr_count']} valor(es) e {resultado.get('model_ocr_count',0)} modelo(s) lidos; "
        f"{resultado.get('text_false_positive_count', 0)} caixa(s) de texto "
        "removida(s); "
        f"{resultado.get('text_overlap_review_count', 0)} caixa(s) "
        "marcada(s) para revisão; "
        f"{resultado.get('bbox_adjustment_count', 0)} caixa(s) "
        "reduzida(s) para não incluir texto. "
        + (
            f"Etapa 7 executada automaticamente: {terminal_resultado['terminal_count']} "
            f"terminal(is) em {terminal_resultado['count']} componente(s), "
            "seguidos automaticamente pela detecção de linhas."
            if terminal_resultado
            else "A detecção automática de terminais não foi concluída; veja terminal_detection_warning."
        )
    )
    return resposta


@app.put("/projects/{project_id}/ocr/components")
@time_project_operation('electrical_review')
def salvar_componentes_ocr(project_id: str, payload: dict):
    projeto = buscar_projeto(project_id)
    if projeto is None:
        raise HTTPException(status_code=404, detail="Projeto não encontrado.")

    resultado = _carregar_json_texto(projeto.get("ocr_info"))
    detections_info = _carregar_json_texto(projeto.get("detections_info"))
    if not resultado or not detections_info:
        raise HTTPException(status_code=409, detail="Execute o OCR antes de editar.")

    edicoes = payload.get("components")
    if not isinstance(edicoes, list):
        raise HTTPException(status_code=400, detail="components deve ser uma lista.")

    project_dir = PROJECTS_DIR / project_id
    source_path = project_dir / projeto["stored_filename"]
    output_json = project_dir / "components.json"
    output_image = project_dir / "ocr.png"
    amostras_adicionadas = 0
    aviso_biblioteca = None

    try:
        resultado_corrigido = aplicar_correcoes_manuais(
            resultado=resultado, edicoes=edicoes, detections_info=detections_info,
        )
        anteriores = {c['detection_id']:c for c in resultado.get('components',[])}
        somente_modelos = len(resultado_corrigido.get('components',[]))==len(anteriores) and all(
            c['type']==anteriores[c['detection_id']]['type']
            and c.get('bbox')==anteriores[c['detection_id']].get('bbox')
            for c in resultado_corrigido.get('components',[])
        )
        resultado = resultado_corrigido
        output_json.write_text(
            json.dumps(resultado, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        salvar_imagem_componentes(source_path, resultado, output_image)
        atualizar_ocr(
            project_id=project_id,
            status="ocr_revisado",
            ocr_filename="ocr.png",
            ocr_info=json.dumps(resultado, ensure_ascii=False),
        )
    except ValueError as erro:
        raise HTTPException(status_code=400, detail=str(erro))
    except Exception as erro:
        raise HTTPException(
            status_code=500,
            detail=f"Não foi possível salvar as correções: {erro}",
        )

    terminal_resultado = None
    terminal_warning = None
    try:
        if somente_modelos:
            _sincronizar_modelos_terminais(project_id,projeto,resultado,previous_components={'components':list(anteriores.values())})
        else:
            terminal_resultado = _gerar_terminais_automaticamente(
                project_id=project_id, projeto=projeto, components_info=resultado,
                trigger="ocr_manual_edit",
            )
    except Exception as erro:
        terminal_warning = str(erro)

    projeto_atualizado = buscar_projeto(project_id)
    resposta = preparar_resposta_projeto(projeto_atualizado)
    resposta["terminals_generated_automatically"] = terminal_resultado is not None
    resposta["automatic_stage8"] = (terminal_resultado or {}).get("automatic_stage8")
    resposta["terminal_preview_url"] = resposta.get("terminals_url")
    if terminal_warning:
        resposta["terminal_detection_warning"] = terminal_warning
    resposta["ocr_library_samples_added"] = amostras_adicionadas
    if aviso_biblioteca:
        resposta["ocr_library_warning"] = aviso_biblioteca
    resposta["mensagem"] = (
        "Referências, valores, modelos e parâmetros salvos. "
        + (
            "Valores e modelos registrados; posições dos fios e terminais preservadas."
            if somente_modelos and not terminal_warning else
            f"Terminais recalculados automaticamente: {terminal_resultado['terminal_count']} ponto(s), "
            "seguidos pela atualização das linhas."
            if terminal_resultado
            else "Os terminais não puderam ser recalculados; veja terminal_detection_warning."
        )
    )
    return resposta


def _sincronizar_modelos_terminais(project_id,projeto,resultado,previous_components=None):
    folder=PROJECTS_DIR/project_id
    path=folder/'terminals.json'
    saved=_carregar_json_texto(path.read_text(encoding='utf8')) if path.exists() else _carregar_json_texto(projeto.get('terminals_info'))
    models={c['detection_id']:c for c in resultado.get('components',[])}
    valid_cache=cache_valid(folder,'geometry',geometry_key(BASE_DIR,folder,projeto,previous_components or resultado),GEOMETRY_OUTPUTS)
    renames={}
    if saved:
        for component in saved.get('components',[]):
            original=models.get(component.get('detection_id'))
            if original and original.get('type')==component.get('type'):
                old_id=component.get('component_id');new_id=original.get('component_id')
                if old_id and new_id and old_id!=new_id:
                    renames[old_id]=new_id;component['component_id']=new_id
                    for terminal in component.get('terminals',[]):
                        terminal['pin_id']=new_id+'.'+terminal['pin']
                for field in ('reference','value','model','electrical_parameters','parameter_policy','parameter_pending'):
                    if field in original:component[field]=copy.deepcopy(original[field])
        path.write_text(json.dumps(saved,ensure_ascii=False,indent=2),encoding='utf8')
        atualizar_terminais(project_id=project_id,status='ocr_revisado',terminals_filename='terminals.png',terminals_info=json.dumps(saved,ensure_ascii=False))
    if renames:
        # Identity changes do not alter image geometry. Translate references in
        # one pass so swaps such as R1 -> R2 and R2 -> R1 remain unambiguous.
        def renamed(value):
            if isinstance(value,dict):return {renamed(key):renamed(item) for key,item in value.items()}
            if isinstance(value,list):return [renamed(item) for item in value]
            if isinstance(value,str):
                if value in renames:return renames[value]
                for separator in (':','.'):
                    prefix,mark,suffix=value.partition(separator)
                    if mark and prefix in renames:return renames[prefix]+mark+suffix
            return value
        wires=None
        for name in ('wires.json','junctions.json','terminal_corrections.json'):
            data=_json_projeto(folder,name)
            if data is None:continue
            data=renamed(data)
            if name=='wires.json':
                data.pop('stage10',None);data.pop('stage11',None);wires=data
            (folder/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
        if wires is not None:
            atualizar_fios(project_id=project_id,status='ocr_revisado',wires_filename=projeto.get('wires_filename') or 'wires.png',wires_info=json.dumps(wires,ensure_ascii=False))
    _remover_artefatos_desatualizados(folder,('connectivity.json','connectivity.png','circuit.json'))
    if saved and (valid_cache or renames):
        salvar_imagem_terminais(folder/str(projeto['stored_filename']),saved,folder/'terminals.png')
    if valid_cache:
        cache_store(folder,'geometry',geometry_key(BASE_DIR,folder,projeto,resultado),GEOMETRY_OUTPUTS)


@app.post("/projects/{project_id}/terminals")
def detectar_terminais(project_id: str):
    projeto = buscar_projeto(project_id)
    if projeto is None:
        raise HTTPException(status_code=404, detail="Projeto não encontrado.")

    components_info = _carregar_json_texto(projeto.get("ocr_info"))
    trigger = "manual_endpoint_ocr"
    if not components_info or not components_info.get("components"):
        detections_info = _carregar_json_texto(projeto.get("detections_info"))
        components_info = _components_info_from_detections(detections_info or {})
        trigger = "manual_endpoint_detections"
    if not components_info.get("components"):
        raise HTTPException(
            status_code=409,
            detail="Execute o reconhecimento de componentes antes da detecção de terminais.",
        )

    try:
        resultado = _gerar_terminais_automaticamente(
            project_id=project_id,
            projeto=projeto,
            components_info=components_info,
            trigger=trigger,
        )
    except (FileNotFoundError, ValueError) as erro:
        raise HTTPException(status_code=409, detail=str(erro))
    except Exception as erro:
        raise HTTPException(
            status_code=500,
            detail=f"Erro durante a detecção de terminais: {erro}",
        )

    projeto_atualizado = buscar_projeto(project_id)
    resposta = preparar_resposta_projeto(projeto_atualizado)
    resposta["automatic_stage8"] = resultado.get("automatic_stage8")
    resposta["terminal_preview_url"] = resposta.get("terminals_url")
    resposta["mensagem"] = (
        f"Etapa 7 concluída: {resultado['terminal_count']} terminal(is) "
        f"estimado(s) em {resultado['count']} componente(s); "
        f"{resultado['review_count']} componente(s) requer(em) confirmação de orientação/polaridade. "
        "Os fios foram detectados antes dos terminais; todos os pontos ficam dentro das caixas."
    )
    return resposta




@symbol_analysis_session
def _executar_etapa8_fios(
    project_id: str,
    projeto: dict,
    *,
    source: str = "original",
    mask_components: bool = True,
    mask_text: bool = True,
    min_segment_length: float = 8.0,
    components_override: dict | None = None,
    generation_trigger: str = "wires_endpoint",
):
    project_dir = PROJECTS_DIR / project_id
    source = str(source or "original").strip().lower()
    if source not in {"original", "processed"}:
        raise ValueError("source deve ser original ou processed.")
    if source == "processed":
        filename = projeto.get("processed_filename")
        if not filename:
            raise ValueError("Este projeto ainda não possui imagem processada.")
    else:
        filename = projeto.get("stored_filename")
    source_path = project_dir / str(filename)
    image = cv2.imread(str(source_path))
    if image is None:
        raise FileNotFoundError(f"Não foi possível abrir a imagem {source_path}.")

    terminals_info = _carregar_json_texto(projeto.get("terminals_info"))
    ocr_info = _carregar_json_texto(projeto.get("ocr_info"))
    detections_info = _carregar_json_texto(projeto.get("detections_info"))
    components_info = copy.deepcopy(components_override or _latest_components_for_project(projeto))
    if not components_info.get("components") and terminals_info:
        components_info = copy.deepcopy(terminals_info)
    if not components_info.get("components"):
        raise ValueError("Execute o reconhecimento de componentes antes de detectar os fios.")
    # Restore the approved 5.13.6 masking/filtering context. These internal
    # anchors protect ink during component erasure; final pins are published
    # only after the immutable wire mask/graph have been computed.
    from backend.terminal_detection import detectar_terminais_componentes
    from backend.processing_progress import report
    report('Analisando os símbolos e as ligações…')
    wire_components_info = detectar_terminais_componentes(
        components_info,image.shape[1],image.shape[0],image,
        legacy_wire_context=True)
    # Rebuild the same legacy anchors on every run: final automatic contact
    # coordinates must never change tomorrow's wire mask. Explicit manual
    # ground truth remains available to the legacy component masking logic.
    old_components={c.get("component_id"):c for c in (terminals_info or {}).get("components",[])}
    for component in wire_components_info["components"]:
        old=old_components.get(component.get("component_id"))
        manual=bool(old and (old.get("terminal_detection") or {}).get("status")=="manual_ground_truth"
                    and old.get("bbox")==component.get("bbox"))
        if manual:
            component["terminals"]=copy.deepcopy(old.get("terminals") or [])
        elif component.get("type") in {"transistor_npn","transistor_pnp"}:
            component["terminals"] = []

    # Color must come from the source photo, never a processed monochrome view
    # or a preview with colored detection boxes. Visual certainty comes from
    # the detections, independently of pending OCR/terminal review.
    color_reference_image = image
    if source == "processed":
        color_reference_image = cv2.imread(str(project_dir / str(projeto.get("stored_filename"))))
        if color_reference_image is None:
            raise FileNotFoundError("Não foi possível abrir a imagem original para comparar a tinta.")

    from backend.processing_progress import report
    report('Identificando os fios do circuito…')
    result, wire_mask, skeleton, erased_mask = detectar_fios(
        image,
        components_info=wire_components_info,
        ocr_info=ocr_info,
        source_image=str(filename),
        mask_components=bool(mask_components),
        mask_text=bool(mask_text),
        min_segment_length=float(min_segment_length),
        color_reference_image=color_reference_image,
        color_components_info=detections_info or ocr_info or components_info,
    )
    result["project_id"] = project_id
    result["generated_from"] = (
        "ocr_components" if ocr_info else "detections" if detections_info else "components"
    )
    result["terminal_context_available"] = True
    result["wire_detection_version"] = "5.13.6"
    result["pipeline_order"] = ["components", "wires", "terminals", "junctions"]
    result["ocr_context_available"] = bool(ocr_info)

    output_json = project_dir / "wires.json"
    output_overlay = project_dir / "wires.png"
    output_mask = project_dir / "wires_mask.png"
    output_skeleton = project_dir / "wires_skeleton.png"
    output_erased = project_dir / "wires_erased_mask.png"
    output_refinement = project_dir / "transistor_wire_refinement.png"
    output_junctions_json = project_dir / "junctions.json"
    output_junctions_preview = project_dir / "junctions.png"

    # Persist the legacy wire result before final terminal placement. Terminal
    # refinement never feeds back into this mask, skeleton or graph.
    output_json.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    salvar_mascara(wire_mask,output_mask,invert_for_view=True)
    salvar_mascara(skeleton,output_skeleton,invert_for_view=True)
    salvar_mascara(erased_mask,output_erased,invert_for_view=False)
    _remover_artefatos_desatualizados(project_dir,('terminals.json','terminals.png','junctions.json','junctions.png'))
    report('Posicionando os terminais sobre os fios…')
    refined_terminals = executar_deteccao_terminais(
        image_path=project_dir / str(projeto["stored_filename"]),
        components_info=components_info,
        output_json_path=project_dir / "terminals.json",
        output_image_path=project_dir / "terminals.png",
        wires_info=result,skeleton=skeleton,save_preview=False)
    _apply_saved_terminal_corrections(project_id,refined_terminals)
    # Preserve manual edits even when they have not yet entered the correction
    # registry (e.g. older project versions), but never trust stale automatic pins.
    old_by_id={c.get("component_id"):c for c in (terminals_info or {}).get("components",[])}
    for component in refined_terminals.get("components") or []:
        old=old_by_id.get(component.get("component_id"))
        if old and (old.get("terminal_detection") or {}).get("status")=="manual_ground_truth":
            component["terminals"]=copy.deepcopy(old.get("terminals") or [])
            component["terminal_detection"]=copy.deepcopy(old["terminal_detection"])
    from backend.terminal_wire_contacts import constrain_terminal_positions
    constrain_terminal_positions(refined_terminals,color_reference_image,require_ink=True)
    refined_terminals.update(generation_trigger=generation_trigger,
        auto_generated_after_ocr=generation_trigger.startswith("ocr"),
        generated_from="detections" if all(c.get("provisional_from_detection") for c in components_info["components"]) else "ocr_components",
        source_component_count=len(components_info["components"]))
    refinement_summary=refined_terminals.get("stage8_transistor_refinement")
    result["transistor_terminal_refinement"]=refinement_summary
    result["transistor_wire_refinement"]=refinement_summary
    result["terminal_detection_after_wires"]=refined_terminals.get("wire_terminal_contacts")
    (project_dir / "terminals.json").write_text(json.dumps(refined_terminals,ensure_ascii=False,indent=2),encoding="utf-8")
    salvar_imagem_terminais(project_dir / str(projeto["stored_filename"]),refined_terminals,project_dir / "terminals.png")
    save_transistor_wire_refinement_preview(color_reference_image,refined_terminals,result,output_refinement)
    atualizar_terminais_pos_fios(project_id=project_id,status="terminais_detectados_apos_fios",
        terminals_filename="terminals.png",terminals_info=json.dumps(refined_terminals,ensure_ascii=False))

    # Etapa 9 automática: transforma os candidatos topológicos da Etapa 8 em
    # decisões elétricas conservadoras. T conecta; cruzamento de quatro vias
    # só conecta quando existe ponto; ambiguidades permanecem para revisão.
    stage9_warning = None
    try:
        report('Conferindo as junções do circuito…')
        stage9 = resolver_juncoes(
            color_reference_image if (result.get("ink_color_profile") or {}).get("active") else image,
            result,
            terminals_info=refined_terminals or terminals_info,
        )
        result["stage9"] = stage9
        output_junctions_json.write_text(
            json.dumps(stage9, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        salvar_imagem_juncoes(image, stage9, output_junctions_preview)
        try:
            _gerar_resultados_eletricos(project_id, projeto, result, refined_terminals or terminals_info, color_reference_image)
        except Exception as erro:
            result['stage10_11_warning']=str(erro)
    except Exception as erro:
        stage9_warning = str(erro)
        result["stage9_warning"] = stage9_warning

    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    salvar_imagem_fios(image, result, output_overlay)

    final_status = "fios_detectados"
    if refinement_summary and refinement_summary.get("transistor_count", 0):
        final_status = "fios_detectados_transistores_refinados"
    if result.get("stage9"):
        final_status = (
            "juncoes_resolvidas"
            if not result["stage9"].get("review_count")
            else "juncoes_aguardando_revisao"
        )
    atualizar_fios(
        project_id=project_id,
        status=final_status,
        wires_filename="wires.png",
        wires_info=json.dumps(result, ensure_ascii=False),
    )
    return result


@app.post("/projects/{project_id}/wires")
def detectar_fios_projeto(
    project_id: str,
    source: str = "original",
    mask_components: bool = True,
    mask_text: bool = True,
    min_segment_length: float = 8.0,
):
    projeto = buscar_projeto(project_id)
    if projeto is None:
        raise HTTPException(status_code=404, detail="Projeto não encontrado.")
    if min_segment_length < 3 or min_segment_length > 100:
        raise HTTPException(status_code=400, detail="min_segment_length deve ficar entre 3 e 100 pixels.")
    try:
        result = _executar_etapa8_fios(
            project_id,
            projeto,
            source=source,
            mask_components=mask_components,
            mask_text=mask_text,
            min_segment_length=min_segment_length,
        )
    except (ValueError, FileNotFoundError) as erro:
        raise HTTPException(status_code=409, detail=str(erro))
    except Exception as erro:
        raise HTTPException(status_code=500, detail=f"Erro durante a detecção de fios: {erro}")

    response = preparar_resposta_projeto(buscar_projeto(project_id))
    response["wire_preview_url"] = response.get("wires_url")
    tr = result.get("transistor_terminal_refinement") or {}
    response["transistor_wire_refinement_url"] = (
        f"http://127.0.0.1:8010/projects-files/{project_id}/transistor_wire_refinement.png"
        if tr.get("transistor_count", 0) else None
    )
    response["mensagem"] = (
        f"Etapa 8 concluída: {result['segment_count']} segmento(s) de fio, "
        f"{result['endpoint_candidate_count']} endpoint(s) candidato(s) e "
        f"{result['junction_candidate_count']} nó(s) candidato(s). "
        f"Transistores refinados por linhas: {tr.get('refined_count', 0)}/{tr.get('transistor_count', 0)}. "
        f"Etapa 9 automática: {(result.get('stage9') or {}).get('connected_junction_count', 0)} junção(ões) conectada(s), "
        f"{(result.get('stage9') or {}).get('unconnected_crossing_count', 0)} cruzamento(s) separado(s) e "
        f"{(result.get('stage9') or {}).get('review_count', 0)} revisão(ões). "
        "Confira a imagem de junções e corrija somente os casos necessários."
    )
    return response


@app.get("/projects/{project_id}/stage8-status")
def obter_status_etapa8(project_id: str):
    projeto = buscar_projeto(project_id)
    if projeto is None:
        raise HTTPException(status_code=404, detail="Projeto não encontrado.")
    wires = _carregar_json_texto(projeto.get("wires_info"))
    generated = bool(wires and (PROJECTS_DIR / project_id / "wires.json").exists())
    base_url = f"http://127.0.0.1:8010/projects-files/{project_id}"
    return {
        "project_id": project_id,
        "stage": 8,
        "generated": generated,
        "criterion_met": bool(generated and wires.get("segment_count", 0) > 0) if wires else False,
        "segment_count": int(wires.get("segment_count", 0)) if wires else 0,
        "node_count": int(wires.get("node_count", 0)) if wires else 0,
        "endpoint_candidate_count": int(wires.get("endpoint_candidate_count", 0)) if wires else 0,
        "junction_candidate_count": int(wires.get("junction_candidate_count", 0)) if wires else 0,
        "wires_url": f"{base_url}/wires.png" if generated else None,
        "wires_json_url": f"{base_url}/wires.json" if generated else None,
        "wires_mask_url": f"{base_url}/wires_mask.png" if generated else None,
        "wires_skeleton_url": f"{base_url}/wires_skeleton.png" if generated else None,
        "transistor_wire_refinement_url": (f"{base_url}/transistor_wire_refinement.png" if generated and (PROJECTS_DIR / project_id / "transistor_wire_refinement.png").exists() else None),
        "transistor_terminal_refinement": (wires.get("transistor_terminal_refinement") if wires else None),
        "stage9_generated": bool(wires and wires.get("stage9")),
        "next_step": (
            "Revisar as junções da Etapa 9"
            if generated and wires and (wires.get("stage9") or {}).get("review_count", 0)
            else "Etapa 10 - consolidar conectividade"
            if generated and wires and wires.get("stage9")
            else "Execute POST /projects/{project_id}/wires"
        ),
    }


def _executar_etapa9_juncoes(project_id: str, projeto: dict):
    project_dir = PROJECTS_DIR / project_id
    wires = _json_projeto(project_dir, "wires.json") or _carregar_json_texto(projeto.get("wires_info"))
    if not wires or not wires.get("segments"):
        raise ValueError("Os fios ainda não estão disponíveis. Volte à revisão de Terminais para prepará-los.")
    source_path = project_dir / str(projeto.get("stored_filename"))
    image = cv2.imread(str(source_path))
    if image is None:
        raise FileNotFoundError(f"Não foi possível abrir a imagem {source_path}.")
    stage9 = resolver_juncoes(
        image,
        wires,
        terminals_info=_carregar_json_texto(projeto.get("terminals_info")),
    )
    wires["stage9"] = stage9
    (project_dir / "junctions.json").write_text(
        json.dumps(stage9, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    salvar_imagem_juncoes(image, stage9, project_dir / "junctions.png")
    _remover_artefatos_desatualizados(project_dir,('connectivity.json','connectivity.png','circuit.json'))
    try:
        _gerar_resultados_eletricos(project_id, projeto, wires, _carregar_json_texto(projeto.get('terminals_info')), image)
    except Exception as erro:
        wires['stage10_11_warning']=str(erro)
    (project_dir / "wires.json").write_text(
        json.dumps(wires, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    atualizar_fios(
        project_id=project_id,
        status=("juncoes_resolvidas" if not stage9.get("review_count") else "juncoes_aguardando_revisao"),
        wires_filename=projeto.get("wires_filename") or "wires.png",
        wires_info=json.dumps(wires, ensure_ascii=False),
    )
    return stage9


@app.post("/projects/{project_id}/junctions")
def detectar_juncoes_projeto(project_id: str):
    projeto = buscar_projeto(project_id)
    if projeto is None:
        raise HTTPException(status_code=404, detail="Projeto não encontrado.")
    try:
        stage9 = _executar_etapa9_juncoes(project_id, projeto)
    except (ValueError, FileNotFoundError) as erro:
        raise HTTPException(status_code=409, detail=str(erro))
    except Exception as erro:
        raise HTTPException(status_code=500, detail=f"Não foi possível preparar as junções: {erro}")
    response = preparar_resposta_projeto(buscar_projeto(project_id))
    response["mensagem"] = (
        f"Junções preparadas: {stage9['connected_junction_count']} junção(ões) conectada(s), "
        f"{stage9['unconnected_crossing_count']} cruzamento(s) sem conexão e "
        f"{stage9['review_count']} caso(s) para revisão."
    )
    return response


def _preparar_revisao_juncoes(project_id: str, *, refresh_preview=False):
    """Reuse current decisions and repair only missing review prerequisites."""
    from backend.performance import project_lock
    folder=PROJECTS_DIR/project_id
    with project_lock(folder):
        projeto=buscar_projeto(project_id)
        if projeto is None:raise HTTPException(404,'Projeto não encontrado.')
        wires=_json_projeto(folder,'wires.json') or _carregar_json_texto(projeto.get('wires_info')) or {}
        terms=_json_projeto(folder,'terminals.json') or _carregar_json_texto(projeto.get('terminals_info')) or {}
        stage9=wires.get('stage9')
        if not stage9:
            current={**projeto,'wires_info':json.dumps(wires,ensure_ascii=False),'terminals_info':json.dumps(terms,ensure_ascii=False)}
            try:stage9=_executar_etapa9_juncoes(project_id,current)
            except (ValueError,FileNotFoundError) as e:raise HTTPException(409,str(e))
            except Exception as e:raise HTTPException(500,'Não foi possível preparar as junções: '+str(e)) from e
            wires=_json_projeto(folder,'wires.json') or {**wires,'stage9':stage9}
        preview=folder/'junctions.png'
        if refresh_preview or not preview.is_file() or not preview.stat().st_size:
            image=cv2.imread(str(folder/str(projeto.get('stored_filename'))))
            if image is None:raise HTTPException(409,'A foto do circuito não foi encontrada. Confira a imagem enviada ao projeto.')
            salvar_imagem_juncoes(image,stage9,preview)
        return projeto,wires,stage9,terms


@app.post('/projects/{project_id}/junctions/prepare')
def preparar_revisao_juncoes_projeto(project_id: str,payload:dict|None=None):
    _,_,stage9,_=_preparar_revisao_juncoes(project_id,refresh_preview=(payload or {}).get('refresh_preview') is True)
    preview=PROJECTS_DIR/project_id/'junctions.png'
    return dict(project_id=project_id,junctions_info=stage9,
                junctions_url=f'/projects-files/{project_id}/junctions.png',preview_version=str(preview.stat().st_mtime_ns))


@app.post("/projects/{project_id}/junctions/click-analysis")
def analisar_clique_juncao_projeto(project_id: str, payload: dict):
    projeto,wires,stage9,terms = _preparar_revisao_juncoes(project_id)
    try:
        analysis = analisar_clique_juncao(
            stage9,
            float(payload.get("x")),
            float(payload.get("y")),
            terminals_info=terms,
            snap_radius_px=payload.get("snap_radius_px"),
        )
    except (TypeError, ValueError) as erro:
        raise HTTPException(status_code=400, detail=f"Clique inválido: {erro}")
    analysis.pop("_selected", None)
    return analysis


@app.put("/projects/{project_id}/junctions/corrections")
def corrigir_juncoes_projeto(project_id: str, payload: dict):
    projeto,wires,stage9,terms = _preparar_revisao_juncoes(project_id)
    try:
        corrected = aplicar_correcoes_juncoes(
            stage9,
            payload.get("corrections") or [],
            terminals_info=terms,
        )
    except ValueError as erro:
        raise HTTPException(status_code=400, detail=str(erro))
    project_dir = PROJECTS_DIR / project_id
    image = cv2.imread(str(project_dir / str(projeto.get("stored_filename"))))
    if image is None:
        raise HTTPException(status_code=409, detail="Imagem original não encontrada.")
    wires["stage9"] = corrected
    (project_dir / "junctions.json").write_text(
        json.dumps(corrected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    salvar_imagem_juncoes(image, corrected, project_dir / "junctions.png")
    _remover_artefatos_desatualizados(project_dir,('connectivity.json','connectivity.png','circuit.json'))
    try:
        _gerar_resultados_eletricos(project_id, projeto, wires, _carregar_json_texto(projeto.get('terminals_info')), image)
    except Exception as erro:
        wires['stage10_11_warning']=str(erro)
    (project_dir / "wires.json").write_text(
        json.dumps(wires, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    atualizar_fios(
        project_id=project_id,
        status=("juncoes_corrigidas" if not corrected.get("review_count") else "juncoes_aguardando_revisao"),
        wires_filename=projeto.get("wires_filename") or "wires.png",
        wires_info=json.dumps(wires, ensure_ascii=False),
    )
    response = preparar_resposta_projeto(buscar_projeto(project_id))
    response["mensagem"] = (
        f"Correções salvas. Restam {corrected['review_count']} junção(ões) para revisar."
    )
    return response


@app.get("/projects/{project_id}/stage9-status")
def obter_status_etapa9(project_id: str):
    projeto = buscar_projeto(project_id)
    if projeto is None:
        raise HTTPException(status_code=404, detail="Projeto não encontrado.")
    stage9 = (_carregar_json_texto(projeto.get("wires_info")) or {}).get("stage9")
    return {
        "project_id": project_id,
        "stage": 9,
        "generated": bool(stage9),
        "connected_junction_count": int((stage9 or {}).get("connected_junction_count", 0)),
        "unconnected_crossing_count": int((stage9 or {}).get("unconnected_crossing_count", 0)),
        "review_count": int((stage9 or {}).get("review_count", 0)),
        "ready_for_stage10": bool((stage9 or {}).get("ready_for_stage10", False)),
        "next_step": (
            "Etapa 10 - consolidar conectividade"
            if stage9 and stage9.get("ready_for_stage10")
            else "Revise as junções pendentes"
            if stage9
            else "Execute POST /projects/{project_id}/junctions"
        ),
    }

@app.get("/real-dataset/classes")
def obter_classes_reais():
    return load_real_classes()


def _json_projeto(project_dir: Path, filename: str):
    from backend.performance import read_document
    return read_document(project_dir/filename)


def _resultados_eletricos_atuais(projeto, wires=None, terminals=None):
    project_dir=PROJECTS_DIR/projeto['project_id']
    connectivity=_json_projeto(project_dir,'connectivity.json')
    if not connectivity:return None,None
    saved_wires=_json_projeto(project_dir,'wires.json')
    saved_terminals=_json_projeto(project_dir,'terminals.json')
    if not saved_wires or not saved_terminals:return None,None
    try:
        signature=source_signature(saved_wires,saved_terminals)
        if connectivity.get('source_signature')!=signature:return None,None
        if wires and terminals and source_signature(wires,terminals)!=signature:return None,None
    except (ValueError,TypeError):return None,None
    circuit=_json_projeto(project_dir,'circuit.json')
    if circuit and (circuit.get('source_connectivity_signature')!=signature or not validar_circuito(circuit)['valid']):circuit=None
    return connectivity,circuit


def _gerar_resultados_eletricos(project_id, projeto, wires, terminals, image, *, include_circuit=True):
    project_dir=PROJECTS_DIR/project_id
    wires.pop('stage10',None);wires.pop('stage11',None);wires.pop('stage10_11_warning',None);wires.pop('stage12',None)
    _remover_artefatos_desatualizados(project_dir,('connectivity.json','connectivity.png','circuit.json'))
    connectivity=construir_conectividade(image,wires,terminals or {})
    connectivity['project_id']=project_id
    (project_dir/'connectivity.json').write_text(json.dumps(connectivity,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    salvar_imagem_redes(image,connectivity,wires,project_dir/'connectivity.png')
    wires['stage10']={key:connectivity[key] for key in ('schema_version','source_signature','net_count','terminal_count','attached_terminal_count','unconnected_terminal_count','ambiguous_terminal_count','review_count','criterion_met')}
    if include_circuit:
        circuit=gerar_circuito(connectivity,project_id=project_id,name=projeto.get('name'))
        (project_dir/'circuit.json').write_text(json.dumps(circuit,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
        wires['stage11']=dict(schema_version=circuit['schema_version'],generated=True,criterion_met=True,
                              source_signature=connectivity['source_signature'],validation=circuit['validation'])
    wires['electrical_pipeline_order']=['components','wires','terminals','junctions','connectivity','circuit']
    return connectivity


def _salvar_estado_eletrico(project_id, projeto, wires):
    (PROJECTS_DIR/project_id/'wires.json').write_text(json.dumps(wires,ensure_ascii=False,indent=2),encoding='utf8')
    atualizar_fios(project_id=project_id,status='circuito_json_gerado' if wires.get('stage11') else 'conectividade_construida',
                   wires_filename=projeto.get('wires_filename') or 'wires.png',wires_info=json.dumps(wires,ensure_ascii=False))


@app.post('/projects/{project_id}/connectivity')
def construir_redes_projeto(project_id: str):
    projeto=buscar_projeto(project_id)
    if projeto is None:raise HTTPException(status_code=404,detail='Projeto nao encontrado.')
    project_dir=PROJECTS_DIR/project_id
    wires=_json_projeto(project_dir,'wires.json');terminals=_json_projeto(project_dir,'terminals.json')
    if not wires or not terminals:raise HTTPException(status_code=409,detail='Execute as etapas de fios e terminais antes de construir redes.')
    image=cv2.imread(str(project_dir/projeto['stored_filename']))
    try:
        result=_gerar_resultados_eletricos(project_id,projeto,wires,terminals,image,include_circuit=False)
        _salvar_estado_eletrico(project_id,projeto,wires)
    except ValueError as erro:raise HTTPException(status_code=409,detail=str(erro))
    response=preparar_resposta_projeto(buscar_projeto(project_id))
    response['mensagem']=f"Etapa 10: {result['net_count']} redes, {result['attached_terminal_count']}/{result['terminal_count']} pinos associados."
    return response


@app.post('/projects/{project_id}/circuit')
def gerar_circuito_projeto(project_id: str):
    projeto=buscar_projeto(project_id)
    if projeto is None:raise HTTPException(status_code=404,detail='Projeto nao encontrado.')
    connectivity,_=_resultados_eletricos_atuais(projeto)
    if not connectivity:raise HTTPException(status_code=409,detail='Construa ou atualize as redes da etapa 10 antes de gerar circuit.json.')
    _,existing=_resultados_eletricos_atuais(projeto)
    if existing and existing.get('review'):
        raise HTTPException(status_code=409,detail='Este circuito possui revisao humana. Abra Revisar circuito para editar; reconstruir as redes arquiva a revisao e gera uma nova versao automatica.')
    try:
        circuit=gerar_circuito(connectivity,project_id=project_id,name=projeto.get('name'))
    except ValueError as erro:raise HTTPException(status_code=409,detail=str(erro))
    project_dir=PROJECTS_DIR/project_id
    (project_dir/'circuit.json').write_text(json.dumps(circuit,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    wires=_json_projeto(project_dir,'wires.json')
    wires['stage11']=dict(schema_version=circuit['schema_version'],generated=True,criterion_met=True,
                         source_signature=connectivity['source_signature'],validation=circuit['validation'])
    _salvar_estado_eletrico(project_id,projeto,wires)
    response=preparar_resposta_projeto(buscar_projeto(project_id))
    response['mensagem']='Etapa 11: circuit.json valido e independente da imagem. Pendencias registradas para a revisao.'
    return response


@app.get('/projects/{project_id}/circuit')
def obter_circuito_projeto(project_id: str):
    projeto=buscar_projeto(project_id)
    if projeto is None:raise HTTPException(status_code=404,detail='Projeto nao encontrado.')
    _,circuit=_resultados_eletricos_atuais(projeto)
    if not circuit:raise HTTPException(status_code=409,detail='Circuito ainda nao gerado ou desatualizado; atualize as redes e o JSON.')
    return circuit


@app.get('/projects/{project_id}/circuit/download')
def baixar_circuito_projeto(project_id: str):
    obter_circuito_projeto(project_id)
    return FileResponse(PROJECTS_DIR/project_id/'circuit.json',media_type='application/json',filename='circuit.json')


@app.get('/projects/{project_id}/stage10-status')
def obter_status_etapa10(project_id: str):
    projeto=buscar_projeto(project_id)
    if projeto is None:raise HTTPException(status_code=404,detail='Projeto nao encontrado.')
    connectivity,_=_resultados_eletricos_atuais(projeto)
    return dict(project_id=project_id,stage=10,generated=bool(connectivity),criterion_met=bool((connectivity or {}).get('criterion_met')),
                net_count=(connectivity or {}).get('net_count',0),attached_terminal_count=(connectivity or {}).get('attached_terminal_count',0),
                unconnected_terminal_count=(connectivity or {}).get('unconnected_terminal_count',0),review_count=(connectivity or {}).get('review_count',0),
                next_step='Etapa 11 - gerar circuit.json' if connectivity else 'Construir as redes da etapa 10')


@app.get('/projects/{project_id}/stage11-status')
def obter_status_etapa11(project_id: str):
    projeto=buscar_projeto(project_id)
    if projeto is None:raise HTTPException(status_code=404,detail='Projeto nao encontrado.')
    _,circuit=_resultados_eletricos_atuais(projeto)
    return dict(project_id=project_id,stage=11,generated=bool(circuit),criterion_met=bool(circuit),
                validation=(circuit or {}).get('validation'),next_step='Etapa 12 - revisao humana do circuito' if circuit else 'Gerar circuit.json')


def _circuito_para_revisao(project_id):
    projeto=buscar_projeto(project_id)
    if projeto is None:raise HTTPException(status_code=404,detail='Projeto nao encontrado.')
    _,circuit=_resultados_eletricos_atuais(projeto)
    if not circuit:raise HTTPException(status_code=409,detail='Gere um circuit.json atualizado na etapa 11 antes de revisar.')
    from backend.contact_repair import repair_isolated_contacts
    folder=PROJECTS_DIR/project_id
    with _circuit_review_lock:
        connectivity,current=_resultados_eletricos_atuais(projeto)
        if not current:raise HTTPException(status_code=409,detail='O circuito mudou. Atualize a tela.')
        source=folder/projeto['stored_filename']
        # Photo-free reviewed circuits stay usable. Only evidence-backed,
        # isolated contacts can change an existing net; the previous circuit
        # is retained in the ordinary revision history before saving.
        image=(lambda:cv2.imread(str(source))) if source.is_file() else None
        repaired,changes=repair_isolated_contacts(current,connectivity,_json_projeto(folder,'wires.json') or {},image)
        if changes:_salvar_revisao(project_id,projeto,repaired)
        circuit=repaired
    return projeto,circuit


@app.get('/projects/{project_id}/circuit-review')
def obter_revisao_circuito(project_id: str):
    projeto,circuit=_circuito_para_revisao(project_id)
    displayed=review_component_models(circuit)
    return dict(circuit=displayed,revision=revision(circuit),issues=review_issues(displayed),original_filename=projeto.get('stored_filename'))


@app.post('/projects/{project_id}/circuit-review/preview')
def previa_netlist_revisao(project_id: str,payload: dict):
    with _circuit_review_lock:
        _,current=_circuito_para_revisao(project_id)
        if payload.get('base_revision')!=revision(current):
            raise HTTPException(409,'O circuito mudou. Recarregue a revisão antes de atualizar a prévia.')
        try:
            draft=apply_review(current,payload)
            preview=copy.deepcopy(draft);preview.update(status='confirmed',confirmed_by_user=True)
            from backend.connection_preview import connection_preview
            netlist=connection_preview(preview)
        except (ValueError,TypeError) as error:
            return dict(ready=False,message=str(error),issues=review_issues(draft) if 'draft' in locals() else [],netlist=None)
    return dict(ready=True,preview=True,netlist=netlist['netlist'].replace('Circuit Scanner - confirmed circuit','Circuit Scanner - preview (not confirmed)',1),
                nodes=netlist['nodes'],components=netlist['components'],warnings=netlist['warnings'],circuit=draft)


def _salvar_revisao(project_id,projeto,circuit):
    folder=PROJECTS_DIR/project_id
    _remover_artefatos_desatualizados(folder,OLD_RESULT_ARTIFACTS+KICAD_ARTIFACTS+EXPORT_ARTIFACTS)
    previous=folder/'circuit.json'
    if previous.exists():
        history=folder/'circuit_review_history';history.mkdir(exist_ok=True)
        shutil.copy2(previous,history/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:8]+'.json'))
    temporary=folder/('circuit_'+uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(circuit,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    temporary.replace(folder/'circuit.json')
    wires=_json_projeto(folder,'wires.json') or {}
    if wires.get('stage11'):wires['stage11']['validation']=circuit['validation']
    wires['stage12']=dict(reviewed=True,confirmed_by_user=circuit['confirmed_by_user'],revision=revision(circuit))
    _salvar_estado_eletrico(project_id,projeto,wires)


def _aplicar_revisao_projeto(project_id,payload,confirm=False):
    with _circuit_review_lock:
        projeto,current=_circuito_para_revisao(project_id)
        if payload.get('base_revision')!=revision(current):
            raise HTTPException(status_code=409,detail='O circuito mudou desde que a tela foi aberta. Recarregue a versao salva antes de editar novamente.')
        try:
            result=confirm_review(current,payload.get('acknowledge_warnings')) if confirm else apply_review(current,payload)
        except (ValueError,TypeError) as erro:raise HTTPException(status_code=422,detail=str(erro))
        _salvar_revisao(project_id,projeto,result)
        if confirm:
            # Confirmation changes review metadata, not the authored circuit.
            # Keep its native parameters and layout current for the next edit/export.
            folder=PROJECTS_DIR/project_id
            for name in ('falstad_edit.json','visual_geometry.json'):
                saved=_json_projeto(folder,name)
                if saved and saved.get('source_revision')==revision(current):
                    saved['source_revision']=revision(result)
                    temporary=folder/(name+'.tmp')
                    temporary.write_text(json.dumps(saved,ensure_ascii=False,indent=2),encoding='utf8')
                    temporary.replace(folder/name)
        return dict(circuit=result,revision=revision(result),issues=review_issues(result))


@app.post('/projects/{project_id}/circuit-review')
def salvar_revisao_circuito(project_id: str,payload: dict):
    return _aplicar_revisao_projeto(project_id,payload)


@app.post('/projects/{project_id}/circuit-review/confirm')
def confirmar_revisao_circuito(project_id: str,payload: dict):
    return _aplicar_revisao_projeto(project_id,payload,confirm=True)


@app.get('/projects/{project_id}/circuit-review-view',response_class=HTMLResponse)
def tela_revisao_circuito(project_id: str):
    if buscar_projeto(project_id) is None:raise HTTPException(status_code=404,detail='Projeto nao encontrado.')
    from backend.simulation_visual import simulation_page
    return HTMLResponse(simulation_page(project_id))


@app.get('/projects/{project_id}/stage12-status')
def obter_status_etapa12(project_id: str):
    projeto,circuit=_circuito_para_revisao(project_id)
    return dict(project_id=project_id,stage=12,review_saved=bool(circuit.get('review')),criterion_met=circuit['confirmed_by_user'],
                confirmed_by_user=circuit['confirmed_by_user'],issues=review_issues(circuit),next_step='Simular no Falstad' if circuit['confirmed_by_user'] else 'Revisar e confirmar o circuito')


@app.get('/projects/{project_id}/circuit-view',response_class=HTMLResponse)
def abrir_circuito_projeto(project_id: str):
    if buscar_projeto(project_id) is None:raise HTTPException(status_code=404,detail='Projeto nao encontrado.')
    return HTMLResponse(content=circuit_page(project_id))


def _circuito_confirmado(project_id):
    project,circuit=_circuito_para_revisao(project_id)
    if circuit.get('confirmed_by_user') is not True or circuit.get('status')!='confirmed':
        raise HTTPException(409,'Confira e confirme o circuito antes de simular.')
    return project,circuit

@app.get('/simulation/status')
def status_simulador():
    return dict(ready=(BASE_DIR/'backend/vendor/circuitjs/circuitjs.html').is_file(),engine='Falstad/CircuitJS')

@app.get('/projects/{project_id}/simulation-view',response_class=HTMLResponse)
def abrir_simulacao_projeto(project_id:str):
    if buscar_projeto(project_id) is None:raise HTTPException(404,'Projeto não encontrado.')
    from backend.simulation_visual import simulation_page
    return simulation_page(project_id)

@app.get("/component-trainer", response_class=HTMLResponse)
def abrir_treinador_componentes():
    """Editor visual de caixas e painel de treinamento, sem usar /docs."""
    return HTMLResponse(content=component_trainer_html())


@app.get("/component-learning/status")
def obter_status_treinamento_componentes():
    return component_training_status()


@app.post("/component-learning/train")
def iniciar_treinamento_componentes(
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 8,
    patience: int = 15,
    device: str = "cpu",
):
    try:
        return start_component_training(
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            patience=patience,
            device=device,
        )
    except ValueError as erro:
        raise HTTPException(status_code=400, detail=str(erro))
    except RuntimeError as erro:
        raise HTTPException(status_code=409, detail=str(erro))
    except Exception as erro:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao iniciar treinamento de componentes: {erro}",
        )


@app.post("/component-learning/resume")
def retomar_treinamento_componentes():
    try:
        return resume_component_training()
    except ValueError as erro:
        raise HTTPException(status_code=400, detail=str(erro))
    except RuntimeError as erro:
        raise HTTPException(status_code=409, detail=str(erro))
    except Exception as erro:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao retomar treinamento de componentes: {erro}",
        )


@app.post("/component-learning/stop-and-validate")
def finalizar_treinamento_componentes_e_validar():
    try:
        return request_component_training_stop()
    except ValueError as erro:
        raise HTTPException(status_code=400, detail=str(erro))
    except RuntimeError as erro:
        raise HTTPException(status_code=409, detail=str(erro))
    except Exception as erro:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao solicitar parada do treinamento: {erro}",
        )


@app.post("/component-learning/rollback")
def restaurar_modelo_componentes_anterior():
    try:
        return rollback_component_model()
    except ValueError as erro:
        raise HTTPException(status_code=400, detail=str(erro))
    except RuntimeError as erro:
        raise HTTPException(status_code=409, detail=str(erro))
    except Exception as erro:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao restaurar modelo anterior: {erro}",
        )


@app.post("/component-learning/foundation-from-active")
def fixar_modelo_ativo_como_base_estavel():
    try:
        return set_foundation_from_active_model()
    except ValueError as erro:
        raise HTTPException(status_code=400, detail=str(erro))
    except RuntimeError as erro:
        raise HTTPException(status_code=409, detail=str(erro))
    except Exception as erro:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao fixar base estável: {erro}",
        )


@app.get("/component-learning/dataset-audit")
def auditar_anotacoes_componentes():
    try:
        return audit_real_annotations()
    except Exception as erro:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao auditar anotações: {erro}",
        )


@app.post("/component-learning/diagnostic")
def iniciar_diagnostico_componentes(
    max_images: int = 60,
    confidence: float = 0.25,
    imgsz: int = 640,
):
    try:
        return start_component_diagnostic(
            max_images=max_images,
            confidence=confidence,
            imgsz=imgsz,
        )
    except ValueError as erro:
        raise HTTPException(status_code=400, detail=str(erro))
    except RuntimeError as erro:
        raise HTTPException(status_code=409, detail=str(erro))
    except Exception as erro:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao iniciar diagnóstico: {erro}",
        )


@app.get("/component-learning/diagnostic/status")
def obter_status_diagnostico_componentes():
    return component_diagnostic_status()


@app.get("/real-dataset/info")
def obter_real_dataset_info():
    return real_dataset_info()


@app.get("/real-dataset/images")
def obter_imagens_reais():
    return list_real_images()


@app.get("/real-dataset/images/{image_id}")
def obter_imagem_real(image_id: str):
    record = get_real_image(image_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Imagem real não encontrada.")
    return record


@app.post("/real-dataset/upload")
async def enviar_imagem_real(
    file: UploadFile = File(...),
    name: str = Form(""),
    source_type: str = Form("handwritten"),
    source_group: str = Form(""),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nenhum arquivo foi enviado.")
    try:
        return upload_real_image(
            file.file,
            file.filename,
            name,
            source_type=source_type,
            source_group=source_group,
        )
    except ValueError as erro:
        raise HTTPException(status_code=400, detail=str(erro))
    except Exception as erro:
        raise HTTPException(status_code=500, detail=f"Erro ao salvar imagem real: {erro}")


@app.put("/real-dataset/images/{image_id}/annotations")
def salvar_anotacoes_reais(image_id: str, payload: dict):
    try:
        boxes = payload.get("boxes", [])
        return save_annotations(image_id, boxes)
    except FileNotFoundError as erro:
        raise HTTPException(status_code=404, detail=str(erro))
    except ValueError as erro:
        raise HTTPException(status_code=400, detail=str(erro))
    except Exception as erro:
        raise HTTPException(status_code=500, detail=f"Erro ao salvar anotações: {erro}")


@app.post("/real-dataset/build-mixed")
def gerar_dataset_misto(
    real_train_augments: int = 2,
    seed: int = 4411,
    text_negative_count: int = 450,
):
    try:
        manifest = build_mixed_dataset(
            real_train_augments=real_train_augments,
            seed=seed,
            text_negative_count=text_negative_count,
        )
        return {
            "mensagem": "Dataset misto gerado com sucesso.",
            "manifest": manifest,
        }
    except ValueError as erro:
        raise HTTPException(status_code=400, detail=str(erro))
    except Exception as erro:
        raise HTTPException(status_code=500, detail=f"Erro ao gerar dataset misto: {erro}")


from backend.production_api import register_production
import sys as _production_sys
register_production(app, _production_sys.modules[__name__])
_circuitjs_root=Path(__file__).parent/'vendor'/'circuitjs'
if (_circuitjs_root/'circuitjs.html').is_file():
    app.mount('/simulators/circuitjs', StaticFiles(directory=_circuitjs_root), name='circuitjs')

from backend.guided_workflow import register_guided
from backend.guided_page import ADAPTERS
ADAPTERS['junctions']=ADAPTERS['junctions'].replace('save:async()=>{','save:async()=>{if(!reviewReady)throw Error("Aguarde a imagem das junções carregar antes de continuar.");')
ADAPTERS['circuit']="window.GuidedStep={save:async()=>{await window.GuidedReady;return await window.confirmVisualCircuit()},saveBeforeLeave:async()=>window.hasNativeEdits()?await window.saveNativeEdits():true,isDirty:()=>window.hasNativeEdits()};guidedReady(window.GuidedReady);"
ADAPTERS['simulation']=ADAPTERS['circuit']
_guided_page = register_guided(app, _production_sys.modules[__name__])

@app.get('/projects/{project_id}/workflow', response_class=HTMLResponse)
def abrir_fluxo_guiado(project_id: str):
    if buscar_projeto(project_id) is None:
        raise HTTPException(404, 'Projeto não encontrado.')
    return _guided_page(project_id)


@app.get('/projects/{project_id}/performance')
def obter_tempos_projeto(project_id: str):
    if buscar_projeto(project_id) is None:raise HTTPException(404,'Projeto não encontrado.')
    return read_performance_json(PROJECTS_DIR/project_id/'performance.json') or {'stages':{},'recent_runs':[]}
