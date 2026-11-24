"""Treinamento seguro do detector de componentes YOLO.

O processo roda em thread para a interface continuar responsiva, usa o modelo
ativo como ponto de partida e somente o substitui depois que ``best.pt`` foi
produzido e validado. O peso anterior sempre fica em ``models/backups``.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import csv
import hashlib
import json
import shutil
import threading
import traceback

from backend.real_dataset import (
    MIXED_DATASET_DIR,
    REAL_EVAL_DIR,
    active_dataset_dirs,
    load_classes,
)


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "circuit_scanner_yolo.pt"
FOUNDATION_MODEL_PATH = MODELS_DIR / "circuit_scanner_yolo_foundation.pt"
BACKUP_DIR = MODELS_DIR / "backups"
RUNS_DIR = MODELS_DIR / "training" / "component_recognition"
STATUS_PATH = MODELS_DIR / "component_training_status.json"
SUMMARY_PATH = MODELS_DIR / "training_summary_mixed.json"

_LOCK = threading.Lock()
_THREAD: threading.Thread | None = None
_STOP_REQUESTED = threading.Event()


def _now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _last_epoch(run_dir: Path):
    results_path = run_dir / "results.csv"
    if not results_path.exists():
        return None
    try:
        with results_path.open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            return None
        row = {str(k).strip(): v for k, v in rows[-1].items()}
        epoch = int(float(row.get("epoch", len(rows) - 1))) + 1
        metrics = {}
        for key, value in row.items():
            if key == "epoch" or value in (None, ""):
                continue
            try:
                metrics[key] = round(float(value), 6)
            except (TypeError, ValueError):
                continue
        return {"epoch": epoch, "metrics": metrics}
    except Exception:
        return None


def component_training_status():
    status = _read_json(
        STATUS_PATH,
        {
            "state": "idle",
            "message": "Nenhum treinamento de componentes foi iniciado.",
        },
    )
    with _LOCK:
        thread_alive = bool(_THREAD and _THREAD.is_alive())
    if status.get("state") == "running" and not thread_alive:
        status = dict(status)
        status["state"] = "interrupted"
        status["message"] = (
            "O processo foi interrompido (provavelmente por reinício do servidor). "
            "O modelo anterior continua preservado."
        )
    run_dir = Path(status.get("run_dir", "")) if status.get("run_dir") else None
    last_checkpoint = run_dir / "weights" / "last.pt" if run_dir else None
    progress = _last_epoch(run_dir) if run_dir else None
    if progress:
        status["current_epoch"] = progress["epoch"]
        status["progress_percent"] = round(
            100.0 * progress["epoch"] / max(1, int(status.get("epochs", 1))), 1
        )
        status["latest_metrics"] = progress["metrics"]
    active_mixed, active_real = active_dataset_dirs()
    status["model_exists"] = MODEL_PATH.exists()
    status["foundation_model_exists"] = FOUNDATION_MODEL_PATH.exists()
    status["dataset_ready"] = (active_mixed / "dataset.yaml").exists()
    status["active_mixed_dataset"] = str(active_mixed.relative_to(BASE_DIR))
    status["active_real_evaluation"] = str(active_real.relative_to(BASE_DIR))
    status["summary"] = _read_json(SUMMARY_PATH)
    status["resume_available"] = bool(
        status.get("state") in {"interrupted", "failed"}
        and last_checkpoint
        and last_checkpoint.exists()
    )
    status["resume_checkpoint"] = (
        str(last_checkpoint) if status["resume_available"] else None
    )
    return status


def _metric_value(metrics, path: str):
    current = metrics
    for part in path.split("."):
        current = getattr(current, part, None)
        if current is None:
            return None
    try:
        return round(float(current), 6)
    except (TypeError, ValueError):
        return None


def _validation_summary(result):
    summary = {
        "map50_95": _metric_value(result, "box.map"),
        "map50": _metric_value(result, "box.map50"),
        "map75": _metric_value(result, "box.map75"),
        "precision": _metric_value(result, "box.mp"),
        "recall": _metric_value(result, "box.mr"),
    }
    names = getattr(result, "names", {}) or {}
    if isinstance(names, (list, tuple)):
        names = {index: name for index, name in enumerate(names)}
    maps = getattr(getattr(result, "box", None), "maps", None)
    per_class = {}
    if maps is not None:
        for class_id, value in enumerate(maps):
            try:
                per_class[str(names.get(class_id, class_id))] = round(
                    float(value), 6
                )
            except (TypeError, ValueError):
                continue
    summary["map50_95_by_class"] = per_class
    return summary


def _negative_false_positive_summary(model, dataset_dir: Path, imgsz: int):
    """Mede caixas inventadas em imagens cuja anotação é vazia."""
    image_dir = dataset_dir / "images" / "val"
    label_dir = dataset_dir / "labels" / "val"
    negative_images = []
    for image_path in sorted(image_dir.glob("*")):
        if not image_path.is_file():
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.exists() and not label_path.read_text(encoding="utf-8").strip():
            negative_images.append(str(image_path))
        if len(negative_images) >= 250:
            break
    if not negative_images:
        return {"image_count": 0, "false_boxes": 0, "false_boxes_per_image": None}
    results = model.predict(
        source=negative_images,
        conf=0.25,
        imgsz=int(imgsz),
        verbose=False,
        stream=False,
    )
    false_boxes = 0
    for result in results:
        boxes = getattr(result, "boxes", None)
        false_boxes += len(boxes) if boxes is not None else 0
    return {
        "image_count": len(negative_images),
        "false_boxes": int(false_boxes),
        "false_boxes_per_image": round(false_boxes / len(negative_images), 6),
    }


def _candidate_quality_gate(candidate_metrics, baseline_metrics, candidate_negatives, baseline_negatives):
    """Impede que um treino com precisão pior substitua o modelo ativo."""
    reasons = []
    cand_precision = candidate_metrics.get("precision")
    cand_map50 = candidate_metrics.get("map50")
    cand_recall = candidate_metrics.get("recall")
    base_precision = baseline_metrics.get("precision")
    base_map50 = baseline_metrics.get("map50")

    if cand_precision is None or cand_precision < 0.60:
        reasons.append("candidate_precision_below_0.60")
    if cand_map50 is None or cand_map50 < 0.45:
        reasons.append("candidate_map50_below_0.45")
    if cand_recall is None or cand_recall < 0.35:
        reasons.append("candidate_recall_below_0.35")
    if (
        cand_precision is not None and base_precision is not None
        and cand_precision + 0.04 < base_precision
    ):
        reasons.append("precision_regressed_more_than_0.04")
    if (
        cand_map50 is not None and base_map50 is not None
        and cand_map50 + 0.05 < base_map50
    ):
        reasons.append("map50_regressed_more_than_0.05")

    # Uma média boa pode esconder o colapso de uma classe rara. Comparamos a
    # AP50-95 por classe e bloqueamos regressões materiais, especialmente em
    # transistores, MOSFETs e amplificador operacional.
    candidate_by_class = candidate_metrics.get("map50_95_by_class") or {}
    baseline_by_class = baseline_metrics.get("map50_95_by_class") or {}
    collapsed_classes = []
    for class_name, candidate_ap in candidate_by_class.items():
        baseline_ap = baseline_by_class.get(class_name)
        if baseline_ap is None:
            continue
        if float(candidate_ap) < 0.20 and float(candidate_ap) + 0.08 < float(baseline_ap):
            collapsed_classes.append(str(class_name))
    if collapsed_classes:
        reasons.append(
            "per_class_ap_collapsed:" + ",".join(sorted(collapsed_classes))
        )

    cand_fp = candidate_negatives.get("false_boxes_per_image")
    base_fp = baseline_negatives.get("false_boxes_per_image")
    negative_count = int(candidate_negatives.get("image_count") or 0)
    if negative_count >= 5 and cand_fp is not None:
        limit = max(0.35, (float(base_fp) * 1.15 + 0.05) if base_fp is not None else 0.35)
        if float(cand_fp) > limit:
            reasons.append("false_positives_on_negative_images_increased")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "candidate_precision": cand_precision,
        "baseline_precision": base_precision,
        "candidate_map50": cand_map50,
        "baseline_map50": base_map50,
        "candidate_recall": cand_recall,
        "collapsed_classes": collapsed_classes,
        "candidate_negative_audit": candidate_negatives,
        "baseline_negative_audit": baseline_negatives,
    }


def _validate_model_classes(model):
    expected = {int(item["id"]): str(item["key"]) for item in load_classes()}
    names = model.names
    if isinstance(names, (list, tuple)):
        received = {index: str(name) for index, name in enumerate(names)}
    else:
        received = {int(key): str(value) for key, value in dict(names).items()}
    if received != expected:
        raise RuntimeError(
            "O modelo treinado não preservou o mapa de classes. "
            f"Esperado: {expected}; recebido: {received}."
        )
    return received


def _stable_foundation_model(YOLO):
    """Cria/reutiliza uma origem imutável para evitar degradação cumulativa.

    O comportamento antigo treinava em cima do último peso ativo. Se um treino
    aprendesse falsos positivos, o treino seguinte começava justamente desse
    erro. Na primeira execução desta versão usamos o backup mais antigo (peso
    anterior aos ciclos de ajuste) quando disponível e o preservamos.
    """
    if FOUNDATION_MODEL_PATH.exists():
        model = YOLO(str(FOUNDATION_MODEL_PATH))
        _validate_model_classes(model)
        return FOUNDATION_MODEL_PATH
    previous = sorted(
        BACKUP_DIR.glob("circuit_scanner_yolo_*.pt"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    source = previous[0] if previous else MODEL_PATH
    model = YOLO(str(source))
    _validate_model_classes(model)
    FOUNDATION_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = FOUNDATION_MODEL_PATH.with_suffix(".candidate.pt")
    shutil.copy2(source, temporary)
    temporary.replace(FOUNDATION_MODEL_PATH)
    return FOUNDATION_MODEL_PATH


def _train_worker(config: dict):
    run_dir = Path(config["run_dir"])
    status = {
        **config,
        "state": "running",
        "started_at": config.get("started_at") or _now(),
        "resumed_at": _now() if config.get("resume") else None,
        "message": (
            "Retomando o treinamento pelo último checkpoint..."
            if config.get("resume")
            else "Treinando o reconhecimento de componentes..."
        ),
    }
    _write_json(STATUS_PATH, status)
    backup_path = None
    try:
        from ultralytics import YOLO

        mixed_dataset_dir, real_eval_dir = active_dataset_dirs()
        dataset_yaml = mixed_dataset_dir / "dataset.yaml"
        if not dataset_yaml.exists():
            raise FileNotFoundError(
                "Gere o dataset misto antes de iniciar o treinamento."
            )
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "O modelo base models/circuit_scanner_yolo.pt não foi encontrado."
            )

        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = BACKUP_DIR / f"circuit_scanner_yolo_{stamp}.pt"
        shutil.copy2(MODEL_PATH, backup_path)

        foundation_path = None
        if config.get("resume"):
            checkpoint = Path(config["checkpoint"])
            if not checkpoint.exists():
                raise FileNotFoundError(
                    f"Checkpoint de retomada não encontrado: {checkpoint}"
                )
            model = YOLO(str(checkpoint))
            _validate_model_classes(model)
            model.add_callback("on_train_epoch_end", _stop_after_current_epoch)
            # O Ultralytics recupera época, otimizador, agenda de aprendizado,
            # paciência e argumentos originais diretamente de last.pt.
            model.train(resume=True)
        else:
            foundation_path = _stable_foundation_model(YOLO)
            model = YOLO(str(foundation_path))
            _validate_model_classes(model)
            model.add_callback("on_train_epoch_end", _stop_after_current_epoch)
            model.train(
                data=str(dataset_yaml),
                epochs=int(config["epochs"]),
                imgsz=int(config["imgsz"]),
                batch=int(config["batch"]),
                patience=int(config["patience"]),
                device=config["device"],
                project=str(RUNS_DIR),
                name=config["run_name"],
                exist_ok=True,
                plots=True,
                verbose=True,
                # Augmentações moderadas e coerentes com esquemas elétricos.
                # O dataset já contém recortes focados e negativos reais;
                # mosaic agressivo só criaria circuitos impossíveis.
                mosaic=0.35,
                close_mosaic=min(10, max(1, int(config["epochs"]) // 4)),
                mixup=0.0,
                degrees=3.0,
                translate=0.08,
                scale=0.25,
                perspective=0.0004,
                fliplr=0.50,
                flipud=0.50,
                hsv_h=0.01,
                hsv_s=0.25,
                hsv_v=0.25,
            )

        best_path = run_dir / "weights" / "best.pt"
        if not best_path.exists():
            raise RuntimeError("O treino terminou sem produzir weights/best.pt.")
        candidate = YOLO(str(best_path))
        class_mapping = _validate_model_classes(candidate)
        mixed_metrics = _validation_summary(
            candidate.val(data=str(dataset_yaml), split="val", plots=False)
        )
        real_yaml = real_eval_dir / "dataset.yaml"
        real_metrics = None
        use_real_gate = False
        if real_yaml.exists() and any((real_eval_dir / "images" / "val").glob("*")):
            real_metrics = _validation_summary(
                candidate.val(data=str(real_yaml), split="val", plots=False)
            )

        # Comparamos o candidato com o peso que estava ativo antes do treino.
        # A validação padrão do Ultralytics, sozinha, não impede um modelo
        # superconfiante de inventar dezenas de componentes em pautas.
        baseline = YOLO(str(backup_path))
        _validate_model_classes(baseline)
        if real_metrics is not None:
            baseline_metrics = _validation_summary(
                baseline.val(data=str(real_yaml), split="val", plots=False)
            )
            gate_candidate_metrics = real_metrics
            use_real_gate = True
        else:
            baseline_metrics = _validation_summary(
                baseline.val(data=str(dataset_yaml), split="val", plots=False)
            )
            gate_candidate_metrics = mixed_metrics

        negative_audit_dir = real_eval_dir if real_yaml.exists() else mixed_dataset_dir
        candidate_negatives = _negative_false_positive_summary(
            candidate, negative_audit_dir, int(config["imgsz"])
        )
        baseline_negatives = _negative_false_positive_summary(
            baseline, negative_audit_dir, int(config["imgsz"])
        )
        quality_gate = _candidate_quality_gate(
            gate_candidate_metrics,
            baseline_metrics,
            candidate_negatives,
            baseline_negatives,
        )

        manifest = _read_json(mixed_dataset_dir / "manifest.json", {})
        if not quality_gate["passed"]:
            rejected_summary = {
                "completed_at": _now(),
                "candidate_model": str(best_path),
                "active_model_preserved": str(MODEL_PATH.relative_to(BASE_DIR)),
                "backup": str(backup_path.relative_to(BASE_DIR)),
                "mixed_validation": mixed_metrics,
                "real_only_validation": real_metrics,
                "baseline_validation_for_gate": baseline_metrics,
                "quality_gate_dataset": "real_only_val" if use_real_gate else "mixed_val",
                "quality_gate": quality_gate,
                "foundation_model": (
                    str(foundation_path.relative_to(BASE_DIR))
                    if foundation_path else "resume_checkpoint"
                ),
                "dataset_manifest": manifest,
                "safety": "Candidato rejeitado; modelo ativo anterior preservado.",
            }
            _write_json(SUMMARY_PATH, rejected_summary)
            status.update({
                "state": "rejected",
                "completed_at": _now(),
                "message": (
                    "Treinamento terminou, mas o modelo novo foi rejeitado por piorar "
                    "a precisão ou os falsos positivos. O modelo anterior foi preservado."
                ),
                "backup_path": str(backup_path.relative_to(BASE_DIR)),
                "summary": rejected_summary,
                "progress_percent": 100.0,
            })
            return

        candidate_copy = MODEL_PATH.with_suffix(".candidate.pt")
        shutil.copy2(best_path, candidate_copy)
        candidate_copy.replace(MODEL_PATH)
        summary = {
            "completed_at": _now(),
            "model": str(MODEL_PATH.relative_to(BASE_DIR)),
            "model_sha256": _sha256(MODEL_PATH),
            "backup": str(backup_path.relative_to(BASE_DIR)),
            "class_mapping": class_mapping,
            "mixed_validation": mixed_metrics,
            "real_only_validation": real_metrics,
            "baseline_validation_for_gate": baseline_metrics,
            "quality_gate_dataset": "real_only_val" if use_real_gate else "mixed_val",
            "quality_gate": quality_gate,
            "foundation_model": (
                str(foundation_path.relative_to(BASE_DIR))
                if foundation_path else "resume_checkpoint"
            ),
            "training": {
                key: config[key]
                for key in ("epochs", "imgsz", "batch", "patience", "device")
            },
            "resumed_from_checkpoint": config.get("checkpoint"),
            "dataset_manifest": manifest,
            "safety": "Modelo ativo substituído somente após validar best.pt.",
        }
        _write_json(SUMMARY_PATH, summary)
        status.update(
            {
                "state": "completed",
                "completed_at": _now(),
                "message": "Treinamento concluído e modelo ativo atualizado.",
                "backup_path": str(backup_path.relative_to(BASE_DIR)),
                "summary": summary,
                "progress_percent": 100.0,
            }
        )
    except Exception as error:
        status.update(
            {
                "state": "failed",
                "completed_at": _now(),
                "message": str(error),
                "traceback": traceback.format_exc(limit=8),
                "backup_path": (
                    str(backup_path.relative_to(BASE_DIR)) if backup_path else None
                ),
            }
        )
    finally:
        _write_json(STATUS_PATH, status)


def start_component_training(
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 8,
    patience: int = 15,
    device: str = "cpu",
):
    global _THREAD
    epochs, imgsz, batch, patience = map(int, (epochs, imgsz, batch, patience))
    if not 1 <= epochs <= 500:
        raise ValueError("epochs deve ficar entre 1 e 500.")
    if imgsz not in {320, 416, 512, 640, 768, 960, 1024}:
        raise ValueError("imgsz deve ser 320, 416, 512, 640, 768, 960 ou 1024.")
    if not 1 <= batch <= 128:
        raise ValueError("batch deve ficar entre 1 e 128.")
    if not 1 <= patience <= epochs:
        raise ValueError("patience deve ficar entre 1 e epochs.")
    active_mixed, _ = active_dataset_dirs()
    if not (active_mixed / "dataset.yaml").exists():
        raise ValueError("Gere o dataset misto antes de treinar.")
    if not MODEL_PATH.exists():
        raise ValueError("O modelo base circuit_scanner_yolo.pt não foi encontrado.")

    with _LOCK:
        if _THREAD and _THREAD.is_alive():
            raise RuntimeError("Já existe um treinamento de componentes em andamento.")
        _STOP_REQUESTED.clear()
        run_name = datetime.now().strftime("train_%Y%m%d_%H%M%S")
        config = {
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "patience": patience,
            "device": str(device or "cpu"),
            "run_name": run_name,
            "run_dir": str((RUNS_DIR / run_name).resolve()),
        }
        _THREAD = threading.Thread(
            target=_train_worker,
            args=(config,),
            daemon=True,
            name="component-recognition-training",
        )
        _THREAD.start()
    return {
        "state": "running",
        "message": "Treinamento iniciado em segundo plano.",
        "training_origin": "stable_foundation_model",
        **config,
    }


def resume_component_training():
    """Retoma a última execução interrompida usando weights/last.pt."""
    global _THREAD
    effective = component_training_status()
    if effective.get("state") not in {"interrupted", "failed"}:
        raise ValueError("Não existe treinamento interrompido para retomar.")
    checkpoint_text = effective.get("resume_checkpoint")
    if not checkpoint_text:
        raise ValueError(
            "O checkpoint weights/last.pt não foi encontrado nessa execução."
        )
    checkpoint = Path(checkpoint_text)
    run_dir = Path(effective["run_dir"])
    config = {
        "epochs": int(effective.get("epochs", 50)),
        "imgsz": int(effective.get("imgsz", 640)),
        "batch": int(effective.get("batch", 8)),
        "patience": int(effective.get("patience", 15)),
        "device": str(effective.get("device", "cpu")),
        "run_name": str(effective.get("run_name", run_dir.name)),
        "run_dir": str(run_dir),
        "started_at": effective.get("started_at"),
        "resume": True,
        "checkpoint": str(checkpoint),
    }
    with _LOCK:
        if _THREAD and _THREAD.is_alive():
            raise RuntimeError("Já existe um treinamento de componentes em andamento.")
        _STOP_REQUESTED.clear()
        _THREAD = threading.Thread(
            target=_train_worker,
            args=(config,),
            daemon=True,
            name="component-recognition-training-resume",
        )
        _THREAD.start()
    return {
        "state": "running",
        "message": "Treinamento retomado pelo último checkpoint.",
        **config,
    }


def _stop_after_current_epoch(trainer):
    """Callback Ultralytics: termina com segurança e mantém weights/best.pt."""
    if _STOP_REQUESTED.is_set():
        trainer.stop = True


def request_component_training_stop():
    """Pede parada ao fim da época atual; o worker ainda valida o melhor peso."""
    with _LOCK:
        if not (_THREAD and _THREAD.is_alive()):
            raise ValueError("Não existe treinamento em andamento.")
        _STOP_REQUESTED.set()
        status = _read_json(STATUS_PATH, {}) or {}
        status.update({
            "state": "stopping",
            "message": (
                "Parada solicitada. A época atual será concluída e o melhor "
                "checkpoint será validado normalmente."
            ),
            "stop_requested_at": _now(),
        })
        _write_json(STATUS_PATH, status)
    return {
        "state": "stopping",
        "message": status["message"],
    }


def rollback_component_model():
    """Restaura o backup mais recente criado antes do último treinamento."""
    with _LOCK:
        if _THREAD and _THREAD.is_alive():
            raise RuntimeError("Não é possível restaurar durante um treinamento.")
        backups = sorted(
            BACKUP_DIR.glob("circuit_scanner_yolo_*.pt"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        if not backups:
            raise ValueError("Nenhum backup anterior do modelo foi encontrado.")
        selected = backups[0]
        try:
            from ultralytics import YOLO
        except Exception as error:
            raise RuntimeError("A biblioteca ultralytics não está instalada.") from error
        restored_model = YOLO(str(selected))
        _validate_model_classes(restored_model)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rejected_copy = BACKUP_DIR / f"rejected_active_{stamp}.pt"
        if MODEL_PATH.exists():
            shutil.copy2(MODEL_PATH, rejected_copy)
        temporary = MODEL_PATH.with_suffix(".rollback.pt")
        shutil.copy2(selected, temporary)
        temporary.replace(MODEL_PATH)
        status = {
            "state": "rolled_back",
            "completed_at": _now(),
            "message": "Modelo anterior restaurado com sucesso.",
            "restored_backup": str(selected.relative_to(BASE_DIR)),
            "rejected_model_copy": (
                str(rejected_copy.relative_to(BASE_DIR))
                if rejected_copy.exists() else None
            ),
            "model_sha256": _sha256(MODEL_PATH),
        }
        _write_json(STATUS_PATH, status)
        return status


def set_foundation_from_active_model():
    """Fixa o peso ativo confirmado pelo usuário como origem dos próximos treinos."""
    with _LOCK:
        if _THREAD and _THREAD.is_alive():
            raise RuntimeError(
                "Não é possível alterar a base estável durante um treinamento."
            )
        if not MODEL_PATH.exists():
            raise ValueError("O modelo ativo não foi encontrado.")
        try:
            from ultralytics import YOLO
        except Exception as error:
            raise RuntimeError("A biblioteca ultralytics não está instalada.") from error
        active = YOLO(str(MODEL_PATH))
        class_mapping = _validate_model_classes(active)
        if FOUNDATION_MODEL_PATH.exists():
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            history = BACKUP_DIR / f"foundation_{stamp}.pt"
            BACKUP_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(FOUNDATION_MODEL_PATH, history)
        temporary = FOUNDATION_MODEL_PATH.with_suffix(".candidate.pt")
        shutil.copy2(MODEL_PATH, temporary)
        temporary.replace(FOUNDATION_MODEL_PATH)
        status = {
            "state": "foundation_updated",
            "completed_at": _now(),
            "message": (
                "Modelo ativo fixado como base estável dos próximos treinamentos."
            ),
            "foundation_model": str(FOUNDATION_MODEL_PATH.relative_to(BASE_DIR)),
            "foundation_sha256": _sha256(FOUNDATION_MODEL_PATH),
            "class_mapping": class_mapping,
        }
        _write_json(STATUS_PATH, status)
        return status
