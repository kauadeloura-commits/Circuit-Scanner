from pathlib import Path
from datetime import datetime
import hashlib
import json
import random
import shutil
from uuid import uuid4

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
REAL_DATA_DIR = BASE_DIR / "real_data"
REAL_IMAGES_DIR = REAL_DATA_DIR / "images"
REAL_ANNOTATIONS_DIR = REAL_DATA_DIR / "annotations"
REAL_META_DIR = REAL_DATA_DIR / "meta"
MIXED_DATASET_DIR = BASE_DIR / "dataset_mixed"
REAL_EVAL_DIR = BASE_DIR / "dataset_real_eval"
DATASET_BUILDS_DIR = BASE_DIR / "dataset_builds"
ACTIVE_DATASET_PATH = DATASET_BUILDS_DIR / "active.json"
SYNTHETIC_DATASET_DIR = BASE_DIR / "dataset"
OCR_TEXT_NEGATIVE_DIR = BASE_DIR / "text_negatives"
CLASSES_PATH = SYNTHETIC_DATASET_DIR / "classes.json"

for directory in (REAL_DATA_DIR, REAL_IMAGES_DIR, REAL_ANNOTATIONS_DIR, REAL_META_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def active_dataset_dirs():
    """Retorna o último dataset concluído sem depender de uma pasta removível.

    No Windows, antivírus, Explorer ou outro processo pode manter um diretório
    aberto e fazer ``shutil.rmtree`` falhar com WinError 5. Os novos builds são
    imutáveis e um pequeno arquivo JSON escolhe qual deles está ativo.
    """
    try:
        pointer = json.loads(ACTIVE_DATASET_PATH.read_text(encoding="utf-8"))
        mixed = BASE_DIR / pointer["mixed_relative_path"]
        real_eval = BASE_DIR / pointer["real_eval_relative_path"]
        if (mixed / "dataset.yaml").exists():
            return mixed, real_eval
    except Exception:
        pass
    return MIXED_DATASET_DIR, REAL_EVAL_DIR

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_SOURCE_TYPES = {"handwritten", "cad"}
TEXT_NEGATIVE_TOKENS = (
    "led", "LED", "Led", "LED1", "LED2", "R1", "R2", "R3", "R4",
    "R5", "R10", "C1", "C2", "C3", "D1", "Q1", "GND", "VCC",
    "10V", "5V", "3K", "5K", "10K", "3K3", "100nF", "10uF",
    "1N4148", "BC547", "2N2222", "LM358", "100ohm",
)


def load_classes():
    """Carrega o registro estável de classes, mesmo antes de gerar o dataset.

    A página de anotação precisa funcionar em uma instalação nova, quando
    ainda existem apenas as prévias sintéticas. Mantemos os IDs vindos do
    gerador para que as anotações nunca mudem de significado entre treinos.
    """
    if not CLASSES_PATH.exists():
        from backend.dataset_generator import CLASSES

        CLASSES_PATH.parent.mkdir(parents=True, exist_ok=True)
        classes = [
            {"id": int(class_id), "key": key, "name": name}
            for class_id, key, name in CLASSES
        ]
        CLASSES_PATH.write_text(
            json.dumps(classes, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return classes
    return json.loads(CLASSES_PATH.read_text(encoding="utf-8"))


def _read_image_size(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Não foi possível abrir a imagem real.")
    height, width = image.shape[:2]
    return int(width), int(height)


def _next_real_id():
    highest = 0
    for path in REAL_META_DIR.glob("REAL_*.json"):
        try:
            highest = max(highest, int(path.stem.split("_")[1]))
        except Exception:
            continue
    return f"REAL_{highest + 1:04d}"


def _meta_path(image_id: str):
    return REAL_META_DIR / f"{image_id}.json"


def _annotation_path(image_id: str):
    return REAL_ANNOTATIONS_DIR / f"{image_id}.json"


def upload_real_image(
    file_obj,
    filename: str,
    name: str = "",
    source_type: str = "handwritten",
    source_group: str = "",
):
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError("Envie uma imagem JPG, JPEG ou PNG.")
    source_type = str(source_type or "handwritten").strip().lower()
    if source_type not in ALLOWED_SOURCE_TYPES:
        raise ValueError("Origem inválida. Use handwritten ou cad.")

    image_id = _next_real_id()
    stored_filename = f"{image_id}{extension}"
    destination = REAL_IMAGES_DIR / stored_filename

    with destination.open("wb") as output:
        shutil.copyfileobj(file_obj, output)

    try:
        width, height = _read_image_size(destination)
    except Exception:
        destination.unlink(missing_ok=True)
        raise

    clean_group = str(source_group or "").strip()[:120]
    source_group_id = (
        hashlib.sha256(
            f"{source_type}:{clean_group.casefold()}".encode("utf-8")
        ).hexdigest()[:20]
        if clean_group
        else image_id
    )
    meta = {
        "image_id": image_id,
        "name": name.strip() or Path(filename).stem,
        "original_filename": Path(filename).name,
        "stored_filename": stored_filename,
        "width": width,
        "height": height,
        "source_type": source_type,
        "source_group": clean_group,
        "source_group_id": source_group_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    _meta_path(image_id).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _annotation_path(image_id).write_text(
        json.dumps({"image_id": image_id, "boxes": []}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return get_real_image(image_id)


def get_real_image(image_id: str):
    meta_path = _meta_path(image_id)
    if not meta_path.exists():
        return None

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    # Imagens cadastradas antes da área CAD continuam válidas e são tratadas
    # como desenhos/fotos feitos à mão.
    meta.setdefault("source_type", "handwritten")
    meta.setdefault("source_group", "")
    meta.setdefault("source_group_id", image_id)
    ann_path = _annotation_path(image_id)
    annotations = {"image_id": image_id, "boxes": []}
    if ann_path.exists():
        annotations = json.loads(ann_path.read_text(encoding="utf-8"))

    meta["annotations"] = annotations.get("boxes", [])
    meta["annotation_count"] = len(meta["annotations"])
    meta["annotations_reviewed"] = bool(
        annotations.get("reviewed") or annotations.get("updated_at")
    )
    meta["image_url"] = f"http://127.0.0.1:8010/real-data-files/images/{meta['stored_filename']}"
    return meta


def list_real_images():
    records = []
    for meta_path in sorted(REAL_META_DIR.glob("REAL_*.json")):
        record = get_real_image(meta_path.stem)
        if record:
            records.append(record)
    return records


def save_annotations(image_id: str, boxes):
    record = get_real_image(image_id)
    if not record:
        raise FileNotFoundError("Imagem real não encontrada.")

    class_ids = {item["id"] for item in load_classes()}
    width = record["width"]
    height = record["height"]
    clean = []

    for index, box in enumerate(boxes, start=1):
        class_id = int(box.get("class_id", -1))
        if class_id not in class_ids:
            raise ValueError(f"Classe inválida na caixa {index}.")

        x1 = float(box.get("x1", 0))
        y1 = float(box.get("y1", 0))
        x2 = float(box.get("x2", 0))
        y2 = float(box.get("y2", 0))

        x1, x2 = sorted((max(0.0, x1), min(float(width), x2)))
        y1, y2 = sorted((max(0.0, y1), min(float(height), y2)))

        if x2 - x1 < 3 or y2 - y1 < 3:
            raise ValueError(f"Caixa {index} é pequena ou inválida.")

        clean.append({
            "class_id": class_id,
            "x1": round(x1, 2),
            "y1": round(y1, 2),
            "x2": round(x2, 2),
            "y2": round(y2, 2),
        })

    payload = {
        "image_id": image_id,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "reviewed": True,
        "boxes": clean,
    }
    _annotation_path(image_id).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return get_real_image(image_id)


def real_dataset_info():
    records = list_real_images()
    annotated = [r for r in records if r["annotation_count"] > 0]
    reviewed_negatives = [
        r for r in records
        if r.get("annotations_reviewed") and r["annotation_count"] == 0
    ]
    total_boxes = sum(r["annotation_count"] for r in annotated)
    by_class = {item["key"]: 0 for item in load_classes()}
    class_by_id = {item["id"]: item["key"] for item in load_classes()}
    images_by_source = {key: 0 for key in ALLOWED_SOURCE_TYPES}
    annotated_images_by_source = {key: 0 for key in ALLOWED_SOURCE_TYPES}
    boxes_by_source = {key: 0 for key in ALLOWED_SOURCE_TYPES}

    for record in records:
        source_type = record.get("source_type", "handwritten")
        images_by_source[source_type] = images_by_source.get(source_type, 0) + 1
        if record["annotation_count"] > 0:
            annotated_images_by_source[source_type] = (
                annotated_images_by_source.get(source_type, 0) + 1
            )
            boxes_by_source[source_type] = (
                boxes_by_source.get(source_type, 0) + record["annotation_count"]
            )

    for record in annotated:
        for box in record["annotations"]:
            key = class_by_id.get(int(box["class_id"]))
            if key:
                by_class[key] += 1

    active_mixed, _ = active_dataset_dirs()
    return {
        "images": len(records),
        "annotated_images": len(annotated),
        "boxes": total_boxes,
        "reviewed_negative_images": len(reviewed_negatives),
        "boxes_by_class": by_class,
        "images_by_source": images_by_source,
        "annotated_images_by_source": annotated_images_by_source,
        "boxes_by_source": boxes_by_source,
        "mixed_dataset_ready": (active_mixed / "manifest.json").exists(),
        "active_mixed_dataset": (
            str(active_mixed.relative_to(BASE_DIR))
            if (active_mixed / "manifest.json").exists()
            else None
        ),
    }


def _to_yolo_label(record):
    width = float(record["width"])
    height = float(record["height"])
    lines = []
    for box in record["annotations"]:
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        cx = ((x1 + x2) / 2.0) / width
        cy = ((y1 + y2) / 2.0) / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height
        lines.append(
            f"{int(box['class_id'])} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def _copy_synthetic(target: Path):
    counts = {"train": 0, "val": 0, "test": 0}
    for split in counts:
        source_images = SYNTHETIC_DATASET_DIR / "images" / split
        source_labels = SYNTHETIC_DATASET_DIR / "labels" / split
        if not source_images.exists():
            continue
        for image_path in source_images.iterdir():
            if not image_path.is_file():
                continue
            label_path = source_labels / f"{image_path.stem}.txt"
            new_name = f"syn_{image_path.name}"
            shutil.copy2(image_path, target / "images" / split / new_name)
            if label_path.exists():
                shutil.copy2(
                    label_path,
                    target / "labels" / split / f"syn_{label_path.name}",
                )
            counts[split] += 1
    return counts


def _photometric_augmentation(
    image,
    rng: random.Random,
    variant_index: int | None = None,
):
    """Varia cor e iluminação sem alterar as caixas dos componentes.

    O ciclo de variantes garante que o treino veja, no mínimo, mudanças de
    matiz/saturação, escala de cinza, inversão de contraste e dominantes RGB.
    Isso evita que o modelo associe uma classe à cor do símbolo ou do fundo.
    """
    out = image.copy()

    alpha = rng.uniform(0.75, 1.30)
    beta = rng.uniform(-35, 35)
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    if rng.random() < 0.65:
        sigma = rng.uniform(2.0, 12.0)
        noise = np.random.default_rng(rng.randint(0, 2**32 - 1)).normal(
            0, sigma, out.shape
        ).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if rng.random() < 0.35:
        kernel = rng.choice([3, 5])
        out = cv2.GaussianBlur(out, (kernel, kernel), 0)

    mode = (
        variant_index % 4
        if variant_index is not None
        else rng.randrange(4)
    )
    if mode == 0:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int16) + rng.randint(25, 155)) % 180
        hsv[:, :, 1] = np.clip(
            hsv[:, :, 1].astype(np.float32) * rng.uniform(0.35, 1.80),
            0,
            255,
        ).astype(np.uint8)
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    elif mode == 1:
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif mode == 2:
        out = 255 - out
    else:
        gains = np.array(
            [rng.uniform(0.45, 1.65) for _ in range(3)],
            dtype=np.float32,
        )
        out = np.clip(out.astype(np.float32) * gains, 0, 255).astype(np.uint8)

    return out


def _monochrome_training_variants(image):
    """Vistas P&B obrigatórias para toda imagem real de treino.

    As caixas e IDs não mudam. Logo, qualquer classe presente na anotação
    (inclusive classes acrescentadas futuramente em classes.json) recebe os
    mesmos exemplos CLAHE, Otsu e limiar adaptativo.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    border = np.concatenate((gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]))
    if float(np.median(border)) < 127.0:
        gray = 255 - gray
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.GaussianBlur(clahe, (3, 3), 0)
    _, otsu = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    block = max(15, min(61, int(round(min(gray.shape[:2]) / 12.0))))
    if block % 2 == 0:
        block += 1
    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block,
        9,
    )
    return {
        "mono_clahe": cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR),
        "mono_otsu": cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR),
        "mono_adaptive": cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR),
    }


def _save_image(path: Path, image):
    suffix = path.suffix.lower()
    ext = ".jpg" if suffix in {".jpg", ".jpeg"} else ".png"
    params = [int(cv2.IMWRITE_JPEG_QUALITY), 82] if ext == ".jpg" else []
    ok, buffer = cv2.imencode(ext, image, params)
    if not ok:
        raise RuntimeError("Falha ao salvar imagem do dataset misto.")
    buffer.tofile(str(path))


def _labels_for_crop(boxes, crop_x1, crop_y1, crop_x2, crop_y2):
    """Converte as caixas reais para um recorte, sem criar caixas parciais.

    Uma caixa entra somente quando seu centro está no recorte e pelo menos
    72% do símbolo continua visível. Isso evita ensinar ao YOLO que metade de
    um componente é um exemplo completo.
    """
    crop_w = float(max(1, crop_x2 - crop_x1))
    crop_h = float(max(1, crop_y2 - crop_y1))
    lines = []
    included = []
    for box in boxes:
        bx1, by1 = float(box["x1"]), float(box["y1"])
        bx2, by2 = float(box["x2"]), float(box["y2"])
        center_x = (bx1 + bx2) / 2.0
        center_y = (by1 + by2) / 2.0
        if not (crop_x1 <= center_x <= crop_x2 and crop_y1 <= center_y <= crop_y2):
            continue
        ix1, iy1 = max(bx1, crop_x1), max(by1, crop_y1)
        ix2, iy2 = min(bx2, crop_x2), min(by2, crop_y2)
        intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area = max(1.0, (bx2 - bx1) * (by2 - by1))
        if intersection / area < 0.72:
            continue
        local_x1, local_y1 = ix1 - crop_x1, iy1 - crop_y1
        local_x2, local_y2 = ix2 - crop_x1, iy2 - crop_y1
        cx = ((local_x1 + local_x2) / 2.0) / crop_w
        cy = ((local_y1 + local_y2) / 2.0) / crop_h
        bw = (local_x2 - local_x1) / crop_w
        bh = (local_y2 - local_y1) / crop_h
        lines.append(
            f"{int(box['class_id'])} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
        )
        included.append(box)
    return "\n".join(lines) + ("\n" if lines else ""), included


def _real_page_tiles(record, image, rng: random.Random):
    """Gera recortes na escala dos símbolos e fundos difíceis da mesma folha.

    As fotos do usuário contêm muitos componentes pequenos na mesma página.
    Reduzir a página inteira para 640 px desperdiça os mais de cem exemplos de
    cada classe. Os recortes abaixo fazem cada caixa anotada ocupar uma escala
    útil e usam fios/pautas sem caixa como negativos reais.
    """
    height, width = image.shape[:2]
    boxes = list(record.get("annotations") or [])
    positive = []
    origins_seen = set()
    for box_index, box in enumerate(boxes):
        bw = max(3.0, float(box["x2"]) - float(box["x1"]))
        bh = max(3.0, float(box["y2"]) - float(box["y1"]))
        # Cerca de 3,6 corpos do componente por lado preserva fios próximos,
        # mas remove a maior parte do restante do circuito.
        side = int(round(max(256.0, 3.6 * max(bw, bh))))
        side = min(side, max(width, height), 768)
        crop_w, crop_h = min(side, width), min(side, height)
        cx = (float(box["x1"]) + float(box["x2"])) / 2.0
        cy = (float(box["y1"]) + float(box["y2"])) / 2.0
        x1 = int(round(max(0.0, min(width - crop_w, cx - crop_w / 2.0))))
        y1 = int(round(max(0.0, min(height - crop_h, cy - crop_h / 2.0))))
        key = (x1 // 24, y1 // 24, crop_w // 32, crop_h // 32)
        if key in origins_seen:
            continue
        origins_seen.add(key)
        x2, y2 = x1 + crop_w, y1 + crop_h
        label, included = _labels_for_crop(boxes, x1, y1, x2, y2)
        if not included:
            continue
        positive.append({
            "name": f"focus_{box_index:04d}",
            "image": image[y1:y2, x1:x2].copy(),
            "label": label,
            "box_count": len(included),
        })

    # Fundos retirados das próprias folhas: pautas, fios, textos e espaços
    # entre símbolos. São negativos muito mais úteis do que imagens vazias.
    negative = []
    tile = min(448, width, height)
    if tile >= 160:
        stride = max(80, int(tile * 0.55))
        candidates = []
        xs = list(range(0, max(1, width - tile + 1), stride))
        ys = list(range(0, max(1, height - tile + 1), stride))
        if xs[-1] != width - tile:
            xs.append(width - tile)
        if ys[-1] != height - tile:
            ys.append(height - tile)
        for y1 in ys:
            for x1 in xs:
                label, included = _labels_for_crop(
                    boxes, x1, y1, x1 + tile, y1 + tile
                )
                if not included and not label:
                    candidates.append((x1, y1))
        rng.shuffle(candidates)
        for neg_index, (x1, y1) in enumerate(candidates[:4]):
            negative.append({
                "name": f"background_{neg_index:02d}",
                "image": image[y1:y1 + tile, x1:x1 + tile].copy(),
                "label": "",
                "box_count": 0,
            })
    return positive, negative


def _write_real_tiles(target: Path, split: str, base_name: str, tiles):
    count = 0
    positive_count = 0
    negative_count = 0
    for tile in tiles:
        name = f"{base_name}_{tile['name']}.png"
        _save_image(target / "images" / split / name, tile["image"])
        (target / "labels" / split / f"{Path(name).stem}.txt").write_text(
            tile["label"], encoding="utf-8"
        )
        count += 1
        if tile["box_count"]:
            positive_count += 1
        else:
            negative_count += 1
    return {
        "total": count,
        "positive": positive_count,
        "negative": negative_count,
    }


def _negative_split(index: int, total: int) -> str:
    if total < 10:
        return "train"
    if index < round(total * 0.80):
        return "train"
    if index < round(total * 0.90):
        return "val"
    return "test"


def _generate_text_negatives(target: Path, count: int, seed: int):
    counts = {"train": 0, "val": 0, "test": 0}
    rng = random.Random(seed + 773)
    fonts = (
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_PLAIN,
        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    )
    for index in range(count):
        split = _negative_split(index, count)
        tamanho = rng.choice((384, 448, 512, 640))
        fundo = rng.randint(232, 255)
        imagem = np.full((tamanho, tamanho, 3), fundo, dtype=np.uint8)
        if rng.random() < 0.80:
            passo = rng.randint(25, 42)
            inicio = rng.randint(8, passo)
            cor_linha = rng.randint(190, 225)
            for y in range(inicio, tamanho, passo):
                cv2.line(imagem, (0, y), (tamanho, y), (cor_linha,) * 3, 1)

        quantidade = rng.randint(4, 10)
        for _ in range(quantidade):
            texto = rng.choice(TEXT_NEGATIVE_TOKENS)
            if rng.random() < 0.20:
                texto = f"{rng.choice('RCDQLUV')}{rng.randint(1, 29)}"
            fonte = rng.choice(fonts)
            escala = rng.uniform(0.65, 1.45)
            espessura = rng.choice((1, 2, 2, 3))
            (largura, altura), _ = cv2.getTextSize(
                texto,
                fonte,
                escala,
                espessura,
            )
            x = rng.randint(8, max(8, tamanho - largura - 8))
            y = rng.randint(altura + 8, max(altura + 8, tamanho - 8))
            cor = rng.randint(5, 80)
            cv2.putText(
                imagem,
                texto,
                (x, y),
                fonte,
                escala,
                (cor, cor, cor),
                espessura,
                cv2.LINE_AA,
            )

        angulo = rng.uniform(-4.0, 4.0)
        matriz = cv2.getRotationMatrix2D(
            (tamanho / 2, tamanho / 2),
            angulo,
            1.0,
        )
        imagem = cv2.warpAffine(
            imagem,
            matriz,
            (tamanho, tamanho),
            flags=cv2.INTER_LINEAR,
            borderValue=(fundo, fundo, fundo),
        )
        imagem = _photometric_augmentation(imagem, rng)
        nome = f"hardtext_generated_{index:05d}.jpg"
        _save_image(target / "images" / split / nome, imagem)
        (target / "labels" / split / f"{Path(nome).stem}.txt").write_text(
            "",
            encoding="utf-8",
        )
        counts[split] += 1
    return counts


def _copy_ocr_text_negatives(target: Path, seed: int):
    counts = {"train": 0, "val": 0, "test": 0}
    imagens = sorted(OCR_TEXT_NEGATIVE_DIR.glob("text_*.jpg"))
    rng = random.Random(seed + 991)
    rng.shuffle(imagens)
    for index, source in enumerate(imagens):
        split = _negative_split(index, len(imagens))
        nome = f"hardtext_ocr_{source.name}"
        shutil.copy2(source, target / "images" / split / nome)
        (target / "labels" / split / f"{Path(nome).stem}.txt").write_text(
            "",
            encoding="utf-8",
        )
        counts[split] += 1
    return counts


def _assign_real_splits(records, seed: int):
    """Divide por grupos de origem sem vazar páginas/versões relacionadas."""
    by_source = {}
    for record in records:
        source_type = record.get("source_type", "handwritten")
        group_id = record.get("source_group_id") or record["image_id"]
        by_source.setdefault(source_type, {}).setdefault(group_id, []).append(
            record["image_id"]
        )

    mapping = {}
    rng = random.Random(seed)
    for source_type in sorted(by_source):
        groups = list(by_source[source_type].items())
        rng.shuffle(groups)
        n = len(groups)
        if n < 5:
            split_names = ["train"] * n
        else:
            n_test = max(1, round(n * 0.10))
            n_val = max(1, round(n * 0.20))
            n_train = max(1, n - n_val - n_test)
            split_names = (
                ["train"] * n_train
                + ["val"] * n_val
                + ["test"] * (n - n_train - n_val)
            )
        for (_, image_ids), split in zip(groups, split_names):
            for image_id in image_ids:
                mapping[image_id] = split
    return mapping


def build_mixed_dataset(
    real_train_augments: int = 2,
    seed: int = 4411,
    text_negative_count: int = 450,
):
    real_train_augments = int(real_train_augments)
    if real_train_augments < 0 or real_train_augments > 10:
        raise ValueError("real_train_augments deve ficar entre 0 e 10.")
    text_negative_count = int(text_negative_count)
    if text_negative_count < 100 or text_negative_count > 2000:
        raise ValueError("text_negative_count deve ficar entre 100 e 2000.")

    reviewed = [
        record for record in list_real_images()
        if record["annotation_count"] > 0 or record.get("annotations_reviewed")
    ]
    if not reviewed:
        raise ValueError("Ainda não há imagens reais revisadas.")

    # Cada geração recebe um diretório próprio. Não apagamos o build anterior:
    # isso elimina o WinError 5 quando alguma pasta está aberta/bloqueada.
    build_id = (
        datetime.now().strftime("build_%Y%m%d_%H%M%S_")
        + uuid4().hex[:8]
    )
    build_root = DATASET_BUILDS_DIR / build_id
    mixed_dataset_dir = build_root / "dataset_mixed"
    real_eval_dir = build_root / "dataset_real_eval"

    for split in ("train", "val", "test"):
        (mixed_dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (mixed_dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        (real_eval_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (real_eval_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    synthetic_counts = _copy_synthetic(mixed_dataset_dir)
    generated_text_negative_counts = _generate_text_negatives(
        mixed_dataset_dir,
        text_negative_count,
        seed,
    )
    ocr_text_negative_counts = _copy_ocr_text_negatives(
        mixed_dataset_dir,
        seed,
    )
    split_map = _assign_real_splits(reviewed, seed)
    real_counts = {"train": 0, "val": 0, "test": 0}
    real_counts_by_source = {
        source_type: {"train": 0, "val": 0, "test": 0}
        for source_type in sorted(ALLOWED_SOURCE_TYPES)
    }
    real_negative_counts = {"train": 0, "val": 0, "test": 0}
    augmented_count = 0
    monochrome_augmented_count = 0
    real_eval_monochrome_count = 0
    focused_tile_counts = {
        "train": {"total": 0, "positive": 0, "negative": 0},
        "val": {"total": 0, "positive": 0, "negative": 0},
        "test": {"total": 0, "positive": 0, "negative": 0},
    }
    rng = random.Random(seed)

    for record in reviewed:
        split = split_map[record["image_id"]]
        source = REAL_IMAGES_DIR / record["stored_filename"]
        extension = source.suffix.lower()
        base_name = record["image_id"].lower()
        target_image = mixed_dataset_dir / "images" / split / f"{base_name}{extension}"
        target_label = mixed_dataset_dir / "labels" / split / f"{base_name}.txt"
        shutil.copy2(source, target_image)
        label_text = _to_yolo_label(record)
        target_label.write_text(label_text, encoding="utf-8")

        # Cópia somente de originais reais para uma avaliação separada.
        # Isso evita que a métrica seja dominada pelas imagens sintéticas.
        eval_image = real_eval_dir / "images" / split / f"{base_name}{extension}"
        eval_label = real_eval_dir / "labels" / split / f"{base_name}.txt"
        shutil.copy2(source, eval_image)
        eval_label.write_text(label_text, encoding="utf-8")
        real_counts[split] += 1
        source_type = record.get("source_type", "handwritten")
        real_counts_by_source.setdefault(
            source_type, {"train": 0, "val": 0, "test": 0}
        )[split] += 1
        if record["annotation_count"] == 0:
            real_negative_counts[split] += 1

        data = np.fromfile(str(source), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is not None:
            positive_tiles, negative_tiles = _real_page_tiles(
                record, image, rng
            )
            # O treino recebe símbolos ampliados e fundos reais da MESMA
            # origem. Validação/teste recebem esses recortes apenas no
            # conjunto real de avaliação, nunca atravessando os grupos.
            tile_target = (
                mixed_dataset_dir if split == "train" else real_eval_dir
            )
            tile_stats = _write_real_tiles(
                tile_target,
                split,
                base_name,
                positive_tiles + negative_tiles,
            )
            for key, value in tile_stats.items():
                focused_tile_counts[split][key] += int(value)

            monochrome_variants = _monochrome_training_variants(image)
            if split == "train":
                for suffix, monochrome in monochrome_variants.items():
                    mono_name = f"{base_name}_{suffix}.png"
                    mono_image = mixed_dataset_dir / "images" / "train" / mono_name
                    mono_label = (
                        mixed_dataset_dir / "labels" / "train"
                        / f"{Path(mono_name).stem}.txt"
                    )
                    _save_image(mono_image, monochrome)
                    mono_label.write_text(label_text, encoding="utf-8")
                    monochrome_augmented_count += 1
            elif split in {"val", "test"}:
                # Avaliação P&B separada por imagem-fonte: nenhuma variação de
                # treino atravessa para validação/teste.
                for suffix, monochrome in monochrome_variants.items():
                    mono_name = f"{base_name}_{suffix}.png"
                    mono_image = real_eval_dir / "images" / split / mono_name
                    mono_label = (
                        real_eval_dir / "labels" / split
                        / f"{Path(mono_name).stem}.txt"
                    )
                    _save_image(mono_image, monochrome)
                    mono_label.write_text(label_text, encoding="utf-8")
                    real_eval_monochrome_count += 1

        # Somente o treino recebe variações. Assim nenhuma variação da mesma
        # foto original aparece em val/test e evitamos vazamento de dados.
        if split == "train" and real_train_augments:
            if image is None:
                continue
            for aug_index in range(real_train_augments):
                augmented = _photometric_augmentation(
                    image,
                    rng,
                    variant_index=aug_index,
                )
                aug_name = f"{base_name}_aug{aug_index + 1:02d}{extension}"
                aug_image = mixed_dataset_dir / "images" / "train" / aug_name
                aug_label = mixed_dataset_dir / "labels" / "train" / f"{Path(aug_name).stem}.txt"
                _save_image(aug_image, augmented)
                aug_label.write_text(_to_yolo_label(record), encoding="utf-8")
                augmented_count += 1

    classes = load_classes()
    names_lines = "\n".join(f"  {item['id']}: {item['key']}" for item in classes)
    yaml_text = (
        f'path: "{mixed_dataset_dir.resolve().as_posix()}"\n'
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        f"{names_lines}\n"
    )
    (mixed_dataset_dir / "dataset.yaml").write_text(yaml_text, encoding="utf-8")

    real_yaml_text = (
        f'path: "{real_eval_dir.resolve().as_posix()}"\n'
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        f"{names_lines}\n"
    )
    (real_eval_dir / "dataset.yaml").write_text(real_yaml_text, encoding="utf-8")

    total_counts = {
        split: synthetic_counts[split] + real_counts[split]
        + (augmented_count if split == "train" else 0)
        + (monochrome_augmented_count if split == "train" else 0)
        + (focused_tile_counts["train"]["total"] if split == "train" else 0)
        + generated_text_negative_counts[split]
        + ocr_text_negative_counts[split]
        for split in ("train", "val", "test")
    }

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "seed": seed,
        "real_train_augments": real_train_augments,
        "synthetic_counts": synthetic_counts,
        "real_original_counts": real_counts,
        "real_original_counts_by_source": real_counts_by_source,
        "real_reviewed_negative_counts": real_negative_counts,
        "real_augmented_train": augmented_count,
        "real_monochrome_augmented_train": monochrome_augmented_count,
        "real_eval_monochrome_variants": real_eval_monochrome_count,
        "focused_real_tiles": focused_tile_counts,
        "hard_negative_tiles_from_real_pages": {
            split: focused_tile_counts[split]["negative"]
            for split in ("train", "val", "test")
        },
        "generated_text_negative_counts": generated_text_negative_counts,
        "ocr_text_negative_counts": ocr_text_negative_counts,
        "text_negative_count_requested": text_negative_count,
        "total_counts": total_counts,
        "real_split_by_source": split_map,
        "real_source_groups": {
            record["image_id"]: {
                "source_type": record.get("source_type", "handwritten"),
                "source_group_id": record.get("source_group_id") or record["image_id"],
            }
            for record in reviewed
        },
        "real_only_evaluation_dataset": "dataset_real_eval/dataset.yaml",
        "build_id": build_id,
        "mixed_dataset_path": str(mixed_dataset_dir.relative_to(BASE_DIR)),
        "real_eval_path": str(real_eval_dir.relative_to(BASE_DIR)),
        "important_note": (
            "Variações da mesma imagem real ficam somente no split train; "
            "val/test nunca recebem cópias aumentadas da mesma fonte. "
            "Todas as classes anotadas recebem variantes monocromáticas."
            " Imagens CAD e feitas à mão são contabilizadas separadamente; "
            "páginas do mesmo grupo de origem permanecem no mesmo split."
            " O treino também usa recortes focados nos símbolos e recortes"
            " sem caixas da própria página como negativos difíceis."
        ),
    }
    (mixed_dataset_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    DATASET_BUILDS_DIR.mkdir(parents=True, exist_ok=True)
    pointer = {
        "build_id": build_id,
        "activated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "mixed_relative_path": str(mixed_dataset_dir.relative_to(BASE_DIR)),
        "real_eval_relative_path": str(real_eval_dir.relative_to(BASE_DIR)),
    }
    temporary_pointer = ACTIVE_DATASET_PATH.with_suffix(".json.tmp")
    temporary_pointer.write_text(
        json.dumps(pointer, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_pointer.replace(ACTIVE_DATASET_PATH)
    return manifest
