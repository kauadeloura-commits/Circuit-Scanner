from pathlib import Path
import json
import math
import random
import shutil
from datetime import datetime

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"

CLASSES = [
    (0, "resistor", "Resistor"),
    (1, "capacitor", "Capacitor não polarizado"),
    (2, "capacitor_polarized", "Capacitor polarizado"),
    (3, "diode", "Diodo"),
    (4, "led", "LED"),
    (5, "inductor", "Indutor"),
    (6, "source_dc", "Fonte DC"),
    (7, "battery", "Bateria"),
    (8, "gnd", "GND"),
    (9, "vcc", "VCC"),
    (10, "transistor_npn", "Transistor NPN"),
    (11, "transistor_pnp", "Transistor PNP"),
    (12, "mosfet_n", "MOSFET N"),
    (13, "mosfet_p", "MOSFET P"),
    (14, "opamp", "Amplificador operacional genérico"),
]

SPLITS = ("train", "val", "test")


def _ensure_dirs(root=DATASET_DIR):
    for split in SPLITS:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    (root / "previews").mkdir(parents=True, exist_ok=True)
    (root / "meta").mkdir(parents=True, exist_ok=True)


def _line(img, mask, p1, p2, thickness=4):
    cv2.line(img, p1, p2, 20, thickness, cv2.LINE_AA)
    cv2.line(mask, p1, p2, 255, thickness + 2, cv2.LINE_AA)


def _poly(img, mask, pts, closed=False, thickness=4):
    pts = np.asarray(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], closed, 20, thickness, cv2.LINE_AA)
    cv2.polylines(mask, [pts], closed, 255, thickness + 2, cv2.LINE_AA)


def _circle(img, mask, center, radius, thickness=4):
    cv2.circle(img, center, radius, 20, thickness, cv2.LINE_AA)
    cv2.circle(mask, center, radius, 255, thickness + 2, cv2.LINE_AA)


def _arrow(img, mask, p1, p2, thickness=3):
    cv2.arrowedLine(img, p1, p2, 20, thickness, cv2.LINE_AA, tipLength=0.35)
    cv2.arrowedLine(mask, p1, p2, 255, thickness + 2, cv2.LINE_AA, tipLength=0.35)


def _draw_symbol(key, size=256, rng=None):
    rng = rng or random.Random()
    bg = rng.randint(238, 255)
    img = np.full((size, size), bg, np.uint8)
    mask = np.zeros((size, size), np.uint8)
    t = rng.choice([3, 4, 5])
    c = size // 2

    if key == "resistor":
        _line(img, mask, (28,c), (62,c), t)
        if rng.random() < 0.5:
            pts = [(62,c),(76,c-22),(92,c+22),(108,c-22),(124,c+22),(140,c-22),(156,c+22),(174,c)]
            _poly(img, mask, pts, False, t)
        else:
            cv2.rectangle(img, (68,c-28), (170,c+28), 20, t, cv2.LINE_AA)
            cv2.rectangle(mask, (68,c-28), (170,c+28), 255, t+2, cv2.LINE_AA)
        _line(img, mask, (174,c), (228,c), t)

    elif key == "capacitor":
        _line(img, mask, (28,c), (104,c), t)
        _line(img, mask, (104,c-42), (104,c+42), t)
        _line(img, mask, (150,c-42), (150,c+42), t)
        _line(img, mask, (150,c), (228,c), t)

    elif key == "capacitor_polarized":
        _line(img, mask, (28,c), (104,c), t)
        _line(img, mask, (104,c-42), (104,c+42), t)
        # curved plate
        cv2.ellipse(img, (144,c), (18,42), 0, -90, 90, 20, t, cv2.LINE_AA)
        cv2.ellipse(mask, (144,c), (18,42), 0, -90, 90, 255, t+2, cv2.LINE_AA)
        _line(img, mask, (162,c), (228,c), t)
        # plus sign
        _line(img, mask, (78,c-55), (92,c-55), max(2,t-1))
        _line(img, mask, (85,c-62), (85,c-48), max(2,t-1))

    elif key == "diode":
        _line(img, mask, (28,c), (86,c), t)
        pts = [(86,c-38),(86,c+38),(144,c)]
        if rng.random() < 0.65:
            pontos = np.asarray(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(img, [pontos], 20, cv2.LINE_AA)
            cv2.fillPoly(mask, [pontos], 255, cv2.LINE_AA)
        else:
            _poly(img, mask, pts, True, t)
        _line(img, mask, (150,c-42), (150,c+42), t)
        _line(img, mask, (150,c), (228,c), t)

    elif key == "led":
        _line(img, mask, (28,c), (86,c), t)
        pts = [(86,c-38),(86,c+38),(144,c)]
        if rng.random() < 0.50:
            pontos = np.asarray(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(img, [pontos], 20, cv2.LINE_AA)
            cv2.fillPoly(mask, [pontos], 255, cv2.LINE_AA)
        else:
            _poly(img, mask, pts, True, t)
        _line(img, mask, (150,c-42), (150,c+42), t)
        _line(img, mask, (150,c), (228,c), t)
        _arrow(img, mask, (142,c-48), (172,c-78), max(2,t-1))
        _arrow(img, mask, (154,c-30), (184,c-60), max(2,t-1))

    elif key == "inductor":
        _line(img, mask, (28,c), (62,c), t)
        x = 62
        for _ in range(4):
            cv2.ellipse(img, (x+18,c), (18,30), 0, 180, 360, 20, t, cv2.LINE_AA)
            cv2.ellipse(mask, (x+18,c), (18,30), 0, 180, 360, 255, t+2, cv2.LINE_AA)
            x += 36
        _line(img, mask, (206,c), (228,c), t)

    elif key == "source_dc":
        _line(img, mask, (28,c), (78,c), t)
        _circle(img, mask, (128,c), 48, t)
        _line(img, mask, (178,c), (228,c), t)
        _line(img, mask, (113,c-16), (143,c-16), max(2,t-1))
        _line(img, mask, (128,c-31), (128,c-1), max(2,t-1))
        _line(img, mask, (113,c+20), (143,c+20), max(2,t-1))

    elif key == "battery":
        _line(img, mask, (28,c), (92,c), t)
        _line(img, mask, (96,c-46), (96,c+46), t)
        _line(img, mask, (126,c-28), (126,c+28), t)
        _line(img, mask, (150,c-46), (150,c+46), t)
        _line(img, mask, (180,c-28), (180,c+28), t)
        _line(img, mask, (180,c), (228,c), t)

    elif key == "gnd":
        _line(img, mask, (c,30), (c,142), t)
        _line(img, mask, (88,150), (168,150), t)
        _line(img, mask, (100,166), (156,166), t)
        _line(img, mask, (114,182), (142,182), t)

    elif key == "vcc":
        _line(img, mask, (c,226), (c,96), t)
        _poly(img, mask, [(102,96),(128,58),(154,96)], False, t)

    elif key in ("transistor_npn", "transistor_pnp"):
        _circle(img, mask, (132,c), 70, t)
        _line(img, mask, (30,c), (102,c), t)
        _line(img, mask, (104,c-48), (104,c+48), t)
        _line(img, mask, (104,c-30), (166,c-72), t)
        _line(img, mask, (104,c+30), (166,c+72), t)
        _line(img, mask, (166,c-72), (210,c-72), t)
        _line(img, mask, (166,c+72), (210,c+72), t)
        if key == "transistor_npn":
            _arrow(img, mask, (132,c+49), (159,c+68), max(2,t-1))
        else:
            _arrow(img, mask, (159,c+68), (132,c+49), max(2,t-1))

    elif key in ("mosfet_n", "mosfet_p"):
        # generic enhancement MOSFET-style symbol
        _line(img, mask, (40,c), (94,c), t)
        _line(img, mask, (94,c-58), (94,c+58), t)
        _line(img, mask, (122,c-48), (122,c+48), t)
        _line(img, mask, (122,c-36), (176,c-74), t)
        _line(img, mask, (122,c+36), (176,c+74), t)
        _line(img, mask, (176,c-74), (220,c-74), t)
        _line(img, mask, (176,c+74), (220,c+74), t)
        if key == "mosfet_n":
            _arrow(img, mask, (158,c), (128,c), max(2,t-1))
        else:
            _arrow(img, mask, (128,c), (158,c), max(2,t-1))

    elif key == "opamp":
        pts = [(76,54),(76,202),(196,c)]
        _poly(img, mask, pts, True, t)
        _line(img, mask, (28,92), (76,92), t)
        _line(img, mask, (28,164), (76,164), t)
        _line(img, mask, (196,c), (228,c), t)
        _line(img, mask, (48,82), (62,82), max(2,t-1))
        _line(img, mask, (55,75), (55,89), max(2,t-1))
        _line(img, mask, (48,154), (62,154), max(2,t-1))

    else:
        raise ValueError(f"Classe desconhecida: {key}")

    return img, mask


def _transform(img, mask, rng, key=None):
    h, w = img.shape[:2]
    # Componentes podem aparecer horizontalmente, verticalmente ou espelhados
    # em esquemas digitais. A versão anterior usava apenas +/-12 graus e não
    # ensinava, por exemplo, o resistor vertical visto nos diagramas impressos.
    cardinais = (0, 90, 180, 270)
    base_angle = 0 if key in {"vcc", "gnd"} else rng.choice(cardinais)
    angle = base_angle + rng.uniform(-12, 12)
    scale = rng.uniform(0.72, 1.08)
    tx = rng.uniform(-12, 12)
    ty = rng.uniform(-12, 12)
    M = cv2.getRotationMatrix2D((w/2,h/2), angle, scale)
    M[0,2] += tx
    M[1,2] += ty
    bg = int(np.median(img))
    img = cv2.warpAffine(img, M, (w,h), flags=cv2.INTER_LINEAR, borderValue=bg)
    mask = cv2.warpAffine(mask, M, (w,h), flags=cv2.INTER_NEAREST, borderValue=0)

    # light perspective deformation
    if rng.random() < 0.45:
        d = rng.uniform(2, 9)
        src = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
        dst = np.float32([
            [rng.uniform(0,d),rng.uniform(0,d)],
            [w-1-rng.uniform(0,d),rng.uniform(0,d)],
            [w-1-rng.uniform(0,d),h-1-rng.uniform(0,d)],
            [rng.uniform(0,d),h-1-rng.uniform(0,d)],
        ])
        P = cv2.getPerspectiveTransform(src,dst)
        img = cv2.warpPerspective(img,P,(w,h),borderValue=bg)
        mask = cv2.warpPerspective(mask,P,(w,h),flags=cv2.INTER_NEAREST,borderValue=0)

    # uneven illumination
    yy, xx = np.mgrid[0:h, 0:w]
    gx = rng.uniform(-14,14) * (xx / max(1,w-1) - .5)
    gy = rng.uniform(-14,14) * (yy / max(1,h-1) - .5)
    img = np.clip(img.astype(np.float32) + gx + gy + rng.uniform(-6,6), 0, 255).astype(np.uint8)

    # contrast
    alpha = rng.uniform(0.82, 1.22)
    beta = rng.uniform(-8, 8)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    if rng.random() < 0.82:
        paletas = [
            ((255, 255, 255), (25, 25, 25)),
            ((250, 250, 245), (65, 125, 70)),
            ((246, 252, 255), (145, 75, 20)),
            ((235, 245, 228), (85, 40, 135)),
            ((35, 38, 42), (238, 238, 225)),
            ((30, 55, 70), (225, 245, 245)),
        ]
        fundo, tinta = rng.choice(paletas)
        alpha_tinta = (mask.astype(np.float32) / 255.0)[..., None]
        img = (
            np.asarray(fundo, dtype=np.float32) * (1.0 - alpha_tinta)
            + np.asarray(tinta, dtype=np.float32) * alpha_tinta
        ).astype(np.uint8)

    # Aproximadamente 40% do treino passa a imitar diretamente diagramas
    # preto/branco vindos de PDF, scan e impressão. Isso reduz a diferença de
    # domínio entre as imagens sintéticas coloridas e esquemas binários reais.
    if rng.random() < 0.40:
        tinta = rng.choice([0, 8, 16, 24])
        fundo = rng.choice([248, 252, 255])
        cobertura = mask.astype(np.float32) / 255.0
        if img.ndim == 3:
            cobertura = cobertura[..., None]
        img = (fundo * (1.0 - cobertura) + tinta * cobertura).astype(np.uint8)

    # blur
    if rng.random() < 0.45:
        k = rng.choice([3,3,5])
        img = cv2.GaussianBlur(img,(k,k),rng.uniform(0.2,1.1))

    # gaussian noise
    if rng.random() < 0.7:
        sigma = rng.uniform(1.5, 9.0)
        noise = np.random.default_rng(rng.randrange(1, 2**32-1)).normal(0, sigma, img.shape)
        img = np.clip(img.astype(np.float32)+noise,0,255).astype(np.uint8)

    return img, mask


SCHEMATIC_LABELS = {
    "resistor": ("R1", "R_B", "R_C", "10k", "RES"),
    "capacitor": ("C1", "100nF", "C_B", "CAP"),
    "capacitor_polarized": ("C2", "10uF", "+", "CAP"),
    "diode": ("D1", "1N4148", "DIODE"),
    "led": ("LED", "LED1", "D2"),
    "inductor": ("L1", "10mH", "IND"),
    "source_dc": ("V1", "5V", "VDC"),
    "battery": ("BAT", "9V", "B1"),
    "gnd": ("GND", "0V"),
    "vcc": ("VCC", "+5V", "+12V"),
    "transistor_npn": ("NPN", "Q1", "BC547", "B", "C", "E"),
    "transistor_pnp": ("PNP", "Q2", "BC557", "B", "C", "E"),
    "mosfet_n": ("MOSFET N", "M1", "G", "D", "S"),
    "mosfet_p": ("MOSFET P", "M2", "G", "D", "S"),
    "opamp": ("U1", "LM358", "OPAMP", "+", "-"),
}


def _add_schematic_label_clutter(img, key, rng):
    """Adiciona texto próximo sem incluí-lo na máscara do componente."""
    if rng.random() >= 0.78:
        return img
    out = img.copy()
    h, w = out.shape[:2]
    tokens = SCHEMATIC_LABELS.get(key, (key.upper(),))
    amount = 1 if rng.random() < 0.72 else 2
    positions = (
        (int(0.62 * w), int(0.27 * h)),
        (int(0.64 * w), int(0.73 * h)),
        (int(0.05 * w), int(0.20 * h)),
        (int(0.05 * w), int(0.86 * h)),
    )
    chosen = list(positions)
    rng.shuffle(chosen)
    for index in range(amount):
        text = rng.choice(tokens)
        scale = rng.uniform(0.55, 0.90)
        thickness = rng.choice((1, 1, 2))
        x, y = chosen[index]
        cv2.putText(
            out,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            rng.randint(5, 45),
            thickness,
            cv2.LINE_AA,
        )
    return out



def _draw_hard_negative(size=256, rng=None):
    """Gera padrões que NÃO são componentes: fios, nós, T, X e diagonais.

    Esses exemplos usam label YOLO vazio. O objetivo é reduzir falsos positivos
    como nó de conexão -> VCC e linha diagonal -> transistor.
    """
    rng = rng or random.Random()
    bg = rng.randint(245, 255)
    img = np.full((size, size, 3), bg, dtype=np.uint8)
    ink = rng.choice([(0, 0, 0), (15, 15, 15), (30, 30, 30)])
    t = rng.choice([2, 3, 4, 5])
    c = size // 2
    kind = rng.choice([
        "dot", "junction_dot", "t_junction", "x_crossing",
        "single_diagonal", "parallel_diagonals", "v_shape", "wire_corner",
    ])

    if kind == "dot":
        cv2.circle(img, (c, c), rng.randint(4, 10), ink, -1, cv2.LINE_AA)
    elif kind == "junction_dot":
        cv2.line(img, (25, c), (231, c), ink, t, cv2.LINE_AA)
        cv2.line(img, (c, 25), (c, 231), ink, t, cv2.LINE_AA)
        cv2.circle(img, (c, c), rng.randint(4, 9), ink, -1, cv2.LINE_AA)
    elif kind == "t_junction":
        cv2.line(img, (35, c), (221, c), ink, t, cv2.LINE_AA)
        cv2.line(img, (c, c), (c, 225), ink, t, cv2.LINE_AA)
    elif kind == "x_crossing":
        cv2.line(img, (35, 35), (221, 221), ink, t, cv2.LINE_AA)
        cv2.line(img, (221, 35), (35, 221), ink, t, cv2.LINE_AA)
    elif kind == "single_diagonal":
        if rng.random() < 0.5:
            cv2.line(img, (30, 220), (225, 35), ink, t, cv2.LINE_AA)
        else:
            cv2.line(img, (30, 35), (225, 220), ink, t, cv2.LINE_AA)
    elif kind == "parallel_diagonals":
        cv2.line(img, (25, 205), (205, 25), ink, t, cv2.LINE_AA)
        cv2.line(img, (50, 230), (230, 50), ink, t, cv2.LINE_AA)
    elif kind == "v_shape":
        cv2.line(img, (55, 65), (c, 185), ink, t, cv2.LINE_AA)
        cv2.line(img, (201, 65), (c, 185), ink, t, cv2.LINE_AA)
    else:  # wire_corner
        cv2.line(img, (30, c), (c, c), ink, t, cv2.LINE_AA)
        cv2.line(img, (c, c), (c, 35), ink, t, cv2.LINE_AA)

    # Pequenas variações fotográficas sem transformar o padrão em componente.
    if rng.random() < 0.45:
        img = cv2.GaussianBlur(img, (3, 3), rng.uniform(0.2, 0.8))
    if rng.random() < 0.55:
        sigma = rng.uniform(1.0, 5.0)
        noise = np.random.default_rng(rng.randrange(1, 2**32 - 1)).normal(
            0, sigma, img.shape
        )
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img, kind

def _bbox_from_mask(mask):
    ys, xs = np.where(mask > 30)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    pad = 5
    h,w = mask.shape[:2]
    x1=max(0,x1-pad); y1=max(0,y1-pad); x2=min(w-1,x2+pad); y2=min(h-1,y2+pad)
    return x1,y1,x2,y2


def _to_yolo(box, width, height):
    x1,y1,x2,y2 = box
    cx=((x1+x2)/2)/width
    cy=((y1+y2)/2)/height
    bw=(x2-x1)/width
    bh=(y2-y1)/height
    return cx,cy,bw,bh


def _save_jpeg(path, image, rng):
    quality = rng.randint(72, 98)
    ok, buf = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError('Falha ao codificar imagem JPEG.')
    buf.tofile(str(path))


def generate_dataset(per_class=60, seed=4411, root=DATASET_DIR, clear=True):
    if per_class < 3:
        raise ValueError('Use pelo menos 3 imagens por classe.')
    if per_class > 500:
        raise ValueError('Limite desta etapa: 500 imagens por classe por geração.')

    root = Path(root)
    _ensure_dirs(root)

    if clear:
        for split in SPLITS:
            for folder in (root/'images'/split, root/'labels'/split):
                if folder.exists():
                    for p in folder.iterdir():
                        if p.is_file(): p.unlink()

    rng = random.Random(seed)
    counts = {s:0 for s in SPLITS}
    per_class_counts = {}

    # approximately 70/20/10, guaranteeing at least one per split
    n_train = max(1, round(per_class*0.70))
    n_val = max(1, round(per_class*0.20))
    n_test = max(1, per_class - n_train - n_val)
    if n_train+n_val+n_test > per_class:
        n_train = per_class - n_val - n_test

    plan = [('train',n_train),('val',n_val),('test',n_test)]

    for class_id,key,name in CLASSES:
        per_class_counts[key] = {s:0 for s in SPLITS}
        sample_index = 0
        for split, amount in plan:
            for _ in range(amount):
                local_seed = rng.randrange(1, 2**31-1)
                local = random.Random(local_seed)
                img, mask = _draw_symbol(key, 256, local)
                img, mask = _transform(img, mask, local, key=key)
                img = _add_schematic_label_clutter(img, key, local)
                box = _bbox_from_mask(mask)
                if box is None:
                    continue

                filename = f'{key}_{sample_index:05d}'
                image_path = root/'images'/split/f'{filename}.jpg'
                label_path = root/'labels'/split/f'{filename}.txt'
                _save_jpeg(image_path, img, local)
                cx,cy,bw,bh = _to_yolo(box, 256, 256)
                label_path.write_text(
                    f'{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n',
                    encoding='utf-8'
                )
                counts[split] += 1
                per_class_counts[key][split] += 1
                sample_index += 1

    # Hard negatives: imagens sem nenhum componente/caixa. O YOLO aceita
    # arquivos .txt vazios e aprende que essas geometrias são apenas fios/nós.
    negative_total = max(60, int(per_class) * 2)
    neg_train = max(1, round(negative_total * 0.70))
    neg_val = max(1, round(negative_total * 0.20))
    neg_test = max(1, negative_total - neg_train - neg_val)
    negative_counts = {s: 0 for s in SPLITS}
    negative_patterns = {}
    neg_index = 0
    for split, amount in [('train', neg_train), ('val', neg_val), ('test', neg_test)]:
        for _ in range(amount):
            local = random.Random(rng.randrange(1, 2**31 - 1))
            img, kind = _draw_hard_negative(256, local)
            filename = f'hard_negative_{neg_index:05d}'
            _save_jpeg(root/'images'/split/f'{filename}.jpg', img, local)
            (root/'labels'/split/f'{filename}.txt').write_text('', encoding='utf-8')
            negative_counts[split] += 1
            negative_patterns[kind] = negative_patterns.get(kind, 0) + 1
            counts[split] += 1
            neg_index += 1

    manifest = {
        'generated_at': datetime.now().astimezone().isoformat(timespec='seconds'),
        'seed': seed,
        'per_class_requested': per_class,
        'classes': [{'id':i,'key':k,'name':n} for i,k,n in CLASSES],
        'counts': counts,
        'total': sum(counts.values()),
        'per_class': per_class_counts,
        'hard_negatives': {
            'total': int(sum(negative_counts.values())),
            'counts': negative_counts,
            'patterns': negative_patterns,
            'label_policy': 'empty YOLO label file',
        },
        'format': 'YOLO detection labels: class cx cy width height (normalized)',
        'image_size': [256,256],
    }
    (root/'meta'/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest


def generate_previews(root=DATASET_DIR, seed=1234):
    root = Path(root)
    _ensure_dirs(root)
    rng = random.Random(seed)
    cards=[]
    for class_id,key,name in CLASSES:
        img,mask = _draw_symbol(key,256,rng)
        box=_bbox_from_mask(mask)
        if box:
            x1,y1,x2,y2=box
            cv2.rectangle(img,(x1,y1),(x2,y2),100,1,cv2.LINE_AA)
        # label bottom
        canvas=np.full((300,256),255,np.uint8)
        canvas[:256]=img
        cv2.putText(canvas, f'{class_id}: {key}', (8,283), cv2.FONT_HERSHEY_SIMPLEX, .52, 30, 1, cv2.LINE_AA)
        p=root/'previews'/f'{class_id:02d}_{key}.png'
        ok,buf=cv2.imencode('.png',canvas)
        if ok: buf.tofile(str(p))
        cards.append(canvas)

    cols=3
    rows=math.ceil(len(cards)/cols)
    grid=np.full((rows*300,cols*256),245,np.uint8)
    for idx,card in enumerate(cards):
        r=idx//cols; c=idx%cols
        grid[r*300:(r+1)*300,c*256:(c+1)*256]=card
    ok,buf=cv2.imencode('.png',grid)
    if ok: buf.tofile(str(root/'previews'/'catalog.png'))
    return len(cards)


def dataset_info(root=DATASET_DIR):
    root=Path(root)
    result={
        'classes':[{'id':i,'key':k,'name':n} for i,k,n in CLASSES],
        'class_count':len(CLASSES),
        'counts':{},
        'total':0,
        'manifest':None,
        'catalog_url':'http://127.0.0.1:8010/dataset-files/previews/catalog.png',
    }
    for split in SPLITS:
        count=len(list((root/'images'/split).glob('*.*'))) if (root/'images'/split).exists() else 0
        result['counts'][split]=count
        result['total']+=count
    manifest_path=root/'meta'/'manifest.json'
    if manifest_path.exists():
        try: result['manifest']=json.loads(manifest_path.read_text(encoding='utf-8'))
        except Exception: pass
    return result
