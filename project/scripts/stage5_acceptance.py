from datetime import datetime
from pathlib import Path
import json
import sys

import cv2


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.recognition import detectar_componentes


def intersection_over_union(a, b):
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a["x2"] - a["x1"]) * max(
        0.0, a["y2"] - a["y1"]
    )
    area_b = max(0.0, b["x2"] - b["x1"]) * max(
        0.0, b["y2"] - b["y1"]
    )
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def load_ground_truth(image_path, names):
    split = image_path.parent.name
    label_path = (
        ROOT
        / "dataset_real_eval"
        / "labels"
        / split
        / f"{image_path.stem}.txt"
    )
    image = cv2.imread(str(image_path))
    height, width = image.shape[:2]
    ground_truth = []

    for line in label_path.read_text(encoding="utf-8").splitlines():
        class_id, cx, cy, box_width, box_height = map(float, line.split())
        ground_truth.append(
            {
                "class_id": int(class_id),
                "class": names[int(class_id)],
                "bbox": {
                    "x1": (cx - box_width / 2.0) * width,
                    "y1": (cy - box_height / 2.0) * height,
                    "x2": (cx + box_width / 2.0) * width,
                    "y2": (cy + box_height / 2.0) * height,
                },
            }
        )
    return ground_truth


def compare(ground_truth, detections):
    candidates = []
    for prediction_index, prediction in enumerate(detections):
        for truth_index, truth in enumerate(ground_truth):
            candidates.append(
                (
                    intersection_over_union(
                        prediction["bbox"],
                        truth["bbox"],
                    ),
                    prediction_index,
                    truth_index,
                )
            )

    used_predictions = set()
    used_truth = set()
    correct = []
    wrong_class = []

    for iou, prediction_index, truth_index in sorted(
        candidates,
        reverse=True,
    ):
        if iou < 0.50:
            break
        if prediction_index in used_predictions or truth_index in used_truth:
            continue
        used_predictions.add(prediction_index)
        used_truth.add(truth_index)
        prediction = detections[prediction_index]
        truth = ground_truth[truth_index]
        item = {
            "expected": truth["class"],
            "predicted": prediction["class"],
            "confidence": prediction["confidence"],
            "iou": round(iou, 4),
        }
        if prediction["class_id"] == truth["class_id"]:
            correct.append(item)
        else:
            wrong_class.append(item)

    missed = [
        truth["class"]
        for index, truth in enumerate(ground_truth)
        if index not in used_truth
    ]
    false_positives = [
        prediction["class"]
        for index, prediction in enumerate(detections)
        if index not in used_predictions
    ]
    return {
        "real_components": len(ground_truth),
        "predicted_boxes": len(detections),
        "correct_class": len(correct),
        "wrong_class": len(wrong_class),
        "missed": len(missed),
        "false_positive": len(false_positives),
        "correct_details": correct,
        "wrong_class_details": wrong_class,
        "missed_classes": missed,
        "false_positive_classes": false_positives,
    }


def main():
    names = {
        int(item["id"]): item["key"]
        for item in json.loads(
            (ROOT / "dataset" / "classes.json").read_text(encoding="utf-8")
        )
    }
    test_images = sorted(
        (ROOT / "dataset_real_eval" / "images" / "test").glob("*.*")
    )
    validation_images = sorted(
        (ROOT / "dataset_real_eval" / "images" / "val").glob("*.*")
    )
    selected = test_images + validation_images[: max(0, 5 - len(test_images))]
    if len(selected) < 5:
        raise SystemExit("São necessárias cinco imagens reais de avaliação.")

    output_dir = ROOT / "models" / "stage5_acceptance"
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = []

    for image_path in selected[:5]:
        result = detectar_componentes(
            image_path=image_path,
            output_image_path=output_dir / f"{image_path.stem}_detections.png",
            output_json_path=output_dir / f"{image_path.stem}_detections.json",
            confidence=0.25,
            imgsz=640,
            strategy="auto",
        )
        ground_truth = load_ground_truth(image_path, names)
        comparison = compare(ground_truth, result["detections"])
        cases.append(
            {
                "image": str(image_path.relative_to(ROOT)),
                "split": image_path.parent.name,
                **comparison,
            }
        )

    totals = {
        key: sum(case[key] for case in cases)
        for key in (
            "real_components",
            "predicted_boxes",
            "correct_class",
            "wrong_class",
            "missed",
            "false_positive",
        )
    }
    totals["correct_class_rate"] = round(
        totals["correct_class"] / totals["real_components"],
        4,
    )
    payload = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "criteria": "IoU >= 0.50; five real images not used in training",
        "cases": cases,
        "totals": totals,
    }
    output_path = output_dir / "acceptance_summary.json"
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("\nResumo salvo em:", output_path)


if __name__ == "__main__":
    main()
