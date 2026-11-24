from pathlib import Path
import json


ROOT = Path(__file__).resolve().parent.parent
CLASSES_PATH = ROOT / "dataset" / "classes.json"


def load_class_names():
    classes = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    classes.sort(key=lambda item: int(item["id"]))
    return {int(item["id"]): str(item["key"]) for item in classes}


def create_runtime_yaml(dataset_dir: Path, output_path: Path):
    dataset_dir = Path(dataset_dir).resolve()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    names = load_class_names()
    names_text = "\n".join(
        f"  {class_id}: {name}"
        for class_id, name in names.items()
    )
    content = (
        f'path: "{dataset_dir.as_posix()}"\n'
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        f"{names_text}\n"
    )
    output_path.write_text(content, encoding="utf-8")
    return output_path
