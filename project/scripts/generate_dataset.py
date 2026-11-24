from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.dataset_generator import generate_dataset, generate_previews


def main():
    per_class = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    print(f"Gerando {per_class} imagens por classe...")
    manifest = generate_dataset(per_class=per_class, seed=4411, clear=True)
    generate_previews()
    print("Concluído.")
    print("Treino:", manifest["counts"]["train"])
    print("Validação:", manifest["counts"]["val"])
    print("Teste:", manifest["counts"]["test"])
    print("Total:", manifest["total"])


if __name__ == "__main__":
    main()
