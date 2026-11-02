"""A. Dataset
Construye data/mvtec_128.zip a partir de los .tar.xz originales de MVTec AD.

Lee los archivos sin extraerlos, redimensiona cada imagen a 128x128 RGB y
conserva la estructura <clase>/train/good, <clase>/test/<tipo>. Se omite
ground_truth (mascaras) porque la tarea no las usa.

Uso:
    python scripts/prepare_data.py --src C:/Users/<user>/Downloads
"""

import argparse
import io
import tarfile
import zipfile
from pathlib import Path

from PIL import Image

CLASSES = ["cable", "capsule", "screw", "transistor"]
SIZE = 128


def process_class(tar_path: Path, zf: zipfile.ZipFile) -> int:
    count = 0
    with tarfile.open(tar_path, "r:xz") as tar:
        for member in tar:
            parts = Path(member.name).parts
            # <clase>/<train|test>/<tipo>/<img>.png
            if (
                not member.isfile()
                or not member.name.endswith(".png")
                or len(parts) != 4
                or parts[1] not in ("train", "test")
            ):
                continue
            data = tar.extractfile(member).read()
            img = Image.open(io.BytesIO(data)).convert("RGB")
            img = img.resize((SIZE, SIZE), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            zf.writestr("/".join(parts), buf.getvalue())
            count += 1
    return count


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--out", type=Path, default=repo / "data" / "mvtec_128.zip")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_STORED) as zf:
        for cls in CLASSES:
            n = process_class(args.src / f"{cls}.tar.xz", zf)
            print(f"{cls}: {n} imagenes")
    print(f"Listo: {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
