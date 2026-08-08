"""Buat label YOLO awal dari prediksi model untuk foto yang belum dilabeli.

Contoh:
    python3 auto_label.py --model models/best.pt \
        --images data/unlabeled/images --labels data/unlabeled/labels
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Auto-label gambar dengan model YOLO")
    parser.add_argument("--model", default="models/best.pt", help="Lokasi model YOLO")
    parser.add_argument("--images", required=True, help="Folder gambar tanpa label")
    parser.add_argument("--labels", required=True, help="Folder output label YOLO")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence prediksi")
    parser.add_argument("--imgsz", type=int, default=640, help="Ukuran gambar inferensi")
    parser.add_argument("--device", default=None, help="Contoh: 0, cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    image_dir = Path(args.images)
    label_dir = Path(args.labels)
    label_dir.mkdir(parents=True, exist_ok=True)

    if not image_dir.exists():
        raise SystemExit(f"Folder gambar tidak ditemukan: {image_dir}")

    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise SystemExit(f"Tidak ada gambar di: {image_dir}")

    model = YOLO(args.model)
    predict_args = {
        "source": [str(path) for path in image_paths],
        "conf": args.conf,
        "imgsz": args.imgsz,
        "save": False,
        "verbose": False,
    }
    if args.device:
        predict_args["device"] = args.device

    results = model.predict(**predict_args)
    total_boxes = 0

    for image_path, result in zip(image_paths, results):
        lines = []
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                x_center, y_center, width, height = box.xywhn[0].tolist()
                lines.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} "
                    f"{width:.6f} {height:.6f}"
                )

        output_path = label_dir / f"{image_path.stem}.txt"
        output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        total_boxes += len(lines)
        print(f"{image_path.name}: {len(lines)} objek -> {output_path}")

    print(f"\nSelesai: {len(image_paths)} gambar, {total_boxes} prediksi label dibuat.")
    print("Periksa dan koreksi semua file .txt sebelum training ulang.")


if __name__ == "__main__":
    main()
