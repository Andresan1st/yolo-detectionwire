"""Cek hasil label YOLO tanpa layar (cocok untuk server).

- Menggambar box label di setiap foto, digabung jadi lembar kolase (grid)
  yang bisa dibuka/didownload untuk dicek sekilas.
- Mencetak ringkasan: label kosong, koordinat di luar batas, box terlalu kecil,
  dan (jika nama file berisi jumlah, mis. "5kwt_orig.jpg") selisih jumlah box.

Contoh:
    python preview_labels.py --images data/unlabeled/images --labels data/unlabeled/labels
    python preview_labels.py --images data/unlabeled/images --labels data/unlabeled/labels --only-problems
"""

import argparse
import re
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# "5kwt_orig", "2ktw_blur" -> jumlah kawat yang diharapkan
EXPECTED_COUNT_RE = re.compile(r"^(\d+)k(?:wt|tw)", re.IGNORECASE)
TILE = 480  # ukuran tiap foto di kolase


def parse_args():
    parser = argparse.ArgumentParser(description="Preview & cek label YOLO")
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", default="label_preview", help="Folder output kolase")
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--min-size", type=float, default=0.005,
                        help="Box dengan lebar/tinggi (ternormalisasi) di bawah ini dianggap mencurigakan")
    parser.add_argument("--only-problems", action="store_true",
                        help="Hanya masukkan foto bermasalah ke kolase")
    return parser.parse_args()


def read_boxes(label_path: Path):
    if not label_path.exists():
        return None
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 5:
            boxes.append((int(float(parts[0])), *map(float, parts[1:])))
    return boxes


def check(image_path: Path, boxes, min_size):
    problems = []
    if boxes is None:
        return ["label tidak ada"]
    if not boxes:
        problems.append("label kosong")
    for _, cx, cy, w, h in boxes:
        if not (0 <= cx - w / 2 and cx + w / 2 <= 1.0001 and 0 <= cy - h / 2 and cy + h / 2 <= 1.0001):
            problems.append("box keluar batas")
            break
    if any(w < min_size or h < min_size for _, _, _, w, h in boxes):
        problems.append("box sangat kecil")
    match = EXPECTED_COUNT_RE.match(image_path.stem)
    if match and int(match.group(1)) != len(boxes):
        problems.append(f"jumlah {len(boxes)}, harusnya {match.group(1)}")
    return problems


def draw(image, boxes, title, problems):
    height, width = image.shape[:2]
    for _, cx, cy, w, h in boxes or []:
        x1, y1 = int((cx - w / 2) * width), int((cy - h / 2) * height)
        x2, y2 = int((cx + w / 2) * width), int((cy + h / 2) * height)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 230, 0), max(2, width // 300))

    scale = TILE / max(height, width)
    image = cv2.resize(image, (int(width * scale), int(height * scale)))
    tile = np.full((TILE, TILE, 3), 30, np.uint8)
    y0, x0 = (TILE - image.shape[0]) // 2, (TILE - image.shape[1]) // 2
    tile[y0:y0 + image.shape[0], x0:x0 + image.shape[1]] = image

    color = (0, 0, 255) if problems else (255, 255, 255)
    lines = [f"{title[:40]}  [{len(boxes or [])} box]"] + problems
    for i, text in enumerate(lines):
        y = 22 + i * 22
        cv2.putText(tile, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(tile, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return tile


def main():
    args = parse_args()
    image_dir, label_dir, out_dir = Path(args.images), Path(args.labels), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("sheet_*.jpg"):
        old.unlink()

    image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        raise SystemExit(f"Tidak ada gambar di {image_dir}")

    tiles, problem_list, total_boxes = [], [], 0
    for path in image_paths:
        boxes = read_boxes(label_dir / f"{path.stem}.txt")
        problems = check(path, boxes, args.min_size)
        total_boxes += len(boxes or [])
        if problems:
            problem_list.append((path.name, problems))
        if args.only_problems and not problems:
            continue
        image = cv2.imread(str(path))
        if image is None:
            problem_list.append((path.name, ["gambar tidak bisa dibaca"]))
            continue
        tiles.append(draw(image, boxes, path.name, problems))

    per_sheet = args.cols * args.rows
    blank = np.full((TILE, TILE, 3), 30, np.uint8)
    sheets = 0
    for start in range(0, len(tiles), per_sheet):
        chunk = tiles[start:start + per_sheet]
        chunk += [blank] * (per_sheet - len(chunk))
        rows = [np.hstack(chunk[r * args.cols:(r + 1) * args.cols]) for r in range(args.rows)]
        sheets += 1
        cv2.imwrite(str(out_dir / f"sheet_{sheets:03d}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])

    report = out_dir / "problems.txt"
    report.write_text("".join(f"{name}\t{'; '.join(p)}\n" for name, p in problem_list), encoding="utf-8")

    print(f"Gambar       : {len(image_paths)}")
    print(f"Total box    : {total_boxes}  (rata-rata {total_boxes / len(image_paths):.1f} per gambar)")
    print(f"Bermasalah   : {len(problem_list)}  -> daftar di {report}")
    for name, problems in problem_list[:20]:
        print(f"  - {name}: {'; '.join(problems)}")
    if len(problem_list) > 20:
        print(f"  ... dan {len(problem_list) - 20} lainnya (lihat {report})")
    print(f"Kolase       : {sheets} file di {out_dir}/sheet_*.jpg")


if __name__ == "__main__":
    main()
