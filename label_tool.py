from pathlib import Path
import sys

import cv2


ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
WINDOW_NAME = "YOLO Labeling - split_wire"


class Labeler:
    def __init__(self, image_dir: Path, label_dir: Path):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.label_dir.mkdir(parents=True, exist_ok=True)
        self.images = sorted(
            path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        self.index = 0
        self.image = None
        self.display = None
        self.boxes = []
        self.drawing = False
        self.start = None

    def load(self):
        if not self.images:
            raise RuntimeError(f"Tidak ada gambar di {self.image_dir}")
        self.image = cv2.imread(str(self.images[self.index]))
        if self.image is None:
            raise RuntimeError(f"Gambar tidak bisa dibaca: {self.images[self.index]}")
        self.boxes = self.read_labels()
        self.render()

    def read_labels(self):
        label_path = self.label_dir / f"{self.images[self.index].stem}.txt"
        if not label_path.exists() or self.image is None:
            return []
        height, width = self.image.shape[:2]
        boxes = []
        for line in label_path.read_text().splitlines():
            values = line.split()
            if len(values) != 5 or values[0] != "0":
                continue
            _, center_x, center_y, box_width, box_height = map(float, values)
            x1 = int((center_x - box_width / 2) * width)
            y1 = int((center_y - box_height / 2) * height)
            x2 = int((center_x + box_width / 2) * width)
            y2 = int((center_y + box_height / 2) * height)
            boxes.append((max(0, x1), max(0, y1), min(width - 1, x2), min(height - 1, y2)))
        return boxes

    def render(self):
        self.display = self.image.copy()
        for x1, y1, x2, y2 in self.boxes:
            cv2.rectangle(self.display, (x1, y1), (x2, y2), (0, 220, 0), 2)
        title = f"{self.index + 1}/{len(self.images)}  {self.images[self.index].name}  boxes: {len(self.boxes)}"
        cv2.putText(self.display, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
        cv2.imshow(WINDOW_NAME, self.display)

    def save(self):
        height, width = self.image.shape[:2]
        lines = []
        for x1, y1, x2, y2 in self.boxes:
            center_x = ((x1 + x2) / 2) / width
            center_y = ((y1 + y2) / 2) / height
            box_width = (x2 - x1) / width
            box_height = (y2 - y1) / height
            lines.append(f"0 {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}")
        label_path = self.label_dir / f"{self.images[self.index].stem}.txt"
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        print(f"Tersimpan: {label_path}")

    def mouse(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.render()
            cv2.rectangle(self.display, self.start, (x, y), (0, 180, 255), 2)
            cv2.imshow(WINDOW_NAME, self.display)
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            x1, x2 = sorted((self.start[0], x))
            y1, y2 = sorted((self.start[1], y))
            if x2 - x1 > 3 and y2 - y1 > 3:
                self.boxes.append((x1, y1, x2, y2))
            self.render()

    def run(self):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self.mouse)
        self.load()
        while True:
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("s"), ord("S")):
                self.save()
            elif key in (ord("n"), ord("N")) and self.index < len(self.images) - 1:
                self.save()
                self.index += 1
                self.load()
            elif key in (ord("p"), ord("P")) and self.index > 0:
                self.save()
                self.index -= 1
                self.load()
            elif key in (ord("u"), ord("U")) and self.boxes:
                self.boxes.pop()
                self.render()
            elif key == 27 or key in (ord("q"), ord("Q")):
                break
        cv2.destroyAllWindows()


def main():
    split = sys.argv[1].lower() if len(sys.argv) > 1 else "train"
    if split not in {"train", "val"}:
        raise SystemExit("Pakai: python label_tool.py [train|val]")

    dataset_dir = ROOT / "data" / "dataset"
    current_dirs = {
        "train": (dataset_dir / "train" / "images", dataset_dir / "train" / "labels"),
        "val": (dataset_dir / "validation" / "images", dataset_dir / "validation" / "labels"),
    }
    legacy_dirs = {
        "train": (dataset_dir / "images" / "train", dataset_dir / "labels" / "train"),
        "val": (dataset_dir / "images" / "val", dataset_dir / "labels" / "val"),
    }

    image_dir, label_dir = current_dirs[split]
    if not image_dir.exists() and legacy_dirs[split][0].exists():
        image_dir, label_dir = legacy_dirs[split]
    if not image_dir.exists():
        raise SystemExit(f"Folder gambar tidak ditemukan: {image_dir}")

    Labeler(image_dir, label_dir).run()


if __name__ == "__main__":
    main()
