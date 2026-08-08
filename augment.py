"""
Auto-Augmentation Script untuk YOLO Dataset
Menghasilkan 8 variasi dari setiap gambar + otomatis update label
"""

import cv2
import numpy as np
from pathlib import Path
import random
import shutil
import argparse

class YOLOAugmentor:
    def __init__(self, image_input_dir, label_input_dir, image_output_dir, label_output_dir, num_variations=8):
        self.image_input_dir = Path(image_input_dir)
        self.label_input_dir = Path(label_input_dir)
        self.image_output_dir = Path(image_output_dir)
        self.label_output_dir = Path(label_output_dir)
        self.num_variations = num_variations

        # Extensions yang didukung
        self.image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

        # Buat direktori output
        self.image_output_dir.mkdir(parents=True, exist_ok=True)
        self.label_output_dir.mkdir(parents=True, exist_ok=True)

    def load_yolo_labels(self, label_path):
        """Load label YOLO format"""
        if not label_path.exists():
            return []
        boxes = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, cx, cy, w, h = map(float, parts)
                    boxes.append((int(cls), cx, cy, w, h))
        return boxes

    def save_yolo_labels(self, boxes, label_path):
        """Save label YOLO format"""
        with open(label_path, 'w') as f:
            for cls, cx, cy, w, h in boxes:
                f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    def apply_flip_horizontal(self, image, boxes):
        """Flip horizontal"""
        new_image = cv2.flip(image, 1)
        new_boxes = []
        for cls, cx, cy, w, h in boxes:
            new_boxes.append((cls, 1.0 - cx, cy, w, h))
        return new_image, new_boxes

    def apply_brightness(self, image, boxes, factor):
        """Ubah brightness"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float32)
        hsv[:, :, 2] = hsv[:, :, 2] * factor
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        hsv = hsv.astype(np.uint8)
        new_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return new_image, boxes

    def apply_contrast(self, image, boxes, alpha):
        """Ubah contrast"""
        new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        return new_image, boxes

    def apply_rotation_90(self, image, boxes):
        """Rotate 90 derajat clockwise"""
        h, w = image.shape[:2]
        new_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        new_h, new_w = new_image.shape[:2]
        new_boxes = []
        for cls, cx, cy, bw, bh in boxes:
            # Transform koordinat
            new_cx = cy
            new_cy = 1.0 - cx
            new_bw = bh * (new_w / w)
            new_bh = bw * (new_h / h)
            new_boxes.append((cls, new_cx, new_cy, new_bw, new_bh))
        return new_image, new_boxes

    def apply_rotation_180(self, image, boxes):
        """Rotate 180 derajat"""
        new_image = cv2.rotate(image, cv2.ROTATE_180)
        new_boxes = []
        for cls, cx, cy, w, h in boxes:
            new_boxes.append((cls, 1.0 - cx, 1.0 - cy, w, h))
        return new_image, new_boxes

    def apply_rotation_270(self, image, boxes):
        """Rotate 270 derajat clockwise"""
        h, w = image.shape[:2]
        new_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        new_h, new_w = new_image.shape[:2]
        new_boxes = []
        for cls, cx, cy, bw, bh in boxes:
            new_cx = 1.0 - cy
            new_cy = cx
            new_bw = bh * (new_w / w)
            new_bh = bw * (new_h / h)
            new_boxes.append((cls, new_cx, new_cy, new_bw, new_bh))
        return new_image, new_boxes

    def apply_blur(self, image, boxes, kernel_size=5):
        """Tambah blur gaussian"""
        new_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return new_image, boxes

    def apply_noise(self, image, boxes, sigma=10):
        """Tambah noise"""
        noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
        new_image = image.astype(np.float32) + noise
        new_image = np.clip(new_image, 0, 255).astype(np.uint8)
        return new_image, boxes

    def apply_saturation(self, image, boxes, factor):
        """Ubah saturation"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * factor
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        hsv = hsv.astype(np.uint8)
        new_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return new_image, boxes

    def generate_variations(self, image, boxes):
        """Generate semua variasi dari satu gambar"""
        variations = []
        aug_funcs = [
            ("flip_h", self.apply_flip_horizontal),
            ("bright_1.3", lambda img, b: self.apply_brightness(img, b, 1.3)),
            ("bright_0.7", lambda img, b: self.apply_brightness(img, b, 0.7)),
            ("contrast_1.3", lambda img, b: self.apply_contrast(img, b, 1.3)),
            ("contrast_0.7", lambda img, b: self.apply_contrast(img, b, 0.7)),
            ("rot90", self.apply_rotation_90),
            ("rot180", self.apply_rotation_180),
            ("rot270", self.apply_rotation_270),
            ("blur", self.apply_blur),
            ("noise", lambda img, b: self.apply_noise(img, b, 15)),
            ("sat_1.3", lambda img, b: self.apply_saturation(img, b, 1.3)),
            ("sat_0.7", lambda img, b: self.apply_saturation(img, b, 0.7)),
        ]

        # Ambil 8 variasi acak (atau semua jika kurang dari 8)
        selected = random.sample(aug_funcs, min(self.num_variations, len(aug_funcs)))

        for name, func in selected:
            try:
                new_img, new_boxes = func(image.copy(), boxes)
                variations.append((name, new_img, new_boxes))
            except Exception as e:
                print(f"  Warning: Gagal augmentasi {name}: {e}")

        return variations

    def process_all(self):
        """Process semua gambar di direktori"""
        # Ambil semua gambar
        images = sorted([
            f for f in self.image_input_dir.iterdir()
            if f.suffix.lower() in self.image_exts
        ])

        total_original = len(images)
        total_augmented = 0

        print(f"\n📁 Image Input: {self.image_input_dir}")
        print(f"📁 Label Input: {self.label_input_dir}")
        print(f"📁 Image Output: {self.image_output_dir}")
        print(f"📁 Label Output: {self.label_output_dir}")
        print(f"🖼️  Gambar asli: {total_original}")
        print(f"🔄 Variasi per gambar: {self.num_variations}")
        print(f"📊 Estimasi total: {total_original * self.num_variations} augmented")
        print("\n" + "="*50)

        for img_path in images:
            # Load gambar
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"❌ Gagal load: {img_path.name}")
                continue

            # Load labels dari direktori label yang benar
            label_path = self.label_input_dir / f"{img_path.stem}.txt"
            boxes = self.load_yolo_labels(label_path)

            print(f"\n🖼️  {img_path.name} ({len(boxes)} boxes)")

            # Generate variasi
            variations = self.generate_variations(image, boxes)

            # Simpan original juga
            base_name = img_path.stem
            ext = img_path.suffix

            # Simpan original
            cv2.imwrite(str(self.image_output_dir / f"{base_name}_orig{ext}"), image)
            self.save_yolo_labels(boxes, self.label_output_dir / f"{base_name}_orig.txt")

            # Simpan variasi
            for i, (aug_name, aug_img, aug_boxes) in enumerate(variations):
                new_name = f"{base_name}_{aug_name}{ext}"
                new_label = f"{base_name}_{aug_name}.txt"

                success = cv2.imwrite(str(self.image_output_dir / new_name), aug_img)
                if success:
                    self.save_yolo_labels(aug_boxes, self.label_output_dir / new_label)
                    print(f"  ✅ {new_name}")
                    total_augmented += 1
                else:
                    print(f"  ❌ Gagal simpan: {new_name}")

            print(f"  → Total: 1 original + {len(variations)} variasi")

        print("\n" + "="*50)
        print(f"✅ SELESAI!")
        print(f"   Original: {total_original}")
        print(f"   Augmented: {total_augmented}")
        print(f"   Total file: {total_original + total_augmented}")
        print(f"   Lokasi Gambar: {self.image_output_dir}")
        print(f"   Lokasi Label: {self.label_output_dir}")


def main():
    parser = argparse.ArgumentParser(description='YOLO Auto-Augmentation')
    parser.add_argument('--image-input', '-i', type=str,
                        default='data/dataset/images/train',
                        help='Direktori gambar input')
    parser.add_argument('--label-input', '-li', type=str,
                        default='data/dataset/labels/train',
                        help='Direktori label input')
    parser.add_argument('--output', '-o', type=str,
                        default='data/dataset/images/augmented',
                        help='Direktori output gambar')
    parser.add_argument('--label-output', '-lo', type=str,
                        default='data/dataset/labels/augmented',
                        help='Direktori output label')
    parser.add_argument('--variations', '-v', type=int,
                        default=8,
                        help='Jumlah variasi per gambar (default: 8)')

    args = parser.parse_args()

    print("🚀 YOLO Auto-Augmentation Script")
    print("="*50)

    image_input_dir = Path(args.image_input)
    label_input_dir = Path(args.label_input)
    output_dir = Path(args.output)

    if not image_input_dir.exists():
        print(f"❌ Direktori gambar tidak ditemukan: {image_input_dir}")
        return

    if not label_input_dir.exists():
        print(f"❌ Direktori label tidak ditemukan: {label_input_dir}")
        return

    # Pastikan direktori output ada
    output_dir = Path(args.output)
    label_output_dir = Path(args.label_output)
    output_dir.mkdir(parents=True, exist_ok=True)
    label_output_dir.mkdir(parents=True, exist_ok=True)

    augmentor = YOLOAugmentor(
        image_input_dir=image_input_dir,
        label_input_dir=label_input_dir,
        image_output_dir=output_dir,
        label_output_dir=label_output_dir,
        num_variations=args.variations
    )
    augmentor.process_all()

    print("\n✅ Semua augmentasi selesai!")


if __name__ == "__main__":
    main()
