import os
import sys
import zipfile
import cv2
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE_PATH = os.path.join(BASE_DIR, "archive.zip")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "real_plates_extracted")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_and_crop_plates(max_count: int = None):
    print(f"📦 جاري فتح الأرشيف: {ARCHIVE_PATH}...")
    with zipfile.ZipFile(ARCHIVE_PATH, 'r') as z:
        all_names = z.namelist()
        
        # ربط كل صورة بملف الـ Label بتاعها
        label_files = {
            os.path.splitext(os.path.basename(n))[0]: n 
            for n in all_names if n.endswith('.txt') and 'Vehicles Labeling' in n
        }
        
        image_files = [
            n for n in all_names 
            if n.lower().endswith(('.jpg', '.jpeg', '.png')) and 'Vehicles/' in n
        ]

        print(f"🔍 وجدنا {len(image_files)} صورة سيارة و {len(label_files)} ملف إحداثيات لوحة.")

        cropped_count = 0
        skipped_count = 0

        for img_path in image_files:
            base_id = os.path.splitext(os.path.basename(img_path))[0]
            label_path = label_files.get(base_id)
            if not label_path:
                skipped_count += 1
                continue

            try:
                # قراءة ملف الإحداثيات
                txt_content = z.read(label_path).decode('utf-8').strip()
                if not txt_content:
                    skipped_count += 1
                    continue

                lines = txt_content.split('\n')
                # قراءة الصورة من الـ zip في الذاكرة مباشرة
                img_data = z.read(img_path)
                np_arr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if img is None:
                    skipped_count += 1
                    continue

                img_h, img_w = img.shape[:2]

                for p_idx, line in enumerate(lines):
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    
                    # إحداثيات YOLO المقاسة (Normalized)
                    cx = float(parts[1]) * img_w
                    cy = float(parts[2]) * img_h
                    bw = float(parts[3]) * img_w
                    bh = float(parts[4]) * img_h

                    # إضافة هامش أمان بنسبة 4% لضمان عدم قطع حواف اللوحة
                    margin_x = bw * 0.04
                    margin_y = bh * 0.04

                    x1 = max(0, int(cx - bw / 2 - margin_x))
                    y1 = max(0, int(cy - bh / 2 - margin_y))
                    x2 = min(img_w, int(cx + bw / 2 + margin_x))
                    y2 = min(img_h, int(cy + bh / 2 + margin_y))

                    crop = img[y1:y2, x1:x2]
                    # التحقق من أن حجم اللوحة كافي للمعالجة (تجاهل اللوحات التالفة أو متناهية الصغر)
                    if crop.shape[0] < 20 or crop.shape[1] < 45:
                        continue

                    out_name = f"plate_{base_id}_{p_idx}.jpg"
                    out_path = os.path.join(OUTPUT_DIR, out_name)
                    cv2.imwrite(out_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    cropped_count += 1

                if max_count and cropped_count >= max_count:
                    break

            except Exception as e:
                skipped_count += 1
                continue

        print("=" * 60)
        print(f"🎉 تم استخراج وقص {cropped_count} لوحة سيارة حقيقية بنجاح!")
        print(f"📁 تم الحفظ في: {OUTPUT_DIR}")
        print("=" * 60)

if __name__ == "__main__":
    extract_and_crop_plates()
