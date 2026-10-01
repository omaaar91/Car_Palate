import os
import sys
import json
import shutil
import cv2
import numpy as np

# إعداد الترميز للغة العربية
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from core.pipeline import EgyptianALPR

VEHICLES_DIR = os.path.join(BASE_DIR, "archive", "EALPR Vechicles dataset", "Vehicles")
CROPS_DIR = os.path.join(BASE_DIR, "data", "real_plates_extracted")
AUTO_LABELED_DIR = os.path.join(BASE_DIR, "data", "auto_labeled")
CANDIDATES_DIR = os.path.join(AUTO_LABELED_DIR, "candidates")
VERIFIED_DIR = os.path.join(AUTO_LABELED_DIR, "verified")
REVIEW_META_PATH = os.path.join(AUTO_LABELED_DIR, "review_data.json")

os.makedirs(CANDIDATES_DIR, exist_ok=True)
os.makedirs(VERIFIED_DIR, exist_ok=True)

def prepare_500_vehicles(start_num: int = 561, target_count: int = 500):
    print("=" * 70)
    print(f"🚀 بدء تجهيز {target_count} صورة سيارة جديدة كلياً للمراجعة وحساب الدقة...")
    print("=" * 70)

    # 1. النسخ الاحتياطي لملف المراجعة السابق لحفظ أي عمل قديم
    if os.path.exists(REVIEW_META_PATH):
        backup_path = os.path.join(AUTO_LABELED_DIR, "review_data_backup_prev.json")
        shutil.copy2(REVIEW_META_PATH, backup_path)
        print(f"💾 تم حفظ نسخة احتياطية من المراجعات السابقة في: {os.path.basename(backup_path)}")

    # 2. تحميل المحرك الذكي المطور بجميع موديلاته الثلاثة
    engine = EgyptianALPR()
    class_map = {v: k for k, v in engine.recog_model.names.items()}

    # تنظيف مجلد المرشحين الجديد (candidates) لبدء دفعة الـ 500 نظيفة
    for f in os.listdir(CANDIDATES_DIR):
        p = os.path.join(CANDIDATES_DIR, f)
        if os.path.isfile(p):
            os.remove(p)

    review_items = []
    processed_count = 0
    current_id = start_num

    syntax_valid_count = 0
    gov_stats = {}

    while processed_count < target_count and current_id <= 2087:
        str_num = f"{current_id:04d}"
        v_filename = f"{str_num}.jpg"
        v_path = os.path.join(VEHICLES_DIR, v_filename)
        current_id += 1

        if not os.path.exists(v_path):
            continue

        vehicle_img = cv2.imread(v_path)
        if vehicle_img is None:
            continue

        img_h, img_w = vehicle_img.shape[:2]

        # -------------------------------------------------------------
        # الموديل 1: كشف موضع اللوحة مع هامش تكيفي (Adaptive Padding)
        # -------------------------------------------------------------
        p_res = engine.plate_model(vehicle_img, conf=0.25, verbose=False)[0]
        crop = None

        if len(p_res.boxes) > 0:
            best_box = max(p_res.boxes, key=lambda b: float(b.conf[0]))
            bx1, by1, bx2, by2 = map(int, best_box.xyxy[0])
            bw, bh = bx2 - bx1, by2 - by1
            pad_x = int(bw * 0.04)
            pad_y = int(bh * 0.05)
            crop = vehicle_img[max(0, by1 - pad_y):min(img_h, by2 + pad_y), 
                               max(0, bx1 - pad_x):min(img_w, bx2 + pad_x)]
        else:
            # بديل احتياطي: استخدام القص المرجعي من الأرشيف إن وُجد
            crop_cand = os.path.join(CROPS_DIR, f"plate_{str_num}_0.jpg")
            if os.path.exists(crop_cand):
                crop = cv2.imread(crop_cand)

        if crop is None or crop.shape[0] < 14 or crop.shape[1] < 30:
            continue

        # -------------------------------------------------------------
        # الموديل 2 و 3: تحديد الخانات وقراءة الرموز عبر خط المعالجة المطور
        # -------------------------------------------------------------
        rec = engine.recognize_plate_crop(crop)
        text = rec.get("text", "")
        syntax_valid = rec.get("syntax_valid", False)
        char_details = rec.get("char_details", [])
        annotated = rec.get("annotated")
        governorate = rec.get("governorate", "غير محدد")
        badge = rec.get("badge", "")

        if annotated is None or len(char_details) == 0:
            continue

        # بناء أسطر YOLO للتسمية والتدريب اللاحق
        h_crop, w_crop = annotated.shape[:2]
        yolo_lines = []
        for c in char_details:
            c_name = c["class_name"]
            c_id = class_map.get(c_name)
            if c_id is None:
                continue

            lx1, ly1, lx2, ly2 = c["box"]
            cx = ((lx1 + lx2) / 2.0) / w_crop
            cy = ((ly1 + ly2) / 2.0) / h_crop
            bw = (lx2 - lx1) / w_crop
            bh = (ly2 - ly1) / h_crop

            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            bw = min(max(bw, 0.01), 1.0)
            bh = min(max(bh, 0.01), 1.0)
            yolo_lines.append(f"{c_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        # حفظ الصورة المعالجة وملف الـ txt داخل candidates
        base_name = f"vehicle_{str_num}"
        img_out = os.path.join(CANDIDATES_DIR, f"{base_name}.jpg")
        txt_out = os.path.join(CANDIDATES_DIR, f"{base_name}.txt")

        cv2.imwrite(img_out, annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(txt_out, "w", encoding="utf-8") as f_txt:
            f_txt.write("\n".join(yolo_lines) + "\n")

        if syntax_valid:
            syntax_valid_count += 1
        gov_stats[governorate] = gov_stats.get(governorate, 0) + 1

        review_items.append({
            "id": processed_count,
            "filename": f"{base_name}.jpg",
            "vehicle_id": str_num,
            "folder": "candidates",
            "plate_text": text,
            "syntax_valid": syntax_valid,
            "governorate": governorate,
            "badge": badge,
            "chars": [
                {
                    "name": c["class_name"],
                    "arabic": c["arabic"],
                    "is_digit": c["is_digit"],
                    "box": c["box"]
                }
                for c in char_details
            ],
            "img_url": f"/auto_labeled/candidates/{base_name}.jpg",
            # بيانات تقييم الدقة الفعلية
            "initial_plate_text": text,
            "initial_chars": [c["arabic"] for c in char_details],
            "reviewed": False,
            "is_modified": False,
            "accuracy_status": "pending"
        })

        processed_count += 1
        if processed_count % 50 == 0 or processed_count == target_count:
            pct_valid = (syntax_valid_count / processed_count) * 100
            print(f" ⏳ تم تجهيز: {processed_count:3d} / {target_count} | نسبة التطابق القياسي الأولي: {pct_valid:.1f}%")

    # حفظ ملف الـ metadata الكامل للمراجعة
    with open(REVIEW_META_PATH, "w", encoding="utf-8") as f_meta:
        json.dump(review_items, f_meta, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(f"🎉 تم تجهيز {len(review_items)} صورة لوحة جديدة بالكامل في صفحة المراجعة!")
    print(f"📊 النتائج الأولية للـ 3 موديلات معاً قبل أي تدخل بشري:")
    print(f"   - لوحات قياسية مطابقة 100% للمرور المصري: {syntax_valid_count} ({syntax_valid_count / len(review_items) * 100:.1f}%)")
    print(f"   - توزيع المحافظات: {gov_stats}")
    print(f"🔗 افتح صفحة المراجعة الآن: https://localhost:8443/review")
    print("=" * 70)

if __name__ == "__main__":
    prepare_500_vehicles(start_num=561, target_count=500)
