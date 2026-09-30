import os
import sys
import json
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from core.pipeline import EgyptianALPR

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CROPS_DIR = os.path.join(BASE_DIR, "data", "real_plates_extracted")
OUT_DIR = os.path.join(BASE_DIR, "data", "auto_labeled")
VERIFIED_DIR = os.path.join(OUT_DIR, "verified")
CANDIDATES_DIR = os.path.join(OUT_DIR, "candidates")
META_PATH = os.path.join(OUT_DIR, "review_data.json")

os.makedirs(VERIFIED_DIR, exist_ok=True)
os.makedirs(CANDIDATES_DIR, exist_ok=True)

def run_auto_labeling(new_samples_count: int = 250):
    print("🧠 جاري تهيئة محرك التعرف الذكي متعدد المراحل...")
    engine = EgyptianALPR()
    class_map = {v: k for k, v in engine.recog_model.names.items()}

    # تحميل البيانات السابقة لمنع التكرار والحفاظ على التعديلات السابقة
    existing_items = []
    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, "r", encoding="utf-8") as f:
                existing_items = json.load(f)
        except Exception:
            existing_items = []

    existing_filenames = set(item["filename"] for item in existing_items)
    
    # فحص الملفات الموجودة في المجلدات فعلياً
    on_disk_done = set(
        os.path.splitext(f)[0] 
        for f in os.listdir(VERIFIED_DIR) + os.listdir(CANDIDATES_DIR) 
        if f.endswith('.jpg')
    )

    all_crop_files = sorted([f for f in os.listdir(CROPS_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    # اختيار الملفات الجديدة فقط التي لم تُعالج بعد
    unprocessed_files = [f for f in all_crop_files if os.path.splitext(f)[0] not in on_disk_done]
    print(f"📂 إجمالي اللوحات المقصوصة: {len(all_crop_files)} لوحة.")
    print(f"✅ تم معالجتها سابقاً: {len(on_disk_done)} لوحة.")
    print(f"⏳ المتبقي المتاح للمعالجة: {len(unprocessed_files)} لوحة.")

    batch_to_process = unprocessed_files[:new_samples_count]
    print(f"🚀 جاري معالجة الدفعة الجديدة: {len(batch_to_process)} لوحة إضافية...")

    newly_verified = 0
    newly_candidate = 0
    start_id = len(existing_items)

    for idx, fname in enumerate(batch_to_process):
        img_path = os.path.join(CROPS_DIR, fname)
        crop = cv2.imread(img_path)
        if crop is None:
            continue

        base_name = os.path.splitext(fname)[0]
        rec = engine.recognize_plate_crop(crop)
        
        text = rec.get("text", "")
        syntax_valid = rec.get("syntax_valid", False)
        char_details = rec.get("char_details", [])
        annotated = rec.get("annotated")

        # تجاهل الصور التي لم يُكتشف بها عدد كافي من المحارف
        if not char_details or "?" in text or len(char_details) < 4:
            continue

        h, w = annotated.shape[:2]
        yolo_lines = []

        for c in char_details:
            c_name = c["class_name"]
            c_id = class_map.get(c_name)
            if c_id is None:
                continue

            lx1, ly1, lx2, ly2 = c["box"]
            cx = ((lx1 + lx2) / 2.0) / w
            cy = ((ly1 + ly2) / 2.0) / h
            bw = (lx2 - lx1) / w
            bh = (ly2 - ly1) / h

            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            bw = min(max(bw, 0.01), 1.0)
            bh = min(max(bh, 0.01), 1.0)

            yolo_lines.append(f"{c_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if not yolo_lines:
            continue

        target_sub = "verified" if syntax_valid else "candidates"
        target_dir = os.path.join(OUT_DIR, target_sub)

        img_out_path = os.path.join(target_dir, f"{base_name}.jpg")
        txt_out_path = os.path.join(target_dir, f"{base_name}.txt")

        cv2.imwrite(img_out_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(txt_out_path, "w", encoding="utf-8") as f_txt:
            f_txt.write("\n".join(yolo_lines) + "\n")

        if syntax_valid:
            newly_verified += 1
        else:
            newly_candidate += 1

        existing_items.append({
            "id": start_id + idx,
            "filename": f"{base_name}.jpg",
            "folder": target_sub,
            "plate_text": text,
            "syntax_valid": syntax_valid,
            "governorate": rec.get("governorate", "غير محدد"),
            "badge": rec.get("badge", ""),
            "chars": [
                {
                    "name": c["class_name"],
                    "arabic": c["arabic"],
                    "is_digit": c["is_digit"],
                    "box": c["box"]
                }
                for c in char_details
            ],
            "img_url": f"/auto_labeled/{target_sub}/{base_name}.jpg"
        })

        if (idx + 1) % 50 == 0:
            print(f" ⏳ معالجة الدفعة: {idx + 1} / {len(batch_to_process)}... (مؤكدة جديدة: {newly_verified})")

    # حفظ قائمة المراجعة المحدثة
    with open(META_PATH, "w", encoding="utf-8") as f_meta:
        json.dump(existing_items, f_meta, ensure_ascii=False, indent=2)

    total_verified_now = len([f for f in os.listdir(VERIFIED_DIR) if f.endswith('.jpg')])

    print("=" * 65)
    print(f"🎉 تم الانتهاء من معالجة الدفعة الإضافية بنجاح!")
    print(f"✨ لوحات مؤكدة أضيفت الآن: {newly_verified}")
    print(f"⚠️ لوحات بحاجة لمراجعة أضيفت: {newly_candidate}")
    print(f"🏆 إجمالي اللوحات المعتمدة الجاهزة للتدريب الآن: {total_verified_now} لوحة!")
    print(f"📋 إجمالي اللوحات في أداة المراجعة: {len(existing_items)} لوحة")
    print("=" * 65)

if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    run_auto_labeling(new_samples_count=count)
