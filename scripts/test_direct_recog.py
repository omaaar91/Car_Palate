import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

p_model = YOLO(os.path.join(BASE_DIR, "best.pt"))
recog_model = YOLO(os.path.join(BASE_DIR, "best_char_afterTuning.pt"))

from core.deskew import deskew_adaptive
from core.config import CHAR_MAP, DIGITS_SET, CLASS_WEIGHTS
from core.verification import validate_egyptian_syntax

def nms_chars(detections, iou_thresh=0.30):
    if not detections:
        return []
    # فرز تنازلي حسب درجة الثقة
    dets = sorted(detections, key=lambda x: x['conf'], reverse=True)
    keep = []
    for d in dets:
        b = d['box']
        bcx = (b[0] + b[2]) / 2.0
        bw = b[2] - b[0]
        overlap = False
        for k in keep:
            kb = k['box']
            inter = max(0, min(b[2], kb[2]) - max(b[0], kb[0]))
            union = max(b[2], kb[2]) - min(b[0], kb[0])
            iou = inter / union if union > 0 else 0
            kcx = (kb[0] + kb[2]) / 2.0
            kw = kb[2] - kb[0]
            if iou > iou_thresh or abs(bcx - kcx) < min(bw, kw) * 0.52:
                overlap = True
                break
        if not overlap:
            keep.append(d)
    return sorted(keep, key=lambda x: x['box'][0])

def test_vehicles_direct(start_idx=541, count=20):
    print("=" * 80)
    print(f"🚀 اختبار النظام بدون الموديل الثاني (End-to-End Direct Detection) على {count} سيارة:")
    print("=" * 80)
    
    total_standard = 0
    vehicles_dir = os.path.join(BASE_DIR, "archive", "EALPR Vechicles dataset", "Vehicles")

    for num in range(start_idx, start_idx + count):
        fname = f"{num:04d}.jpg"
        img_path = os.path.join(vehicles_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue

        p_res = p_model(img, conf=0.25, verbose=False)[0]
        if len(p_res.boxes) == 0:
            print(f"{fname} | لم يتم كشف اللوحة")
            continue

        box = max(p_res.boxes, key=lambda b: float(b.conf[0]))
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bw, bh = x2 - x1, y2 - y1
        pad_x, pad_y = int(bw * 0.04), int(bh * 0.05)
        img_h, img_w = img.shape[:2]
        crop = img[max(0, y1 - pad_y):min(img_h, y2 + pad_y), 
                   max(0, x1 - pad_x):min(img_w, x2 + pad_x)]
        
        crop_rot, _ = deskew_adaptive(crop)
        h, w = crop_rot.shape[:2]
        pass1 = cv2.resize(crop_rot, (w * 4, h * 4), interpolation=cv2.INTER_LANCZOS4)
        w_tot = pass1.shape[1]

        # تشغيل الموديل 3 مباشرة مع عتبة ثقة ذكية
        c_res = recog_model(pass1, conf=0.22, verbose=False)[0]
        raw_dets = []
        for cb in c_res.boxes:
            cname = recog_model.names[int(cb.cls[0])]
            conf = float(cb.conf[0])
            box_coords = [int(v) for v in cb.xyxy[0]]
            
            # استبعاد الضوضاء الملتصقة جداً بالحافة الخارجية للإطار
            if box_coords[0] < w_tot * 0.03 or box_coords[2] > w_tot * 0.985:
                continue

            raw_dets.append({
                'name': cname,
                'conf': conf,
                'box': box_coords,
                'is_digit': cname in DIGITS_SET
            })

        filtered = nms_chars(raw_dets, iou_thresh=0.30)

        nums = [CHAR_MAP.get(d['name'], d['name']) for d in filtered if d['is_digit']]
        lets = [CHAR_MAP.get(d['name'], d['name']) for d in filtered if not d['is_digit']]
        rev_lets = list(reversed(lets))

        syntax = validate_egyptian_syntax(nums, rev_lets)
        is_std = syntax['is_valid']
        if is_std:
            total_standard += 1

        n_str = ' '.join(nums) if nums else '-'
        l_str = ' '.join(rev_lets) if rev_lets else '-'
        txt = f"[{n_str}] | [{l_str}]"
        status = "✅ قياسية" if is_std else "⚠️ غير قياسية"
        gov = syntax['governorate']
        print(f"{fname} | {txt:<24} | {gov:<12} | {status}")

    print("=" * 80)
    print(f"🎯 النتيجة الإجمالية: {total_standard} من {count} لوحة قياسية ({total_standard / count * 100:.1f}%)")
    print("=" * 80)

if __name__ == "__main__":
    test_vehicles_direct(541, 20)
