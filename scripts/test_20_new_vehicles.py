import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from ultralytics import YOLO
from core.deskew import deskew_adaptive
from core.config import CHAR_MAP, DIGITS_SET, CLASS_WEIGHTS, OFFICIAL_ALLOWED_LETTERS
from core.verification import (
    validate_egyptian_syntax,
    suppress_duplicate_slots,
    prune_spurious_boxes,
    verify_digit_geometry,
    verify_letter_geometry
)

# 1. تحميل النماذج الثلاثة مع النموذج الجديد المطور
print("🧠 جاري تحميل النماذج الذكية الثلاثة معاً...")
plate_model = YOLO(os.path.join(BASE_DIR, "best.pt"))
loc_model   = YOLO(os.path.join(BASE_DIR, "best_char.pt"))
recog_model = YOLO(os.path.join(BASE_DIR, "best_char_afterTuning.pt"))
print("✅ تم تحميل النماذج الثلاثة بنجاح!")

font_path = "C:/Windows/Fonts/tahoma.ttf"
if not os.path.exists(font_path):
    font_path = "C:/Windows/Fonts/arial.ttf"

def render_arabic_text(text: str) -> str:
    try:
        return get_display(arabic_reshaper.reshape(text))
    except Exception:
        return text

def process_single_vehicle(img_path):
    """تشغيل النماذج الثلاثة بتناغم كامل على صورة سيارة حقيقية كاملة"""
    orig_img = cv2.imread(img_path)
    if orig_img is None:
        return None, "فشل تحميل الصورة", False, "", 0.0

    # -------------------------------------------------------------
    # الموديل 1: كشف موضع لوحة السيارة على جسم السيارة الكامل (Plate Detection)
    # -------------------------------------------------------------
    p_res = plate_model(orig_img, conf=0.25, verbose=False)[0]
    if len(p_res.boxes) == 0:
        return None, "لم يتم العثور على لوحة", False, "غير محدد", 0.0

    best_pbox = max(p_res.boxes, key=lambda b: float(b.conf[0]))
    bx1, by1, bx2, by2 = map(int, best_pbox.xyxy[0])
    p_conf = float(best_pbox.conf[0])
    
    bw_plate, bh_plate = bx2 - bx1, by2 - by1
    pad_x = int(bw_plate * 0.04)
    pad_y = int(bh_plate * 0.05)
    img_h, img_w = orig_img.shape[:2]
    crop = orig_img[max(0, by1 - pad_y):min(img_h, by2 + pad_y), max(0, bx1 - pad_x):min(img_w, bx2 + pad_x)]
    if crop.shape[0] < 12 or crop.shape[1] < 25:
        return None, "لوحة صغيرة جداً", False, "غير محدد", 0.0

    # محاذاة وتصحيح الميلان
    crop_rot, angle = deskew_adaptive(crop)
    h, w = crop_rot.shape[:2]

    # تحسين الجودة والدقة الفائقة بهندسة Lanczos4
    pass1 = cv2.resize(crop_rot, (w * 4, h * 4), interpolation=cv2.INTER_LANCZOS4)
    w_tot = pass1.shape[1]
    pass2 = cv2.resize(crop_rot, (int(w * 4.8), int(h * 4.8)), interpolation=cv2.INTER_LANCZOS4)
    scale2 = pass2.shape[1] / w_tot

    # -------------------------------------------------------------
    # الموديل 2: كشف مواضع الخانات والمحارف المادية مع إلغاء التكرار
    # -------------------------------------------------------------
    l_res = loc_model(pass1, conf=0.14, verbose=False)[0]
    raw_b = []
    for bx in l_res.boxes:
        coords = list(map(int, bx.xyxy[0][:4]))
        coords.append(float(bx.conf[0]))
        raw_b.append(coords)

    # إلغاء الصناديق المتداخلة لنفس الخانة (Duplicate Slot Suppression)
    suppressed_b = suppress_duplicate_slots(raw_b, iou_thresh=0.35)

    clean_b = []
    for bx in suppressed_b:
        bw = bx[2] - bx[0]
        bh = bx[3] - bx[1]
        if bx[0] < (w_tot * 0.04) and bx[0] <= 8: continue
        if bx[2] >= (w_tot * 0.985) and (bh / max(1, bw) > 2.5 or bw < w_tot * 0.04): continue
        if bw < (w_tot * 0.03) or (bh / max(1, bw)) > 3.5: continue
        clean_b.append(bx)

    if len(clean_b) < 2:
        return crop_rot, "عدد الخانات غير كافٍ", False, "غير محدد", p_conf

    # -------------------------------------------------------------
    # الموديل 3: مصنف الحروف والأرقام الجديد بعد الضبط الدقيق (Character Recognition)
    # -------------------------------------------------------------
    c1 = recog_model(pass1, conf=0.003, verbose=False)[0]
    c2 = recog_model(pass2, conf=0.003, verbose=False)[0]

    candidates = []
    for cb in c1.boxes:
        candidates.append((list(map(float, cb.xyxy[0])), recog_model.names[int(cb.cls[0])], float(cb.conf[0]) * 1.0))
    for cb in c2.boxes:
        coords = [float(cb.xyxy[0][0])/scale2, float(cb.xyxy[0][1])/scale2, float(cb.xyxy[0][2])/scale2, float(cb.xyxy[0][3])/scale2]
        candidates.append((coords, recog_model.names[int(cb.cls[0])], float(cb.conf[0]) * 0.95))

    # استرجاع أي خانة واضحة ومؤكدة سقطت من الموديل 2 ورصدها الموديل 3 بثقة عالية
    for (cx1, cy1, cx2, cy2), cname, raw_c in candidates:
        if raw_c > 0.40 and len(clean_b) < 7:
            cw = cx2 - cx1
            ccx = (cx1 + cx2) / 2.0
            has_overlap = False
            for bx in clean_b:
                inter = max(0, min(bx[2], cx2) - max(bx[0], cx1))
                union = max(bx[2], cx2) - min(bx[0], cx1)
                iou = inter / union if union > 0 else 0
                bcx = (bx[0] + bx[2]) / 2.0
                bw = bx[2] - bx[0]
                if iou > 0.15 or abs(bcx - ccx) < min(bw, cw) * 0.55:
                    has_overlap = True
                    break
            if not has_overlap and cx1 > (w_tot * 0.03) and cx2 < (w_tot * 0.98):
                clean_b.append([int(cx1), int(cy1), int(cx2), int(cy2)])
    clean_b.sort(key=lambda b: b[0])

    # تنقية الصناديق الزائفة
    clean_b = prune_spurious_boxes(clean_b, candidates, pass1.shape[1], pass1.shape[0])

    # بدون أي شرط مسبق على عدد الحروف أو الأرقام في كل جهة:
    # كل خانة تختار تصنيفها الطبيعي الحر وفق أعلى تطابق وثقة
    stage1_cands = []
    stage1_scores = []
    box_cands_map = []
    for i, (lx1, ly1, lx2, ly2) in enumerate(clean_b):
        best_cand, best_score = None, -1.0
        cands_for_box = []
        for (cx1, cy1, cx2, cy2), cname, raw_c in candidates:
            inter = max(0, min(lx2, cx2) - max(lx1, cx1))
            union = max(lx2, cx2) - min(lx1, cx1)
            iou = inter / union if union > 0 else 0
            if iou > 0.22:
                w_c = raw_c * CLASS_WEIGHTS.get(cname, 1.0)
                cands_for_box.append((w_c, cname))
                if w_c > best_score:
                    best_score = w_c
                    best_cand = cname
        stage1_cands.append(best_cand)
        stage1_scores.append(best_score)
        box_cands_map.append(cands_for_box)

    detected_letters = [c for c in stage1_cands if c not in DIGITS_SET and c is not None]
    all_letter_patches = [(stage1_cands[i], pass1[ly1:ly2, lx1:lx2]) for i, (lx1, ly1, lx2, ly2) in enumerate(clean_b) if stage1_cands[i] not in DIGITS_SET]

    annotated = pass1.copy()
    nums, lets = [], []
    box_labels = []

    for i, (lx1, ly1, lx2, ly2) in enumerate(clean_b):
        cand = stage1_cands[i]
        score = stage1_scores[i]
        patch = pass1[ly1:ly2, lx1:lx2]
        box_cands = box_cands_map[i]
        
        # تجاهل الصناديق الزائفة كالمسامير وحواف اللوحة التي ليس لها محرف مطابق
        if cand is None or score < 0.08:
            continue

        is_digit = (cand in DIGITS_SET)
        if is_digit:
            verified = verify_digit_geometry(patch, cand, score, box_cands)
            ar_char = CHAR_MAP.get(verified, '?')
            nums.append(ar_char)
            box_labels.append(((lx1, ly1, lx2, ly2), ar_char, True))
        else:
            verified = verify_letter_geometry(patch, cand, detected_letters, box_cands, all_letter_patches)
            ar_char = CHAR_MAP.get(verified, '?')
            lets.append(ar_char)
            box_labels.append(((lx1, ly1, lx2, ly2), ar_char, False))

        color = (0, 220, 0) if is_digit else (255, 120, 0)
        cv2.rectangle(annotated, (lx1, ly1), (lx2, ly2), color, 3)

    # رسم الحروف العربية فوق البوكسات
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(annotated_rgb)
    draw = ImageDraw.Draw(pil_img)
    font_char = ImageFont.truetype(font_path, 22)

    for (lx1, ly1, lx2, ly2), ar_char, is_digit in box_labels:
        color = (0, 255, 100) if is_digit else (255, 160, 0)
        reshaped = render_arabic_text(ar_char)
        draw.rectangle([lx1, max(0, ly1 - 28), lx1 + 28, ly1], fill=(0, 0, 0, 220))
        draw.text((lx1 + 4, max(0, ly1 - 28)), reshaped, font=font_char, fill=color)

    annotated = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # ترتيب الحروف للقراءة العربية (من اليمين لليسار)
    arabic_lets = list(reversed(lets))
    syntax_info = validate_egyptian_syntax(nums, arabic_lets)

    num_txt = ' '.join(nums) if nums else "-"
    let_txt = ' '.join(arabic_lets) if arabic_lets else "-"
    full_plate_str = f"[{num_txt}] | [{let_txt}]"

    return annotated, full_plate_str, syntax_info["is_valid"], syntax_info["governorate"], p_conf

def run_test_20(start_num: int = 521, count: int = 20):
    vehicles_dir = os.path.join(BASE_DIR, "archive", "EALPR Vechicles dataset", "Vehicles")
    all_files = sorted([f for f in os.listdir(vehicles_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    test_files = [f for f in all_files if int(os.path.splitext(f)[0]) >= start_num][:count]
    end_num = int(os.path.splitext(test_files[-1])[0]) if test_files else start_num + count - 1

    print(f"📂 تم اختيار {len(test_files)} صورة سيارة جديدة كلياً (من {start_num} إلى {end_num}):")
    print(" ", test_files)
    print("=" * 85)

    print(f"{'#':<3} | {'الصورة':<10} | {'ثقة كشف اللوحة':<16} | {'قراءة اللوحة (3 موديلات)':<28} | {'المحافظة':<12} | {'الحالة'}")
    print("=" * 85)

    results_for_plot = []
    valid_count = 0

    for idx, fname in enumerate(test_files):
        img_path = os.path.join(vehicles_dir, fname)
        viz, plate_str, is_valid, gov, p_conf = process_single_vehicle(img_path)

        status_str = "✅ قياسية" if is_valid else "⚠️ غير قياسية"
        if is_valid:
            valid_count += 1

        print(f"{idx+1:<3} | {fname:<10} | {p_conf*100:>5.1f}%          | {plate_str:<28} | {gov:<12} | {status_str}")
        results_for_plot.append((fname, viz, plate_str, gov, is_valid))

    print("=" * 85)
    print(f"🎯 إجمالي اللوحات القياسية المتوافقة 100% مع قواعد المرور: {valid_count} من {len(test_files)} ({valid_count/len(test_files)*100:.1f}%)")
    print("=" * 85)

    # رسم النتائج في شبكة 5 صفوف × 4 أعمدة
    print("🎨 جاري رسم لوحة الشرف للـ 20 سيارة الجديدة...")
    fig, axes = plt.subplots(5, 4, figsize=(20, 16))
    fig.patch.set_facecolor('#0f172a')

    for i, (fname, viz, plate_str, gov, is_valid) in enumerate(results_for_plot):
        r = i // 4
        c = i % 4
        ax = axes[r, c]
        ax.set_facecolor('#1e293b')

        if viz is not None:
            ax.imshow(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB))
            t_color = '#4ade80' if is_valid else '#facc15'
            ax.set_title(f"{fname}: {plate_str}\nمرور {gov}", fontsize=11, weight='bold', color=t_color, pad=4)
        else:
            ax.text(0.5, 0.5, plate_str, ha='center', va='center', color='red', fontsize=11)
            ax.set_title(fname, fontsize=11, color='red')

        ax.axis('off')

    out_name = f"batch_results_20_vehicles_{start_num}_to_{end_num}.jpg"
    plt.suptitle(f"🔥 اختبار الـ 3 نماذج معاً على 20 سيارة جديدة كلياً ({start_num} - {end_num})",
                 fontsize=17, weight='bold', color='#f8fafc', y=0.995)
    plt.tight_layout()

    out_local = os.path.join(BASE_DIR, out_name)
    plt.savefig(out_local, dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"✅ تم حفظ الصورة المجمعة للـ 20 سيارة في: {out_local}")

    art_dir = r"C:\Users\Omar\.gemini\antigravity-ide\brain\80eb1de8-0e0c-401f-97c8-b20498617d02"
    if os.path.exists(art_dir):
        out_art = os.path.join(art_dir, out_name)
        plt.savefig(out_art, dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"✅ تم حفظ الصورة في مجلد الـ Artifacts: {out_art}")

if __name__ == '__main__':
    start_num = int(sys.argv[1]) if len(sys.argv) > 1 else 521
    run_test_20(start_num=start_num, count=20)
