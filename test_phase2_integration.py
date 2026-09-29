import os
import sys
import cv2
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

from core.config import (
    PLATE_MODEL_PATH, LOC_MODEL_PATH, RECOG_MODEL_PATH,
    CHAR_MAP, DIGITS_SET, CLASS_WEIGHTS,
    OFFICIAL_ALLOWED_LETTERS, CANONICAL_LETTER_MAP,
    SCALE_FACTOR, LOC_CONF, RECOG_CONF
)
from core.deskew import deskew_adaptive
from ultralytics import YOLO

plate_model = YOLO(PLATE_MODEL_PATH)
loc_model   = YOLO(LOC_MODEL_PATH)
recog_model = YOLO(RECOG_MODEL_PATH)

def validate_egyptian_syntax(digits: list, letters: list) -> dict:
    d_count = len(digits)
    l_count = len(letters)
    total = d_count + l_count

    has_zero = any(d in {'0', '٠'} for d in digits)
    all_legal = True
    for l in letters:
        ar_char = CHAR_MAP.get(l, l)
        if ar_char not in OFFICIAL_ALLOWED_LETTERS:
            all_legal = False
            break

    if d_count == 3 and l_count == 3:
        gov = "القاهرة"
        code = "CAIRO_3D_3L"
        is_valid = (not has_zero) and all_legal
    elif d_count == 4 and l_count == 2:
        gov = "الجيزة"
        code = "GIZA_4D_2L"
        is_valid = (not has_zero) and all_legal
    elif d_count == 4 and l_count == 3:
        gov = "المحافظات"
        code = "PROVINCES_4D_3L"
        is_valid = (not has_zero) and all_legal
    else:
        gov = "غير قياسي"
        code = f"NON_STANDARD_{d_count}D_{l_count}L"
        is_valid = False

    return {
        "is_valid": is_valid,
        "governorate": gov,
        "format_code": code,
        "digits_count": d_count,
        "letters_count": l_count,
        "total_chars": total,
        "has_zero": has_zero,
        "all_legal_letters": all_legal,
        "badge": f"مرور {gov} ({d_count} أرقام - {l_count} حروف)" if is_valid else f"لوحة غير قياسية ({d_count} أرقام - {l_count} حروف)"
    }

def prune_spurious_boxes(clean_b: list, candidates: list, plate_w: int, plate_h: int) -> list:
    if len(clean_b) <= 7:
        return clean_b

    clean_b = list(clean_b)
    while len(clean_b) > 7:
        widths = [b[2] - b[0] for b in clean_b]
        heights = [b[3] - b[1] for b in clean_b]
        cy_list = [(b[1] + b[3]) / 2.0 for b in clean_b]

        med_h = np.median(heights)
        med_w = np.median(widths)
        med_cy = np.median(cy_list)

        outlier_scores = []
        for idx, (bx1, by1, bx2, by2) in enumerate(clean_b):
            bw = bx2 - bx1
            bh = by2 - by1
            bcy = (by1 + by2) / 2.0

            y_dev = abs(bcy - med_cy) / max(1.0, med_h)
            max_cand_conf = 0.0
            for (cx1, cy1, cx2, cy2), cname, conf in candidates:
                inter = max(0, min(bx2, cx2) - max(bx1, cx1))
                union = max(bx2, cx2) - min(bx1, cx1)
                iou = inter / union if union > 0 else 0
                if iou > 0.18:
                    max_cand_conf = max(max_cand_conf, conf)

            edge_pen = 1.5 if (bx1 < plate_w * 0.03 or bx2 > plate_w * 0.97) else 1.0
            anomaly = (y_dev * 2.0) + (1.0 - max_cand_conf) * 1.5 + (abs(bw - med_w) / max(1.0, med_w)) * 0.8
            outlier_scores.append((anomaly * edge_pen, idx))

        outlier_scores.sort(reverse=True)
        worst_idx = outlier_scores[0][1]
        clean_b.pop(worst_idx)

    return clean_b

def resolve_split_index(clean_b: list, candidates: list) -> int:
    n = len(clean_b)
    if n >= 7:
        return 3
    elif n <= 4:
        return max(0, n // 2 - 1)
    elif n == 5:
        return 2

    # n == 6: Cairo (split 2) vs Giza (split 3)
    g2 = clean_b[3][0] - clean_b[2][2]
    g3 = clean_b[4][0] - clean_b[3][2]

    b3_x1, b3_y1, b3_x2, b3_y2 = clean_b[3]
    b3_w = b3_x2 - b3_x1
    b3_cx = (b3_x1 + b3_x2) / 2.0

    digit_score = 0.0
    letter_score = 0.0

    for (cx1, cy1, cx2, cy2), cname, raw_c in candidates:
        inter = max(0, min(b3_x2, cx2) - max(b3_x1, cx1))
        union = max(b3_x2, cx2) - min(b3_x1, cx1)
        iou = inter / union if union > 0 else 0
        ccx = (cx1 + cx2) / 2.0
        if iou > 0.18 or (abs(b3_cx - ccx) < b3_w * 0.48 and inter > 0):
            if cname in DIGITS_SET:
                digit_score = max(digit_score, raw_c)
            else:
                letter_score = max(letter_score, raw_c)

    if digit_score > letter_score + 0.15:
        return 3
    elif letter_score > digit_score + 0.15:
        return 2
    else:
        return 2 if g2 > g3 else 3

def verify_digit_geometry(patch: np.ndarray, cand_name: str, cand_score: float, candidate_list: list) -> str:
    # خريطة تحويل الحروف للأرقام المقابلة عند الخطأ في خانة رقمية
    LETTER_TO_DIGIT = {
        'ain': '4', 'alif': '1', 'laam': '1',
        'haa': '5', 'waw': '9', 'seen': '3',
        'sheen': '3', 'baa': '2', 'taa': '2', 'thaa': '2'
    }
    if cand_name in LETTER_TO_DIGIT:
        cand_name = LETTER_TO_DIGIT[cand_name]

    if cand_name == '0':
        alt_digits = [c[1] for c in candidate_list if c[1] in {'1', '2', '3', '4', '5', '6', '7', '8', '9'}]
        cand_name = alt_digits[0] if alt_digits else '5'

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = th.shape[:2]
    if h == 0 or w == 0:
        return cand_name

    cand_names_only = [c[1] for c in candidate_list]

    v_prof = np.sum(th > 0, axis=0)
    col_idx = np.where(v_prof > 0)[0]
    span = col_idx[-1] - col_idx[0] if len(col_idx) else w
    aspect_stroke = span / max(1, h)

    if cand_name == '2' and aspect_stroke < 0.34 and ('1' in cand_names_only or 'alif' in cand_names_only or aspect_stroke < 0.28):
        return '1'

    if cand_name == '3' and cand_score < 0.60 and ('6' in cand_names_only or '7' in cand_names_only or len(cand_names_only) < 3):
        return '7'

    return cand_name

def verify_letter_geometry(patch: np.ndarray, cand_name: str, all_detected_letters: list,
                           candidate_list: list, all_letter_patches: list) -> str:
    DIGIT_TO_LETTER = {
        '4': 'ain', '1': 'alif', '5': 'haa',
        '0': 'haa', '9': 'waw', '3': 'seen', '2': 'baa'
    }
    if cand_name in DIGIT_TO_LETTER:
        cand_name = DIGIT_TO_LETTER[cand_name]

    if cand_name in CANONICAL_LETTER_MAP:
        cand_name = CANONICAL_LETTER_MAP[cand_name]

    if cand_name in ['jeem', 'yaa', 'baa'] and 'qaaf' in all_detected_letters:
        g_cur = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        r_cur = cv2.resize(g_cur, (30, 40))
        for other_name, other_patch in all_letter_patches:
            if other_name == 'qaaf':
                g_other = cv2.cvtColor(other_patch, cv2.COLOR_BGR2GRAY)
                r_other = cv2.resize(g_other, (30, 40))
                corr = cv2.matchTemplate(r_cur, r_other, cv2.TM_CCOEFF_NORMED)[0][0]
                if corr > 0.58:
                    return 'qaaf'

    return cand_name

def test_pipeline(img):
    p_res = plate_model(img, conf=0.35, verbose=False)[0]
    if not p_res.boxes:
        return None
    bx = p_res.boxes[0]
    x1, y1, x2, y2 = map(int, bx.xyxy[0])
    crop = img[y1:y2, x1:x2]

    crop_rot, angle = deskew_adaptive(crop)
    h, w = crop_rot.shape[:2]
    if h < 10 or w < 20: return None

    pass1 = cv2.resize(crop_rot, (int(w * SCALE_FACTOR), int(h * SCALE_FACTOR)), interpolation=cv2.INTER_LANCZOS4)
    w_tot = pass1.shape[1]
    pass2 = cv2.resize(crop_rot, (int(w * (SCALE_FACTOR * 1.2)), int(h * (SCALE_FACTOR * 1.2))), interpolation=cv2.INTER_LANCZOS4)
    scale2 = pass2.shape[1] / w_tot

    l_res = loc_model(pass1, conf=LOC_CONF, verbose=False)[0]
    raw_b = sorted([list(map(int, bx.xyxy[0][:4])) for bx in l_res.boxes], key=lambda x: x[0])

    clean_b = []
    for b in raw_b:
        bw = b[2] - b[0]
        bh = b[3] - b[1]
        if b[0] < (w_tot * 0.04) and b[0] <= 8: continue
        if b[2] >= (w_tot * 0.985) and (bh / max(1, bw) > 2.5 or bw < w_tot * 0.04): continue
        if bw < (w_tot * 0.03) or (bh / max(1, bw)) > 3.5: continue
        clean_b.append(b)

    c1 = recog_model(pass1, conf=RECOG_CONF, verbose=False)[0]
    c2 = recog_model(pass2, conf=RECOG_CONF, verbose=False)[0]

    candidates = []
    for cb in c1.boxes:
        candidates.append((list(map(float, cb.xyxy[0])), recog_model.names[int(cb.cls[0])], float(cb.conf[0]) * 1.0))
    for cb in c2.boxes:
        coords = [float(cb.xyxy[0][0])/scale2, float(cb.xyxy[0][1])/scale2, float(cb.xyxy[0][2])/scale2, float(cb.xyxy[0][3])/scale2]
        candidates.append((coords, recog_model.names[int(cb.cls[0])], float(cb.conf[0]) * 0.95))

    clean_b = prune_spurious_boxes(clean_b, candidates, pass1.shape[1], pass1.shape[0])
    split_idx = resolve_split_index(clean_b, candidates)

    stage1_cands, stage1_scores, box_cands_map = [], [], []
    for i, (lx1, ly1, lx2, ly2) in enumerate(clean_b):
        is_digit = (i <= split_idx)
        best_cand, best_score = None, -1.0
        cands_for_box = []
        bw = lx2 - lx1
        lcx = (lx1 + lx2) / 2.0
        for (cx1, cy1, cx2, cy2), cname, raw_c in candidates:
            inter = max(0, min(lx2, cx2) - max(lx1, cx1))
            union = max(lx2, cx2) - min(lx1, cx1)
            iou = inter / union if union > 0 else 0
            ccx = (cx1 + cx2) / 2.0
            center_dist = abs(lcx - ccx)
            if iou > 0.18 or (center_dist < bw * 0.48 and inter > 0):
                w_c = raw_c * CLASS_WEIGHTS.get(cname, 1.0)
                if is_digit and cname == 'ain': cname, w_c = '4', raw_c
                if is_digit and cname == 'alif': cname, w_c = '1', raw_c
                cands_for_box.append((w_c, cname))
                if (cname in DIGITS_SET) == is_digit and w_c > best_score:
                    best_score = w_c
                    best_cand = cname
        stage1_cands.append(best_cand)
        stage1_scores.append(best_score)
        box_cands_map.append(cands_for_box)

    detected_letters = [c for i, c in enumerate(stage1_cands) if i > split_idx]
    all_letter_patches = [(stage1_cands[i], pass1[ly1:ly2, lx1:lx2]) for i, (lx1, ly1, lx2, ly2) in enumerate(clean_b) if i > split_idx]

    nums, lets = [], []
    for i, (lx1, ly1, lx2, ly2) in enumerate(clean_b):
        is_digit = (i <= split_idx)
        cand = stage1_cands[i]
        score = stage1_scores[i]
        patch = pass1[ly1:ly2, lx1:lx2]
        box_cands = box_cands_map[i]

        if is_digit:
            verified = verify_digit_geometry(patch, cand, score, box_cands)
            nums.append(CHAR_MAP.get(verified, '?'))
        else:
            verified = verify_letter_geometry(patch, cand, detected_letters, box_cands, all_letter_patches)
            lets.append(CHAR_MAP.get(verified, '?'))

    syntax_res = validate_egyptian_syntax(nums, list(reversed(lets)))
    return nums, list(reversed(lets)), syntax_res

images = ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', '6.jpg', '7.jpg', '8.jpg', '9.jpg', 'test1.jpg', 'test2.jpg', 'test3.jpg', 'test4.png']

print("=== Phase 2 Validation Test on All 13 Images ===")
for img_name in images:
    img = cv2.imread(img_name)
    nums, lets, syntax = test_pipeline(img)
    num_txt = " ".join(nums)
    let_txt = " ".join(lets)
    valid_icon = "🟢" if syntax['is_valid'] else "🔴"
    print(f"{img_name:10s} -> [{num_txt}] | [{let_txt}]  {valid_icon} {syntax['badge']}")
