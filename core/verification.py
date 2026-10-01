import cv2
import numpy as np
from core.config import (
    CANONICAL_LETTER_MAP, 
    OFFICIAL_ALLOWED_LETTERS, 
    CHAR_MAP, 
    DIGITS_SET
)

def validate_egyptian_syntax(digits: list, letters: list) -> dict:
    """
    التحقق من صحة الهيكل القانوني للوحة المركبة المصرية وفقاً لتعليمات الإدارة العامة للمرور:
    - القاهرة: 3 أرقام + 3 حروف (إجمالي 6 خانات)
    - الجيزة: 4 أرقام + 2 حروف (إجمالي 6 خانات)
    - باقي المحافظات: 4 أرقام + 3 حروف (إجمالي 7 خانات)
    - استبعاد أي لوحة تحتوي على الرقم 0 (محظور مرورياً)
    - التحقق من انتماء كافة الحروف إلى الـ 17 حرفاً الرسمية
    """
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
def suppress_duplicate_slots(boxes: list, iou_thresh: float = 0.35) -> list:
    """
    إزالة الصناديق المتكررة لنفس الخانة (Duplicate Slot Suppression):
    تمنع تكرار نفس الحرف أو الرقم بسبب تداخل صندوقين لنفس الخانة.
    إذا توفرت درجة الثقة ترتب تنازلياً للاحتفاظ بالصندوق الأكثر دقة وثقة.
    """
    if not boxes:
        return []
    
    # دعم كل من [x1, y1, x2, y2] أو [x1, y1, x2, y2, conf]
    has_conf = (len(boxes[0]) >= 5)
    if has_conf:
        sorted_b = sorted(boxes, key=lambda b: float(b[4]), reverse=True)
    else:
        sorted_b = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)

    keep = []
    for b in sorted_b:
        overlap = False
        bcx = (b[0] + b[2]) / 2.0
        bw = b[2] - b[0]
        for k in keep:
            inter = max(0, min(b[2], k[2]) - max(b[0], k[0]))
            union = max(b[2], k[2]) - min(b[0], k[0])
            iou = inter / union if union > 0 else 0
            kcx = (k[0] + k[2]) / 2.0
            kw = k[2] - k[0]
            if iou > iou_thresh or abs(bcx - kcx) < min(bw, kw) * 0.52:
                overlap = True
                break
        if not overlap:
            keep.append(b)

    keep_sorted = sorted(keep, key=lambda b: b[0])
    return [[int(b[0]), int(b[1]), int(b[2]), int(b[3])] for b in keep_sorted]

def suppress_duplicate_detections(detections: list, iou_thresh: float = 0.30) -> list:
    """
    تنقية الصناديق المتداخلة للمحارف المكتشفة مباشرة (Direct Character NMS):
    تحتفظ بالمحرف صاحب أعلى ثقة وتمنع الازدواجية على نفس الحرف أو الرقم.
    """
    if not detections:
        return []
    sorted_d = sorted(detections, key=lambda d: float(d['conf']), reverse=True)
    keep = []
    for d in sorted_d:
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
    return sorted(keep, key=lambda d: d['box'][0])

def prune_spurious_boxes(clean_b: list, candidates: list, plate_w: int, plate_h: int) -> list:
    """
    تنقية الصناديق الزائفة (Spurious Box Pruning):
    إذا رصد كاشف الخانات أكثر من 7 صناديق (بسبب مسامير التثبيت، الشعار، أو انعكاسات الإطار)،
    يتم استبعاد الصناديق الأكثر شذوذاً عن الخط الأفقي والأقل ثقة في تصنيف المحارف حتى نصل لـ 7 خانات قانونية كحد أقصى.
    """
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

            # 1. الانحراف الرأسي عن خط المحاذاة الأفقي
            y_dev = abs(bcy - med_cy) / max(1.0, med_h)

            # 2. أقصى درجة ثقة لمرشح يتقاطع مع الصندوق
            max_cand_conf = 0.0
            for (cx1, cy1, cx2, cy2), cname, conf in candidates:
                inter = max(0, min(bx2, cx2) - max(bx1, cx1))
                union = max(bx2, cx2) - min(bx1, cx1)
                iou = inter / union if union > 0 else 0
                if iou > 0.18:
                    max_cand_conf = max(max_cand_conf, conf)

            # 3. عقوبة الصناديق الملاصقة لحافة اللوحة بدون محرف قوي
            edge_penalty = 1.5 if (bx1 < plate_w * 0.03 or bx2 > plate_w * 0.97) else 1.0

            # حساب درجة الشذوذ المركبة
            anomaly = (y_dev * 2.0) + (1.0 - max_cand_conf) * 1.5 + (abs(bw - med_w) / max(1.0, med_w)) * 0.8
            outlier_scores.append((anomaly * edge_penalty, idx))

        outlier_scores.sort(reverse=True)
        worst_idx = outlier_scores[0][1]
        clean_b.pop(worst_idx)

    return clean_b

def resolve_split_index(clean_b: list, candidates: list) -> int:
    """
    تحديد الحد الفاصل الحتمي بين الأرقام والحروف:
    إذا كان عدد الخانات 6:
    - المفاضلة الدقيقة بين القاهرة (3 أرقام + 3 حروف) والجيزة (4 أرقام + 2 حروف)
    - عبر دمج قياس الفجوات الهندسية (Gaps) مع الاحتمالية التصنيفية للخانة الرابعة (رقم أم حرف).
    """
    n = len(clean_b)
    if n >= 7:
        return 3  # 4 أرقام (0, 1, 2, 3) والباقي حروف
    elif n <= 4:
        return max(0, n // 2 - 1)
    elif n == 5:
        return 2  # الافتراضي 3 أرقام + 2 حروف

    # حالة n == 6: القاهرة (split_idx = 2) ضد الجيزة (split_idx = 3)
    g2 = clean_b[3][0] - clean_b[2][2] # الفجوة بعد الخانة 3
    g3 = clean_b[4][0] - clean_b[3][2] # الفجوة بعد الخانة 4

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

    # إذا كان محتوى الخانة الرابعة رقماً قوياً، فاللوحة جيزة (4 أرقام)
    if digit_score > letter_score + 0.15:
        return 3
    elif letter_score > digit_score + 0.15:
        return 2
    else:
        # الاعتماد على الفجوة الأوسع
        return 2 if g2 > g3 else 3

def verify_digit_geometry(patch: np.ndarray, cand_name: str, cand_score: float, candidate_list: list) -> str:
    """
    التحقق الهندسي المجهري للأرقام الملتبسة واستبعاد الصفر الممنوع:
    1. تصحيح الخلط بين الحروف والأرقام في خانة الأرقام (مثل عين -> 4، ألف/لام -> 1، هاء -> 5).
    2. استبعاد الرقم صفر (0) قانونياً واستبداله بأقرب رقم منافس أو 5.
    3. حسم 1 ضد 2 بالاعتماد على نحافة السكتة الرأسية ونسبة العرض للارتفاع.
    4. حسم 7 ضد 3 بالاعتماد على مستوى الثقة والمنافسة الهيكلية.
    """
    # تصحيح التسميات الحرفية التي تقع بالخطأ في خانات الأرقام
    LETTER_TO_DIGIT = {
        'ain': '4', 'alif': '1', 'laam': '1',
        'haa': '5', 'waw': '9', 'seen': '3',
        'sheen': '3', 'baa': '2', 'taa': '2', 'thaa': '2'
    }
    if cand_name in LETTER_TO_DIGIT:
        cand_name = LETTER_TO_DIGIT[cand_name]

    # استبعاد الرقم صفر لأنه ممنوع مرورياً في اللوحات المصرية
    if cand_name == '0':
        alt_digits = [c[1] for c in candidate_list if c[1] in {'1', '2', '3', '4', '5', '6', '7', '8', '9'}]
        cand_name = alt_digits[0] if alt_digits else '5'

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = th.shape[:2]
    if h == 0 or w == 0:
        return cand_name

    cand_names_only = [c[1] for c in candidate_list]

    # 1. التحقق من رقم 1 ضد 2:
    v_prof = np.sum(th > 0, axis=0)
    col_idx = np.where(v_prof > 0)[0]
    span = col_idx[-1] - col_idx[0] if len(col_idx) else w
    aspect_stroke = span / max(1, h)

    if cand_name == '2' and aspect_stroke < 0.34 and ('1' in cand_names_only or 'alif' in cand_names_only or aspect_stroke < 0.28):
        return '1'

    # 2. التحقق من رقم 7 ضد 3:
    if cand_name == '3' and cand_score < 0.60 and ('6' in cand_names_only or '7' in cand_names_only or len(cand_names_only) < 3):
        return '7'

    return cand_name

def verify_letter_geometry(patch: np.ndarray, cand_name: str, all_detected_letters: list, 
                           candidate_list: list, all_letter_patches: list) -> str:
    """
    التحقق الهندسي والسياقي للحروف وتطبيق القواعد الرسمية المصرية:
    1. تصحيح الأرقام التي تقع في خانات الحروف (مثل 4 -> عين، 1 -> ألف، 5 -> هاء).
    2. تطبيق جدول التحويل الرسمي للحروف المحظورة قانوناً في المرور المصري.
    3. التحقق من تطابق التوائم (مثل توأم ق ق) بمعامل الارتباط البصري.
    """
    # تصحيح التسميات الرقمية التي تقع في خانات الحروف
    DIGIT_TO_LETTER = {
        '4': 'ain', '1': 'alif', '5': 'haa',
        '0': 'haa', '9': 'waw', '3': 'seen', '2': 'baa'
    }
    if cand_name in DIGIT_TO_LETTER:
        cand_name = DIGIT_TO_LETTER[cand_name]

    # تطبيق جدول التحويل الرسمي للحروف المحظورة
    if cand_name in CANONICAL_LETTER_MAP:
        cand_name = CANONICAL_LETTER_MAP[cand_name]

    # تطابق التوائم (مثل ق ق)
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
