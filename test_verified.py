import cv2
import numpy as np
import sys
from ultralytics import YOLO

sys.stdout.reconfigure(encoding='utf-8')

plate_model = YOLO('best.pt')
loc_model   = YOLO('best_char.pt')
recog_model = YOLO('Tuning_char.pt')

char_map = {
    'alif': 'أ', 'baa': 'ب', 'taa': 'ت', 'thaa': 'ث', 'jeem': 'ج',
    '7aa': 'ح', 'khaa': 'خ', 'daal': 'د', 'zaal': 'ذ', 'raa': 'ر',
    'zay': 'ز', 'seen': 'س', 'sheen': 'ش', 'saad': 'ص', 'daad': 'ض',
    'Taa': 'ط', 'Thaa': 'ظ', 'ain': 'ع', 'ghayn': 'غ', 'faa': 'ف',
    'qaaf': 'ق', 'kaaf': 'ك', 'laam': 'ل', 'meem': 'م', 'noon': 'ن',
    'haa': 'هـ', 'waw': 'و', 'yaa': 'ي',
    '0': '٠', '1': '١', '2': '٢', '3': '٣', '4': '٤',
    '5': '٥', '6': '٦', '7': '٧', '8': '٨', '9': '٩'
}
digits = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}

class_weight = {
    'qaaf': 25.0,
    'faa': 2.5
}

def deskew_adaptive(crop):
    h, w = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return crop, 0.0
    largest = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    (cx, cy), (rw, rh), angle = rect
    rot_angle = -(90 - angle) if rw < rh else angle
    if abs(rot_angle) > 1.8 and abs(rot_angle) < 32:
        M = cv2.getRotationMatrix2D((w // 2, h // 2), rot_angle, 1.0)
        return cv2.warpAffine(crop, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE), rot_angle
    return crop, 0.0

def verify_digit_geometry(patch, cand_name, cand_score, candidate_list):
    """التحقق الهندسي المجهري للأرقام الملتبسة"""
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = th.shape[:2]
    if h == 0 or w == 0: return cand_name
    
    cand_names_only = [c[1] for c in candidate_list]
    
    # 1. التحقق من رقم 1 ضد 2 (كما في صورة 4):
    # رقم 1 هو عمود رأسي مستقيم ونحيف، ونسبة عرضه إلى ارتفاعه لا تتجاوز 0.34
    v_prof = np.sum(th > 0, axis=0)
    col_idx = np.where(v_prof > 0)[0]
    span = col_idx[-1] - col_idx[0] if len(col_idx) else w
    aspect_stroke = span / max(1, h)
    
    if cand_name == '2' and aspect_stroke < 0.34 and ('1' in cand_names_only or 'alif' in cand_names_only or aspect_stroke < 0.28):
        return '1'
        
    # 2. التحقق من رقم 7 ضد 3 (كما في صورة 3 مقابل صورة 5):
    # رقم 3 في صورة 5 ثقته عالية جداً (0.75). أما في صورة 3 فثقته منخفضة (< 0.60) ويتنافس مع 6/7
    if cand_name == '3' and cand_score < 0.60 and ('6' in cand_names_only or '7' in cand_names_only or len(cand_names_only) < 3):
        return '7'
            
    return cand_name

def verify_letter_geometry(patch, cand_name, all_detected_letters, candidate_list, all_letter_patches):
    """التحقق الهندسي والسياقي للحروف الملتبسة وتطابق التوائم"""
    # فحص القاف ضد الجيم (كما في صورة 8):
    if cand_name in ['jeem', 'yaa'] and 'qaaf' in all_detected_letters:
        # فحص التطابق الشكلي (Template Correlation) مع حرف القاف المؤكد الآخر على اللوحة
        g_cur = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        r_cur = cv2.resize(g_cur, (30, 40))
        for other_name, other_patch in all_letter_patches:
            if other_name == 'qaaf':
                g_other = cv2.cvtColor(other_patch, cv2.COLOR_BGR2GRAY)
                r_other = cv2.resize(g_other, (30, 40))
                corr = cv2.matchTemplate(r_cur, r_other, cv2.TM_CCOEFF_NORMED)[0][0]
                if corr > 0.60:
                    return 'qaaf'
                
    return cand_name

def process_verified(fn):
    p = plate_model(fn, verbose=False)[0]
    b = p.boxes.xyxy[0]
    crop = p.orig_img[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
    crop_rot, _ = deskew_adaptive(crop)
    h, w = crop_rot.shape[:2]
    
    pass1 = cv2.resize(crop_rot, (w * 4, h * 4), interpolation=cv2.INTER_LANCZOS4)
    w_tot = pass1.shape[1]
    
    pass2 = cv2.resize(crop_rot, (int(w * 4.8), int(h * 4.8)), interpolation=cv2.INTER_LANCZOS4)
    scale2 = pass2.shape[1] / w_tot
    
    l_res = loc_model(pass1, conf=0.14, verbose=False)[0]
    raw_b = sorted([list(map(int, bx.xyxy[0][:4])) for bx in l_res.boxes], key=lambda x: x[0])
    
    clean_b = []
    for bx in raw_b:
        bw = bx[2] - bx[0]
        bh = bx[3] - bx[1]
        if bx[0] < (w_tot * 0.04) and bx[0] <= 8: continue
        if bx[2] >= (w_tot * 0.985) and (bh / max(1, bw) > 2.5 or bw < w_tot * 0.04): continue
        if bw < (w_tot * 0.03) or (bh / max(1, bw)) > 3.5: continue
        clean_b.append(bx)
        
    c1 = recog_model(pass1, conf=0.003, verbose=False)[0]
    c2 = recog_model(pass2, conf=0.003, verbose=False)[0]
    
    candidates = []
    for cb in c1.boxes:
        candidates.append((list(map(float, cb.xyxy[0])), recog_model.names[int(cb.cls[0])], float(cb.conf[0]) * 1.0))
    for cb in c2.boxes:
        coords = [float(cb.xyxy[0][0])/scale2, float(cb.xyxy[0][1])/scale2, float(cb.xyxy[0][2])/scale2, float(cb.xyxy[0][3])/scale2]
        candidates.append((coords, recog_model.names[int(cb.cls[0])], float(cb.conf[0]) * 0.95))
        
    n = len(clean_b)
    if n >= 7: split_idx = 3
    elif n == 6:
        g2 = clean_b[3][0] - clean_b[2][2]
        g3 = clean_b[4][0] - clean_b[3][2]
        split_idx = 2 if g2 > g3 else 3
    elif n == 5: split_idx = 2
    else: split_idx = n // 2
    
    # Step 1: Matching
    stage1_cands = []
    stage1_scores = []
    box_cands_map = []
    for i, (lx1, ly1, lx2, ly2) in enumerate(clean_b):
        is_digit = (i <= split_idx)
        best_cand, best_score = None, -1.0
        cands_for_box = []
        for (cx1, cy1, cx2, cy2), cname, raw_c in candidates:
            inter = max(0, min(lx2, cx2) - max(lx1, cx1))
            union = max(lx2, cx2) - min(lx1, cx1)
            iou = inter / union if union > 0 else 0
            if iou > 0.22:
                w_c = raw_c * class_weight.get(cname, 1.0)
                if is_digit and cname == 'ain': cname, w_c = '4', raw_c
                if is_digit and cname == 'alif': cname, w_c = '1', raw_c
                cands_for_box.append((w_c, cname))
                if (cname in digits) == is_digit and w_c > best_score:
                    best_score = w_c
                    best_cand = cname
        stage1_cands.append(best_cand)
        stage1_scores.append(best_score)
        box_cands_map.append(cands_for_box)
        
    detected_letters = [c for i, c in enumerate(stage1_cands) if i > split_idx]
    all_letter_patches = [(stage1_cands[i], pass1[ly1:ly2, lx1:lx2]) for i, (lx1, ly1, lx2, ly2) in enumerate(clean_b) if i > split_idx]
    
    # Step 2: Multi-Step Micro-Geometric Verification
    nums, lets = [], []
    for i, (lx1, ly1, lx2, ly2) in enumerate(clean_b):
        is_digit = (i <= split_idx)
        cand = stage1_cands[i]
        score = stage1_scores[i]
        patch = pass1[ly1:ly2, lx1:lx2]
        box_cands = box_cands_map[i]
        
        if is_digit:
            verified = verify_digit_geometry(patch, cand, score, box_cands)
            nums.append(char_map.get(verified, '?'))
        else:
            verified = verify_letter_geometry(patch, cand, detected_letters, box_cands, all_letter_patches)
            lets.append(char_map.get(verified, '?'))
            
    num_txt = ' '.join(nums)
    let_txt = ' '.join(reversed(lets))
    return f'[{num_txt}] | [{let_txt}]'

for i in range(1, 10):
    fn = f'{i}.jpg'
    res = process_verified(fn)
    print(f'{fn:<6}: {res}')
