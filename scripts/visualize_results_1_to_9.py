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

# تحميل النماذج
plate_model = YOLO(os.path.join(BASE_DIR, 'best.pt'))
loc_model   = YOLO(os.path.join(BASE_DIR, 'best_char.pt'))
recog_model = YOLO(os.path.join(BASE_DIR, 'best_char_afterTuning.pt'))

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

font_path = "C:/Windows/Fonts/tahoma.ttf"
if not os.path.exists(font_path):
    font_path = "C:/Windows/Fonts/arial.ttf"

def render_arabic_text(text: str) -> str:
    try:
        return get_display(arabic_reshaper.reshape(text))
    except Exception:
        return text

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
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = th.shape[:2]
    if h == 0 or w == 0: return cand_name
    
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

def verify_letter_geometry(patch, cand_name, all_detected_letters, candidate_list, all_letter_patches):
    if cand_name in ['jeem', 'yaa'] and 'qaaf' in all_detected_letters:
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

def read_egyptian_plate_multipass(fn):
    full_path = os.path.join(BASE_DIR, fn)
    if not os.path.exists(full_path):
        return None, "File Not Found", 0.0
    
    p = plate_model(full_path, verbose=False)[0]
    if len(p.boxes) == 0:
        return None, "No Plate Detected", 0.0
        
    b = p.boxes.xyxy[0]
    crop = p.orig_img[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
    crop_rot, angle = deskew_adaptive(crop)
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
    
    annotated = pass1.copy()
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
            label_text = char_map.get(verified, verified)
        else:
            verified = verify_letter_geometry(patch, cand, detected_letters, box_cands, all_letter_patches)
            lets.append(char_map.get(verified, '?'))
            label_text = char_map.get(verified, verified)
            
        color = (0, 220, 0) if is_digit else (255, 120, 0)
        cv2.rectangle(annotated, (lx1, ly1), (lx2, ly2), color, 3)
        
    # رسم الحروف العربية فوق البوكسات بدقة بواسطة PIL
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(annotated_rgb)
    draw = ImageDraw.Draw(pil_img)
    font_char = ImageFont.truetype(font_path, 22)
    
    for i, (lx1, ly1, lx2, ly2) in enumerate(clean_b):
        is_digit = (i <= split_idx)
        color = (0, 255, 100) if is_digit else (255, 160, 0)
        label_char = nums[i] if is_digit else lets[i - (split_idx + 1)]
        reshaped = render_arabic_text(label_char)
        # خلفية سوداء للنص لسهولة القراءة
        draw.rectangle([lx1, max(0, ly1 - 28), lx1 + 28, ly1], fill=(0, 0, 0, 220))
        draw.text((lx1 + 4, max(0, ly1 - 28)), reshaped, font=font_char, fill=color)

    annotated = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    num_txt = ' '.join(nums)
    let_txt = ' '.join(reversed(lets))
    result_text = f'[{num_txt}] | [{let_txt}]'
    return annotated, result_text, angle

def generate_montage():
    print("🚀 جاري معالجة اللوحات من 1 إلى 9 ورسم النتيجة بالكامل مع الموديل الجديد...")
    fig, axes = plt.subplots(3, 3, figsize=(17, 11))
    fig.patch.set_facecolor('#0f172a') # خلفية عصرية غامقة

    for idx in range(1, 10):
        img_name = f"{idx}.jpg"
        r = (idx - 1) // 3
        c = (idx - 1) % 3
        
        viz, plate_str, ang = read_egyptian_plate_multipass(img_name)
        print(f" 📷 {img_name:<6} ({ang:.1f}°) ➔ {plate_str}")
        
        ax = axes[r, c]
        ax.set_facecolor('#1e293b')
        
        if viz is not None:
            ax.imshow(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB))
            # تنسيق العنوان
            title_text = f"{img_name} ➔ {plate_str}"
            ax.set_title(title_text, fontsize=12, weight='bold', color='#38bdf8', pad=8)
        else:
            ax.text(0.5, 0.5, plate_str, ha='center', va='center', color='red', fontsize=12)
            ax.set_title(img_name, fontsize=12, color='red')
            
        ax.axis('off')

    plt.suptitle("🔥 نتائج الموديل المطور (best_char_afterTuning.pt) على اللوحات من 1 إلى 9", 
                 fontsize=16, weight='bold', color='#f8fafc', y=0.99)
    plt.tight_layout()
    
    out_local = os.path.join(BASE_DIR, 'batch_results_1_to_9_after_tuning.jpg')
    plt.savefig(out_local, dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"✅ تم حفظ الصورة المجمعة محلياً في: {out_local}")
    
    # حفظ في مجلد الـ Artifacts أيضاً للعرض المباشر
    art_dir = r"C:\Users\Omar\.gemini\antigravity-ide\brain\80eb1de8-0e0c-401f-97c8-b20498617d02"
    if os.path.exists(art_dir):
        out_art = os.path.join(art_dir, 'batch_results_1_to_9_after_tuning.jpg')
        plt.savefig(out_art, dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"✅ تم حفظ الصورة في مجلد الـ Artifacts: {out_art}")

if __name__ == '__main__':
    generate_montage()
