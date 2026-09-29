import cv2
import numpy as np

def verify_digit_geometry(patch: np.ndarray, cand_name: str, cand_score: float, candidate_list: list) -> str:
    """
    التحقق الهندسي المجهري للأرقام الملتبسة:
    1. حسم 1 ضد 2 بالاعتماد على نحافة العمود الرأسي ونسبة العرض للارتفاع.
    2. حسم 7 ضد 3 بالاعتماد على مستوى الثقة والمنافسة الهيكلية.
    """
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = th.shape[:2]
    if h == 0 or w == 0:
        return cand_name
        
    cand_names_only = [c[1] for c in candidate_list]
    
    # 1. التحقق من رقم 1 ضد 2:
    # رقم 1 هو عمود رأسي نحيف، ونسبة عرضه إلى ارتفاعه لا تتجاوز 0.34
    v_prof = np.sum(th > 0, axis=0)
    col_idx = np.where(v_prof > 0)[0]
    span = col_idx[-1] - col_idx[0] if len(col_idx) else w
    aspect_stroke = span / max(1, h)
    
    if cand_name == '2' and aspect_stroke < 0.34 and ('1' in cand_names_only or 'alif' in cand_names_only or aspect_stroke < 0.28):
        return '1'
        
    # 2. التحقق من رقم 7 ضد 3:
    # رقم 3 الحقيقي ثقته عالية (> 0.60). أما رقم 7 المائل فيعطي ثقة منخفضة لـ 3 ويتنافس مع 6/7
    if cand_name == '3' and cand_score < 0.60 and ('6' in cand_names_only or '7' in cand_names_only or len(cand_names_only) < 3):
        return '7'
            
    return cand_name

def verify_letter_geometry(patch: np.ndarray, cand_name: str, all_detected_letters: list, 
                           candidate_list: list, all_letter_patches: list) -> str:
    """
    التحقق الهندسي والسياقي للحروف وتطابق التوائم (مثل توأم ق ق):
    إذا تم تأكيد حرف قاف على اللوحة، يُقاس معامل الارتباط البصري مع الحرف المجاور.
    """
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
