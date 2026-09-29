import cv2
import numpy as np

def deskew_adaptive(crop: np.ndarray) -> tuple[np.ndarray, float]:
    """
    محاذاة اللوحة وتصحيح زاوية الميلان تكيفياً باستخدام أدنى مستطيل محيط
    """
    h, w = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    if not cnts:
        return crop, 0.0
        
    largest = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    (cx, cy), (rw, rh), angle = rect
    
    rot_angle = -(90 - angle) if rw < rh else angle
    
    # تصحيح الزاوية إذا كانت بين 1.8 و 32 درجة
    if abs(rot_angle) > 1.8 and abs(rot_angle) < 32:
        M = cv2.getRotationMatrix2D((w // 2, h // 2), rot_angle, 1.0)
        deskewed = cv2.warpAffine(
            crop, M, (w, h), 
            flags=cv2.INTER_CUBIC, 
            borderMode=cv2.BORDER_REPLICATE
        )
        return deskewed, rot_angle
        
    return crop, 0.0
