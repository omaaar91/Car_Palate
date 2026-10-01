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
    
    # تصحيح الزاوية إذا كانت بين 1.8 و 32 درجة مع توسيع الكانفاس لحماية الحروف الطرفية
    if abs(rot_angle) > 1.8 and abs(rot_angle) < 32:
        rad = np.radians(rot_angle)
        sin_a = abs(np.sin(rad))
        cos_a = abs(np.cos(rad))
        new_w = int((h * sin_a) + (w * cos_a))
        new_h = int((h * cos_a) + (w * sin_a))

        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), rot_angle, 1.0)
        M[0, 2] += (new_w / 2.0) - (w / 2.0)
        M[1, 2] += (new_h / 2.0) - (h / 2.0)

        deskewed = cv2.warpAffine(
            crop, M, (new_w, new_h), 
            flags=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_REPLICATE
        )
        return deskewed, rot_angle
        
    return crop, 0.0
