import cv2
import numpy as np
from ultralytics import YOLO

# =============================================================
# 1. تحميل النماذج وجدول المحارف
# =============================================================
plate_model = YOLO("best.pt")        # الموديل الأول: كاشف مكان اللوحة في الشارع/السيارة
loc_model   = YOLO("best_char.pt")   # الموديل الثاني: كاشف مواضع الخانات المادية (أرقام وحروف)
recog_model = YOLO("Tuning_char.pt") # الموديل الثالث: مصنف الحروف والأرقام

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
class_weight = {'qaaf': 25.0, 'faa': 2.5}

def recognize_and_annotate_plate(crop):
    """
    التعرف على محتويات اللوحة مع رسم صناديق الموديل الثاني (خضراء للأرقام وزرقاء للحروف)
    """
    h, w = crop.shape[:2]
    if h < 10 or w < 20: 
        return "", None
    
    # زووم فائق الدقة (Lanczos4 4x Zoom) لمساعدة الموديل الثاني والثالث على قراءة اللوحات البعيدة
    pass1 = cv2.resize(crop, (w * 4, h * 4), interpolation=cv2.INTER_LANCZOS4)
    w_tot = pass1.shape[1]
    
    # الموديل الثاني: تحديد الخانات المادية
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
        
    if not clean_b: 
        return "", pass1
    
    # الموديل الثالث: تصنيف الرموز
    c1 = recog_model(pass1, conf=0.003, verbose=False)[0]
    candidates = []
    for cb in c1.boxes:
        candidates.append((list(map(float, cb.xyxy[0])), recog_model.names[int(cb.cls[0])], float(cb.conf[0])))
        
    n = len(clean_b)
    split_idx = 3 if n >= 7 else (2 if n == 5 else n // 2)
    
    annotated_zoom = pass1.copy()
    nums, lets = [], []
    for i, (lx1, ly1, lx2, ly2) in enumerate(clean_b):
        is_digit = (i <= split_idx)
        best_cand, best_score = None, -1.0
        for (cx1, cy1, cx2, cy2), cname, raw_c in candidates:
            inter = max(0, min(lx2, cx2) - max(lx1, cx1))
            union = max(lx2, cx2) - min(lx1, cx1)
            iou = inter / union if union > 0 else 0
            if iou > 0.22:
                w_c = raw_c * class_weight.get(cname, 1.0)
                if is_digit and cname == 'ain': cname, w_c = '4', raw_c
                if is_digit and cname == 'alif': cname, w_c = '1', raw_c
                if (cname in digits) == is_digit and w_c > best_score:
                    best_score = w_c
                    best_cand = cname
                    
        # رسم صناديق الموديل الثاني والتصنيف
        color = (0, 220, 0) if is_digit else (255, 120, 0)
        cv2.rectangle(annotated_zoom, (lx1, ly1), (lx2, ly2), color, 2)
        cv2.putText(annotated_zoom, str(best_cand or '?'), (lx1 + 2, max(22, ly1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        
        ar = char_map.get(best_cand, '')
        if is_digit: nums.append(ar)
        else: lets.append(ar)
        
    num_str = ' '.join(nums)
    let_str = ' '.join(reversed(lets))
    result_text = f"[{num_str}] | [{let_str}]"
    return result_text, annotated_zoom

def process_video(video_path="Newvideo2.MOV", output_path="output_Newvideo.mp4", show_window=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ تعذر فتح الفيديو: {video_path}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"🎬 جاري تشغيل ومعالجة الفيديو: {video_path} ({w}x{h} @ {fps:.1f} FPS, {total_frames} إطار)")
    print("✨ ميزة الزووم المصغر (HUD Zoom Box) مفعلة لمشاهدة عمل الموديل الثاني لحظياً.")
    print("اضغط 'q' على نافذة العرض للخروج في أي وقت.\n" + "="*50)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    frame_count = 0
    cached_plate_text = ""
    cached_zoom_img = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        
        # 1. الموديل الأول: كشف مكان اللوحة على السيارة
        p_res = plate_model(frame, conf=0.35, verbose=False)[0]
        
        for box in p_res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]
            
            # معالجة اللوحة مع تحديث الزووم كل إطارين لتحقيق أعلى سلاسة وسرعة
            if frame_count % 2 == 0 or cached_zoom_img is None:
                cached_plate_text, cached_zoom_img = recognize_and_annotate_plate(crop)
                
            # رسم مستطيل اللوحة الخارجي على السيارة
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # كتابة قراءة اللوحة فوق السيارة
            label = cached_plate_text if cached_plate_text else "Plate Detected"
            cv2.rectangle(frame, (x1, max(0, y1 - 32)), (x1 + 240, y1), (0, 0, 0), -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
        # 2. رسم نافذة الزووم السينمائية (Picture-in-Picture HUD Zoom) في الزاوية العلوية
        if cached_zoom_img is not None:
            hud_w, hud_h = 320, 160
            zoomed = cv2.resize(cached_zoom_img, (hud_w, hud_h - 45))
            hud_canvas = np.zeros((hud_h, hud_w, 3), dtype=np.uint8)
            hud_canvas[:hud_h-45, :] = zoomed
            cv2.putText(hud_canvas, "ZOOM (Model 2 Boxes):", (8, hud_h - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1)
            cv2.putText(hud_canvas, cached_plate_text, (8, hud_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
            cv2.rectangle(hud_canvas, (0, 0), (hud_w - 1, hud_h - 1), (0, 255, 0), 2)
            
            # تثبيت الـ HUD في أعلى يسار الشاشة
            frame[20:20+hud_h, 20:20+hud_w] = hud_canvas
            
        out.write(frame)
        
        if show_window:
            display_frame = cv2.resize(frame, (int(w * 0.7), int(h * 0.7))) if w > 1000 or h > 1000 else frame
            cv2.imshow("Egyptian ALPR - Video Stream (Press 'q' to quit)", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("⏹ تم إيقاف التشغيل بواسطة المستخدم.")
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ تم الانتهاء بنجاح! تم حفظ الفيديو المعالج في: {output_path}")

if __name__ == "__main__":
    import os
    # اختيار ملف الفيديو المتوفر
    target_video = "Newvideo2.MOV" if os.path.exists("Newvideo2.MOV") else "Newvideo.MOV"
    process_video(target_video)
