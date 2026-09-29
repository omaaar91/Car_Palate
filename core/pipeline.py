import cv2
import numpy as np
from ultralytics import YOLO

from core.config import (
    PLATE_MODEL_PATH, LOC_MODEL_PATH, RECOG_MODEL_PATH,
    CHAR_MAP, DIGITS_SET, CLASS_WEIGHTS,
    PLATE_CONF, LOC_CONF, RECOG_CONF, SCALE_FACTOR
)
from core.deskew import deskew_adaptive
from core.verification import (
    validate_egyptian_syntax, 
    prune_spurious_boxes, 
    resolve_split_index, 
    verify_digit_geometry, 
    verify_letter_geometry
)

class EgyptianALPR:
    """
    نظام التعرف الذكي متعدد المراحل على لوحات المركبات المصرية
    (Multi-Stage Egyptian License Plate Recognition Engine)
    """
    def __init__(self, 
                 plate_model_path: str = PLATE_MODEL_PATH,
                 loc_model_path: str = LOC_MODEL_PATH,
                 recog_model_path: str = RECOG_MODEL_PATH):
        
        print("🧠 جاري تحميل النماذج الذكية الثلاثة...")
        self.plate_model = YOLO(plate_model_path)
        self.loc_model   = YOLO(loc_model_path)
        self.recog_model = YOLO(recog_model_path)
        print("✅ تم تحميل كافة النماذج بنجاح.")

    def recognize_plate_crop(self, crop: np.ndarray) -> dict:
        """
        معالجة اللوحة المقصوصة واستخراج الحروف والأرقام ورسم الصناديق التوضيحية
        """
        crop_rot, angle = deskew_adaptive(crop)
        h, w = crop_rot.shape[:2]
        if h < 10 or w < 20:
            return {
                "text": "", "digits": [], "letters": [], "char_details": [], 
                "annotated": crop_rot, "angle": angle,
                "syntax_valid": False, "governorate": "غير محدد", "format_code": "", "badge": ""
            }

        # 1. الدقة الفائقة بهندسة Lanczos4
        pass1 = cv2.resize(crop_rot, (int(w * SCALE_FACTOR), int(h * SCALE_FACTOR)), interpolation=cv2.INTER_LANCZOS4)
        w_tot = pass1.shape[1]
        pass2 = cv2.resize(crop_rot, (int(w * (SCALE_FACTOR * 1.2)), int(h * (SCALE_FACTOR * 1.2))), interpolation=cv2.INTER_LANCZOS4)
        scale2 = pass2.shape[1] / w_tot

        # 2. الموديل الثاني: كشف مواضع الخانات المادية
        l_res = self.loc_model(pass1, conf=LOC_CONF, verbose=False)[0]
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
            return {
                "text": "", "digits": [], "letters": [], "char_details": [], 
                "annotated": pass1, "angle": angle,
                "syntax_valid": False, "governorate": "غير محدد", "format_code": "", "badge": ""
            }

        # 3. الموديل الثالث: كشف متعدد المقاييس مع تطبيق الأوزان القبلية
        c1 = self.recog_model(pass1, conf=RECOG_CONF, verbose=False)[0]
        c2 = self.recog_model(pass2, conf=RECOG_CONF, verbose=False)[0]

        candidates = []
        for cb in c1.boxes:
            candidates.append((list(map(float, cb.xyxy[0])), self.recog_model.names[int(cb.cls[0])], float(cb.conf[0]) * 1.0))
        for cb in c2.boxes:
            coords = [float(cb.xyxy[0][0])/scale2, float(cb.xyxy[0][1])/scale2, float(cb.xyxy[0][2])/scale2, float(cb.xyxy[0][3])/scale2]
            candidates.append((coords, self.recog_model.names[int(cb.cls[0])], float(cb.conf[0]) * 0.95))

        # تنقية الصناديق الزائفة وتحديد الفاصل بين الأرقام والحروف وفق الهيكل المروري
        clean_b = prune_spurious_boxes(clean_b, candidates, pass1.shape[1], pass1.shape[0])
        split_idx = resolve_split_index(clean_b, candidates)

        # مطابقة المرشحين للخانات
        stage1_cands = []
        stage1_scores = []
        box_cands_map = []
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

        # 4. محرك التحقق الهندسي وتطابق التوائم
        annotated = pass1.copy()
        nums, lets = [], []
        char_details = []

        for i, (lx1, ly1, lx2, ly2) in enumerate(clean_b):
            is_digit = (i <= split_idx)
            cand = stage1_cands[i]
            score = stage1_scores[i]
            patch = pass1[ly1:ly2, lx1:lx2]
            box_cands = box_cands_map[i]

            if is_digit:
                verified = verify_digit_geometry(patch, cand, score, box_cands)
                ar_char = CHAR_MAP.get(verified, '?')
                nums.append(ar_char)
                label_text = verified
            else:
                verified = verify_letter_geometry(patch, cand, detected_letters, box_cands, all_letter_patches)
                ar_char = CHAR_MAP.get(verified, '?')
                lets.append(ar_char)
                label_text = verified

            color = (0, 220, 0) if is_digit else (255, 120, 0)
            cv2.rectangle(annotated, (lx1, ly1), (lx2, ly2), color, 2)
            cv2.putText(annotated, str(label_text or '?'), (lx1 + 2, max(24, ly1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            char_details.append({
                "box": [lx1, ly1, lx2, ly2],
                "class_name": label_text,
                "arabic": ar_char,
                "is_digit": is_digit
            })

        num_txt = " ".join(nums)
        rev_lets = list(reversed(lets))
        let_txt = " ".join(rev_lets)
        full_text = f"[{num_txt}] | [{let_txt}]"

        syntax_info = validate_egyptian_syntax(nums, rev_lets)

        return {
            "text": full_text,
            "digits": nums,
            "letters": rev_lets,
            "char_details": char_details,
            "annotated": annotated,
            "angle": angle,
            "syntax_valid": syntax_info["is_valid"],
            "governorate": syntax_info["governorate"],
            "format_code": syntax_info["format_code"],
            "badge": syntax_info["badge"]
        }

    def process_image(self, image: np.ndarray, add_hud: bool = True) -> dict:
        """
        المعالجة الكاملة للصورة (أو إطار الفيديو): كشف مكان اللوحة، قراءة المحارف، وإضافة الـ HUD الزووم
        """
        img_h, img_w = image.shape[:2]
        p_res = self.plate_model(image, conf=PLATE_CONF, verbose=False)[0]

        annotated_frame = image.copy()
        plates_output = []

        for box in p_res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            bw, bh = x2 - x1, y2 - y1
            if bw < 42 or bh < 14:
                continue
            crop = image[y1:y2, x1:x2]

            # معالجة تفاصيل اللوحة
            recog_res = self.recognize_plate_crop(crop)
            plate_text = recog_res["text"]

            # رسم مستطيل اللوحة على الصورة الأصلية
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            label = plate_text if plate_text else f"Plate: {conf:.2f}"
            cv2.rectangle(annotated_frame, (x1, max(0, y1 - 35)), (x1 + 280, y1), (0, 0, 0), -1)
            cv2.putText(annotated_frame, label, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

            # إضافة نافذة الزووم (HUD Inset) إذا كان ذلك مطلوباً
            if add_hud and recog_res.get("annotated") is not None:
                hud_w, hud_h = 320, 160
                zoomed = cv2.resize(recog_res["annotated"], (hud_w, hud_h - 45))
                hud_canvas = np.zeros((hud_h, hud_w, 3), dtype=np.uint8)
                hud_canvas[:hud_h - 45, :] = zoomed
                badge_lbl = recog_res.get("badge", "ZOOM (Model 2 Boxes):")
                cv2.putText(hud_canvas, badge_lbl, (8, hud_h - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 0), 1)
                cv2.putText(hud_canvas, plate_text, (8, hud_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
                cv2.rectangle(hud_canvas, (0, 0), (hud_w - 1, hud_h - 1), (0, 255, 0), 2)

                # تثبيت الزووم في الزاوية العلوية
                if img_w >= 360 and img_h >= 200:
                    annotated_frame[20:20 + hud_h, 20:20 + hud_w] = hud_canvas

            plates_output.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "text": plate_text,
                "digits": recog_res["digits"],
                "letters": recog_res["letters"],
                "angle": recog_res["angle"],
                "char_details": recog_res["char_details"],
                "syntax_valid": recog_res.get("syntax_valid", False),
                "governorate": recog_res.get("governorate", "غير محدد"),
                "format_code": recog_res.get("format_code", ""),
                "badge": recog_res.get("badge", "")
            })

        return {
            "success": len(plates_output) > 0,
            "plates_count": len(plates_output),
            "plates": plates_output,
            "annotated_image": annotated_frame
        }
