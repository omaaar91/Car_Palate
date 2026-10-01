import os
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display

from core.config import (
    PLATE_MODEL_PATH, LOC_MODEL_PATH, RECOG_MODEL_PATH,
    CHAR_MAP, DIGITS_SET, CLASS_WEIGHTS,
    PLATE_CONF, LOC_CONF, RECOG_CONF, SCALE_FACTOR
)
from core.deskew import deskew_adaptive
from core.verification import (
    validate_egyptian_syntax, 
    suppress_duplicate_slots,
    suppress_duplicate_detections,
    prune_spurious_boxes, 
    resolve_split_index, 
    verify_digit_geometry, 
    verify_letter_geometry
)

# مسار الخط العربي للنظام لدعم الرسم العربي فائق الجودة
FONT_PATH = "C:/Windows/Fonts/tahoma.ttf"
if not os.path.exists(FONT_PATH):
    FONT_PATH = "C:/Windows/Fonts/arial.ttf"

def render_arabic_text(text: str) -> str:
    """إعادة تشكيل النص العربي وعكس اتجاهه ليظهر بشكل صحيح مع مكتبات الرسم"""
    try:
        return get_display(arabic_reshaper.reshape(text))
    except Exception:
        return text

class EgyptianALPR:
    """
    نظام التعرف الذكي متعدد المراحل على لوحات المركبات المصرية
    (Multi-Stage Egyptian License Plate Recognition Engine)
    """
    def __init__(self, 
                 plate_model_path: str = PLATE_MODEL_PATH,
                 loc_model_path: str = LOC_MODEL_PATH,
                 recog_model_path: str = RECOG_MODEL_PATH):
        
        print(f"🧠 جاري تحميل النماذج الذكية الثلاثة (مصنف الرموز: {os.path.basename(recog_model_path)})...")
        self.plate_model = YOLO(plate_model_path)
        self.loc_model   = YOLO(loc_model_path)
        self.recog_model = YOLO(recog_model_path)
        print("✅ تم تحميل كافة النماذج بنجاح.")

    def recognize_plate_crop(self, crop: np.ndarray, direct_mode: bool = True) -> dict:
        """
        معالجة اللوحة المقصوصة واستخراج الحروف والأرقام ورسم الصناديق التوضيحية
        direct_mode: استخدام الموديل 3 مباشرة دون الاعتماد على الموديل 2 لمنع الأخطاء والصناديق الوهمية
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

        if direct_mode:
            # 2. الكشف المباشر فائق الدقة باستخدام الموديل 3 مباشرة (End-to-End Direct Detection)
            c_res = self.recog_model(pass1, conf=0.25, verbose=False)[0]
            raw_dets = []
            for cb in c_res.boxes:
                cname = self.recog_model.names[int(cb.cls[0])]
                conf = float(cb.conf[0])
                coords = list(map(int, cb.xyxy[0][:4]))
                if coords[0] < (w_tot * 0.03) or coords[2] > (w_tot * 0.985):
                    continue
                raw_dets.append({
                    "name": cname,
                    "conf": conf,
                    "box": coords,
                    "is_digit": cname in DIGITS_SET
                })

            filtered_dets = suppress_duplicate_detections(raw_dets, iou_thresh=0.30)
            if len(filtered_dets) > 7:
                filtered_dets = sorted(sorted(filtered_dets, key=lambda d: d["conf"], reverse=True)[:7], key=lambda d: d["box"][0])

            annotated = pass1.copy()
            nums, lets = [], []
            char_details = []
            box_labels = []

            for d in filtered_dets:
                lx1, ly1, lx2, ly2 = d["box"]
                cname = d["name"]
                score = d["conf"]
                patch = pass1[ly1:ly2, lx1:lx2]
                is_digit = d["is_digit"]

                if is_digit:
                    verified = verify_digit_geometry(patch, cname, score, [(score, cname)])
                    ar_char = CHAR_MAP.get(verified, '?')
                    nums.append(ar_char)
                    label_text = verified
                else:
                    verified = verify_letter_geometry(patch, cname, [x["name"] for x in filtered_dets if not x["is_digit"]], [(score, cname)], [])
                    ar_char = CHAR_MAP.get(verified, '?')
                    lets.append(ar_char)
                    label_text = verified

                color = (0, 220, 0) if is_digit else (255, 120, 0)
                cv2.rectangle(annotated, (lx1, ly1), (lx2, ly2), color, 2)
                box_labels.append(((lx1, ly1, lx2, ly2), ar_char, is_digit, label_text))

                char_details.append({
                    "box": [lx1, ly1, lx2, ly2],
                    "class_name": label_text,
                    "arabic": ar_char,
                    "is_digit": is_digit
                })

            if box_labels:
                try:
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(annotated_rgb)
                    draw = ImageDraw.Draw(pil_img)
                    font_char = ImageFont.truetype(FONT_PATH, 20)
                    for (lx1, ly1, lx2, ly2), ar_char, is_digit, _ in box_labels:
                        color = (0, 255, 100) if is_digit else (255, 160, 0)
                        reshaped = render_arabic_text(ar_char)
                        draw.rectangle([lx1, max(0, ly1 - 26), lx1 + 26, ly1], fill=(0, 0, 0, 220))
                        draw.text((lx1 + 4, max(0, ly1 - 26)), reshaped, font=font_char, fill=color)
                    annotated = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                except Exception:
                    for (lx1, ly1, lx2, ly2), _, is_digit, label_text in box_labels:
                        color = (0, 220, 0) if is_digit else (255, 120, 0)
                        cv2.putText(annotated, str(label_text or '?'), (lx1 + 2, max(24, ly1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            num_txt = " ".join(nums) if nums else "-"
            rev_lets = list(reversed(lets))
            let_txt = " ".join(rev_lets) if rev_lets else "-"
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

        # في حال تم تمرير direct_mode = False: استخدام خط المعالجة القديم مع الموديل الثاني
        pass2 = cv2.resize(crop_rot, (int(w * (SCALE_FACTOR * 1.2)), int(h * (SCALE_FACTOR * 1.2))), interpolation=cv2.INTER_LANCZOS4)
        scale2 = pass2.shape[1] / w_tot

        # 2. الموديل الثاني: كشف مواضع الخانات المادية مع إلغاء التكرار
        l_res = self.loc_model(pass1, conf=LOC_CONF, verbose=False)[0]
        raw_b = []
        for bx in l_res.boxes:
            coords = list(map(int, bx.xyxy[0][:4]))
            coords.append(float(bx.conf[0]))
            raw_b.append(coords)

        # دمج وإلغاء الصناديق المتداخلة لنفس الخانة
        suppressed_b = suppress_duplicate_slots(raw_b, iou_thresh=0.35)

        clean_b = []
        for bx in suppressed_b:
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

        # تنقية الصناديق الزائفة كالمسامير والانعكاسات
        clean_b = prune_spurious_boxes(clean_b, candidates, pass1.shape[1], pass1.shape[0])

        # مطابقة المرشحين للخانات بمرونة عالية وحرية كاملة
        stage1_cands = []
        stage1_scores = []
        box_cands_map = []
        for i, (lx1, ly1, lx2, ly2) in enumerate(clean_b):
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
                    cands_for_box.append((w_c, cname))
                    if w_c > best_score:
                        best_score = w_c
                        best_cand = cname
            stage1_cands.append(best_cand)
            stage1_scores.append(best_score)
            box_cands_map.append(cands_for_box)

        detected_letters = [c for c in stage1_cands if c not in DIGITS_SET and c is not None]
        all_letter_patches = [(stage1_cands[i], pass1[ly1:ly2, lx1:lx2]) for i, (lx1, ly1, lx2, ly2) in enumerate(clean_b) if stage1_cands[i] not in DIGITS_SET and stage1_cands[i] is not None]

        # 4. محرك التحقق الهندسي وتطابق التوائم
        annotated = pass1.copy()
        nums, lets = [], []
        char_details = []
        box_labels = []

        for i, (lx1, ly1, lx2, ly2) in enumerate(clean_b):
            cand = stage1_cands[i]
            score = stage1_scores[i]
            patch = pass1[ly1:ly2, lx1:lx2]
            box_cands = box_cands_map[i]

            # استبعاد الصناديق الزائفة مثل المسامير أو الشعار التي ليس لها محرف متطابق
            if cand is None or score < 0.08:
                continue

            is_digit = (cand in DIGITS_SET)
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
            box_labels.append(((lx1, ly1, lx2, ly2), ar_char, is_digit, label_text))

            char_details.append({
                "box": [lx1, ly1, lx2, ly2],
                "class_name": label_text,
                "arabic": ar_char,
                "is_digit": is_digit
            })

        # رسم الحروف والأرقام العربية بوضوح فائق باستخدام PIL
        if box_labels:
            try:
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(annotated_rgb)
                draw = ImageDraw.Draw(pil_img)
                font_char = ImageFont.truetype(FONT_PATH, 20)
                for (lx1, ly1, lx2, ly2), ar_char, is_digit, _ in box_labels:
                    color = (0, 255, 100) if is_digit else (255, 160, 0)
                    reshaped = render_arabic_text(ar_char)
                    draw.rectangle([lx1, max(0, ly1 - 26), lx1 + 26, ly1], fill=(0, 0, 0, 220))
                    draw.text((lx1 + 4, max(0, ly1 - 26)), reshaped, font=font_char, fill=color)
                annotated = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception:
                for (lx1, ly1, lx2, ly2), _, is_digit, label_text in box_labels:
                    color = (0, 220, 0) if is_digit else (255, 120, 0)
                    cv2.putText(annotated, str(label_text or '?'), (lx1 + 2, max(24, ly1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        num_txt = " ".join(nums) if nums else "-"
        rev_lets = list(reversed(lets))
        let_txt = " ".join(rev_lets) if rev_lets else "-"
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
            # تطبيق تمديد تكيفي للهوامش لمنع اقتصاص الحروف الطرفية
            pad_x = int(bw * 0.04)
            pad_y = int(bh * 0.05)
            crop = image[max(0, y1 - pad_y):min(img_h, y2 + pad_y), max(0, x1 - pad_x):min(img_w, x2 + pad_x)]

            # معالجة تفاصيل اللوحة
            recog_res = self.recognize_plate_crop(crop)
            plate_text = recog_res["text"]

            # رسم مستطيل اللوحة على الصورة الأصلية
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            label = plate_text if plate_text else f"Plate: {conf:.2f}"
            cv2.rectangle(annotated_frame, (x1, max(0, y1 - 35)), (x1 + 320, y1), (0, 0, 0), -1)

            # رسم النص العربي على الشريط العلوي
            try:
                ann_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                pil_ann = Image.fromarray(ann_rgb)
                draw_ann = ImageDraw.Draw(pil_ann)
                font_top = ImageFont.truetype(FONT_PATH, 20)
                reshaped_lbl = render_arabic_text(label)
                draw_ann.text((x1 + 8, max(0, y1 - 30)), reshaped_lbl, font=font_top, fill=(0, 255, 255))
                annotated_frame = cv2.cvtColor(np.array(pil_ann), cv2.COLOR_RGB2BGR)
            except Exception:
                cv2.putText(annotated_frame, label, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

            # إضافة نافذة الزووم (HUD Inset) إذا كان ذلك مطلوباً
            if add_hud and recog_res.get("annotated") is not None:
                hud_w, hud_h = 320, 160
                zoomed = cv2.resize(recog_res["annotated"], (hud_w, hud_h - 45))
                hud_canvas = np.zeros((hud_h, hud_w, 3), dtype=np.uint8)
                hud_canvas[:hud_h - 45, :] = zoomed
                badge_lbl = recog_res.get("badge", "ZOOM (Model 2 Boxes):")

                # رسم نصوص الـ HUD باللغة العربية
                try:
                    hud_rgb = cv2.cvtColor(hud_canvas, cv2.COLOR_BGR2RGB)
                    pil_hud = Image.fromarray(hud_rgb)
                    draw_hud = ImageDraw.Draw(pil_hud)
                    font_hud_sm = ImageFont.truetype(FONT_PATH, 14)
                    font_hud_lg = ImageFont.truetype(FONT_PATH, 16)
                    draw_hud.text((8, hud_h - 40), render_arabic_text(badge_lbl), font=font_hud_sm, fill=(0, 255, 0))
                    draw_hud.text((8, hud_h - 22), render_arabic_text(plate_text), font=font_hud_lg, fill=(0, 255, 255))
                    hud_canvas = cv2.cvtColor(np.array(pil_hud), cv2.COLOR_RGB2BGR)
                except Exception:
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

