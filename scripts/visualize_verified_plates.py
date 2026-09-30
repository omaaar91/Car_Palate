import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from core.config import CHAR_MAP
from core.pipeline import EgyptianALPR

ARTIFACTS_DIR = r"C:\Users\Omar\.gemini\antigravity-ide\brain\80eb1de8-0e0c-401f-97c8-b20498617d02"
VERIFIED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "auto_labeled", "verified")

# استخدام خط Tahoma لدعمه الكامل لجميع الحروف العربية
font_path = "C:/Windows/Fonts/tahoma.ttf"
if not os.path.exists(font_path):
    font_path = "C:/Windows/Fonts/arial.ttf"

font_title = ImageFont.truetype(font_path, 21)
font_label = ImageFont.truetype(font_path, 16)
font_small = ImageFont.truetype(font_path, 13)

def render_arabic_text(text: str) -> str:
    try:
        return get_display(arabic_reshaper.reshape(text))
    except Exception:
        return text

def visualize_10_verified_plates():
    engine = EgyptianALPR()
    id_to_name = engine.recog_model.names
    
    files = sorted([f for f in os.listdir(VERIFIED_DIR) if f.endswith(".jpg")])[:10]
    print(f"🔍 جاري رسم وتوثيق {len(files)} لوحة من مجلد الـ Verified...")

    annotated_panels = []

    for idx, img_fname in enumerate(files):
        base_name = os.path.splitext(img_fname)[0]
        txt_fname = f"{base_name}.txt"
        
        img_path = os.path.join(VERIFIED_DIR, img_fname)
        txt_path = os.path.join(VERIFIED_DIR, txt_fname)

        img = cv2.imread(img_path)
        if img is None or not os.path.exists(txt_path):
            continue

        h, w = img.shape[:2]

        with open(txt_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        boxes_info = []
        for line in lines:
            parts = line.split()
            cid = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])

            x1 = int((cx - bw / 2.0) * w)
            y1 = int((cy - bh / 2.0) * h)
            x2 = int((cx + bw / 2.0) * w)
            y2 = int((cy + bh / 2.0) * h)

            c_name = id_to_name.get(cid, str(cid))
            ar_char = CHAR_MAP.get(c_name, c_name)
            is_digit = c_name in [str(i) for i in range(10)]

            boxes_info.append({
                "cid": cid,
                "name": c_name,
                "arabic": ar_char,
                "is_digit": is_digit,
                "box": (x1, y1, x2, y2)
            })

        # فرز الصناديق من اليسار لليمين مكانياً
        boxes_info = sorted(boxes_info, key=lambda b: b["box"][0])

        # تحويل لـ PIL للرسم عالي الدقة
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # تجهيز نص اللوحة: الأرقام من اليسار لليمين، والحروف من اليمين لليسار
        digits = [b["arabic"] for b in boxes_info if b["is_digit"]]
        raw_lets = [b["arabic"] for b in boxes_info if not b["is_digit"]]
        arabic_lets = list(reversed(raw_lets))
        full_plate_str = f"[{' '.join(digits)}] | [{' '.join(arabic_lets)}]"

        for b in boxes_info:
            x1, y1, x2, y2 = b["box"]
            color = (34, 197, 94) if b["is_digit"] else (56, 189, 248) # أخضر للأرقام، سماوي للحروف

            # رسم الصندوق
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # رسم شارة المحرف فوق الصندوق
            label_text = f"{b['arabic']} ({b['cid']})"
            reshaped_label = render_arabic_text(label_text)
            
            bbox_t = draw.textbbox((0, 0), reshaped_label, font=font_label)
            tw, th = bbox_t[2] - bbox_t[0], bbox_t[3] - bbox_t[1]

            tag_y1 = max(0, y1 - th - 7)
            draw.rectangle([x1, tag_y1, x1 + tw + 10, y1], fill=(15, 23, 42))
            draw.rectangle([x1, tag_y1, x1 + tw + 10, y1], outline=color, width=1)
            draw.text((x1 + 5, tag_y1 + 1), reshaped_label, font=font_label, fill=color)

        annotated_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # إنشاء كارت عرض لكل لوحة بحجم 660x250
        panel_w = 660
        panel_h = 250
        panel = np.full((panel_h, panel_w, 3), 15, dtype=np.uint8)

        # هيدر علوي أنيق
        cv2.rectangle(panel, (0, 0), (panel_w, 48), (30, 41, 59), -1)
        cv2.line(panel, (0, 48), (panel_w, 48), (56, 189, 248), 2)

        # توسيط وتكبير صورة اللوحة
        avail_h = panel_h - 58
        avail_w = panel_w - 20
        scale = min(avail_w / w, avail_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized_ann = cv2.resize(annotated_cv, (new_w, new_h))

        px = (panel_w - new_w) // 2
        py = 52 + (avail_h - new_h) // 2
        panel[py:py+new_h, px:px+new_w] = resized_ann

        # إضافة النصوص على الهيدر عبر PIL
        pil_panel = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
        draw_p = ImageDraw.Draw(pil_panel)

        header_str = f"#{idx+1} {base_name} : {full_plate_str}"
        draw_p.text((20, 10), render_arabic_text(header_str), font=font_title, fill=(255, 255, 255))
        draw_p.text((panel_w - 170, 14), f"YOLO: {len(boxes_info)} Labels", font=font_small, fill=(56, 189, 248))

        final_panel = cv2.cvtColor(np.array(pil_panel), cv2.COLOR_RGB2BGR)
        annotated_panels.append(final_panel)

    # دمج الـ 10 لوحات في شبكة أنيقة (5 صفوف × عمودين)
    row_count = (len(annotated_panels) + 1) // 2
    row_imgs = []
    for r in range(row_count):
        left_p = annotated_panels[r * 2]
        right_p = annotated_panels[r * 2 + 1] if (r * 2 + 1) < len(annotated_panels) else np.zeros_like(left_p)
        row = np.hstack([left_p, right_p])
        row_imgs.append(row)

    full_montage = np.vstack(row_imgs)

    out_montage_path = os.path.join(ARTIFACTS_DIR, "showcase_10_verified_plates.jpg")
    cv2.imwrite(out_montage_path, full_montage, [cv2.IMWRITE_JPEG_QUALITY, 93])
    print(f"🎉 تم حفظ لوحة العرض المجمعة بنجاح في:\n   {out_montage_path}")

if __name__ == "__main__":
    visualize_10_verified_plates()
