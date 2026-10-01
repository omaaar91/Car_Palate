import os
import time
import base64
import csv
import io
import cv2
import numpy as np
from pydantic import BaseModel
from typing import List, Optional
from collections import deque, Counter

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles

from core.pipeline import EgyptianALPR
from backend.schemas import ALPRResponse, PlateResult
from backend.database import get_alpr_database, CROPS_DIR

app = FastAPI(
    title="Egyptian License Plate Recognition API",
    description="واجهة برمجية فائقة الدقة للتعرف على لوحات السيارات المصرية باستخدام الذكاء الاصطناعي متعدد المراحل",
    version="2.0.0"
)

# تمكين CORS لدعم الاتصال من مختلف المتصفحات والتطبيقات
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
STATIC_DIR = os.path.join(FRONTEND_DIR, "static")

os.makedirs(STATIC_DIR, exist_ok=True)
AUTO_LABELED_DIR = os.path.join(os.path.dirname(CROPS_DIR), "auto_labeled")
os.makedirs(AUTO_LABELED_DIR, exist_ok=True)

# ربط الملفات الثابتة والصور المقصوصة
app.mount("/crops", StaticFiles(directory=CROPS_DIR), name="crops")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/auto_labeled", StaticFiles(directory=AUTO_LABELED_DIR), name="auto_labeled")

# تهيئة المحرك الذكي
alpr_engine = None

def get_alpr_engine() -> EgyptianALPR:
    global alpr_engine
    if alpr_engine is None:
        alpr_engine = EgyptianALPR()
    return alpr_engine

@app.on_event("startup")
def startup_event():
    get_alpr_engine()
    get_alpr_database()

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Egyptian ALPR API",
        "version": "2.0.0",
        "docs": "/docs",
        "scanner": "/scanner",
        "dashboard": "/dashboard"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": alpr_engine is not None
    }

def decode_image_file(file_bytes: bytes) -> np.ndarray:
    """تحويل بايتات الملف إلى مصفوفة صور OpenCV"""
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("تعذر فك ترميز ملف الصورة.")
    return img

def encode_image_base64(img: np.ndarray, quality: int = 85) -> str:
    """ترميز الصورة إلى Base64 لنقلها عبر الـ API"""
    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer).decode("utf-8")

@app.post("/predict/image", response_model=ALPRResponse)
async def predict_image(
    file: UploadFile = File(...),
    include_annotated: bool = Query(True, description="تضمين الصورة المعالجة بصيغة Base64"),
    add_hud: bool = Query(True, description="إضافة نافذة الزووم المصغرة للوحة")
):
    """
    استقبال صورة سيارة واستخراج أرقام وحروف اللوحة مع التحقق الهندسي وتطابق التوائم
    """
    try:
        content = await file.read()
        image = decode_image_file(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"خطأ في قراءة ملف الصورة: {str(e)}")

    engine = get_alpr_engine()
    result = engine.process_image(image, add_hud=add_hud)

    b64_img = None
    if include_annotated and result.get("annotated_image") is not None:
        b64_img = encode_image_base64(result["annotated_image"])

    plates_data = []
    for p in result["plates"]:
        plates_data.append(PlateResult(
            text=p["text"],
            digits=p["digits"],
            letters=p["letters"],
            confidence=p["confidence"],
            bbox=p["bbox"],
            angle=p["angle"],
            track_id=p.get("track_id"),
            char_details=p.get("char_details"),
            syntax_valid=p.get("syntax_valid"),
            governorate=p.get("governorate"),
            format_code=p.get("format_code"),
            badge=p.get("badge")
        ))

    return ALPRResponse(
        success=result["success"],
        plates_count=result["plates_count"],
        plates=plates_data,
        annotated_image_base64=b64_img,
        message="تم كشف اللوحات بنجاح" if result["success"] else "لم يتم العثور على أي لوحة في الصورة"
    )

@app.post("/predict/frame")
async def predict_frame(
    file: UploadFile = File(...),
    return_image_bytes: bool = Query(False, description="إرجاع الصورة المعالجة كبايتات مباشرة")
):
    """
    نقطة نهاية فائقة السرعة مخصصة لإطارات الفيديو والكاميرا الحية (Live Video Streaming)
    """
    try:
        content = await file.read()
        image = decode_image_file(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    engine = get_alpr_engine()
    result = engine.process_image(image, add_hud=True)

    if return_image_bytes:
        _, buffer = cv2.imencode(".jpg", result["annotated_image"], [cv2.IMWRITE_JPEG_QUALITY, 80])
        return Response(content=buffer.tobytes(), media_type="image/jpeg")

    # إرجاع رد خفيف وسريع للفيديو
    plates_simple = [{"text": p["text"], "bbox": p["bbox"]} for p in result["plates"]]
    return {
        "success": result["success"],
        "count": result["plates_count"],
        "plates": plates_simple,
        "annotated_base64": encode_image_base64(result["annotated_image"], quality=75)
    }

# =============================================================
# مسارات الواجهة الأمامية و PWA
# =============================================================
@app.get("/scanner")
def get_scanner_page():
    scanner_path = os.path.join(FRONTEND_DIR, "scanner.html")
    if os.path.exists(scanner_path):
        return FileResponse(
            scanner_path,
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    return Response(content="<h1>Scanner page not found</h1>", media_type="text/html", status_code=404)

@app.get("/dashboard")
def get_dashboard_page():
    dashboard_path = os.path.join(FRONTEND_DIR, "dashboard.html")
    if os.path.exists(dashboard_path):
        return FileResponse(dashboard_path)
    return Response(content="<h1>Dashboard page not found</h1>", media_type="text/html", status_code=404)

@app.get("/manifest.json")
def get_manifest():
    manifest_path = os.path.join(FRONTEND_DIR, "manifest.json")
    if os.path.exists(manifest_path):
        return FileResponse(manifest_path, media_type="application/manifest+json")
    return Response(content="{}", media_type="application/json", status_code=404)

@app.get("/sw.js")
def get_service_worker():
    sw_path = os.path.join(FRONTEND_DIR, "sw.js")
    if os.path.exists(sw_path):
        return FileResponse(sw_path, media_type="application/javascript", headers={"Service-Worker-Allowed": "/"})
    return Response(content="", media_type="application/javascript", status_code=404)

# =============================================================
# مسار أداة المراجعة السريعة واعتماد الداتا سيت
# =============================================================
@app.get("/review")
def get_review_page():
    review_path = os.path.join(FRONTEND_DIR, "review.html")
    if os.path.exists(review_path):
        return FileResponse(review_path)
    return Response(content="<h1>Review page not found</h1>", media_type="text/html", status_code=404)

@app.get("/api/review/items")
def get_review_items():
    meta_path = os.path.join(AUTO_LABELED_DIR, "review_data.json")
    if os.path.exists(meta_path):
        import json
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

class ReviewConfirmRequest(BaseModel):
    filename: str
    action: str = "confirm"

@app.post("/api/review/confirm")
def review_confirm(req: ReviewConfirmRequest):
    import json
    base_name = os.path.splitext(req.filename)[0]
    cand_img = os.path.join(AUTO_LABELED_DIR, "candidates", f"{base_name}.jpg")
    cand_txt = os.path.join(AUTO_LABELED_DIR, "candidates", f"{base_name}.txt")
    ver_dir = os.path.join(AUTO_LABELED_DIR, "verified")

    # تحديث حالة المراجعة والدقة في ملف البيانات
    meta_path = os.path.join(AUTO_LABELED_DIR, "review_data.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                items = json.load(f)
            for it in items:
                if it["filename"] == req.filename:
                    it["reviewed"] = True
                    if req.action == "confirm":
                        it["accuracy_status"] = "modified" if it.get("is_modified") else "correct"
                        it["folder"] = "verified"
                    else:
                        it["accuracy_status"] = "discarded"
                    break
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print("Error updating review_data.json on confirm:", e)

    if req.action == "confirm" and os.path.exists(cand_img):
        import shutil
        shutil.move(cand_img, os.path.join(ver_dir, f"{base_name}.jpg"))
        if os.path.exists(cand_txt):
            shutil.move(cand_txt, os.path.join(ver_dir, f"{base_name}.txt"))
    elif req.action == "discard":
        for sub in ["candidates", "verified"]:
            f_img = os.path.join(AUTO_LABELED_DIR, sub, f"{base_name}.jpg")
            f_txt = os.path.join(AUTO_LABELED_DIR, sub, f"{base_name}.txt")
            if os.path.exists(f_img): os.remove(f_img)
            if os.path.exists(f_txt): os.remove(f_txt)

    return {"success": True}

@app.get("/api/review/stats")
def get_review_stats():
    import json
    meta_path = os.path.join(AUTO_LABELED_DIR, "review_data.json")
    if not os.path.exists(meta_path):
        return {
            "total": 0, "reviewed": 0, "pending": 0,
            "exact_matches": 0, "modified": 0, "discarded": 0,
            "accuracy_pct": 0.0
        }
    with open(meta_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    reviewed = [it for it in items if it.get("reviewed")]
    exact = [it for it in reviewed if it.get("accuracy_status") == "correct"]
    modified = [it for it in reviewed if it.get("accuracy_status") == "modified"]
    discarded = [it for it in reviewed if it.get("accuracy_status") == "discarded"]

    valid_reviewed = len(exact) + len(modified)
    acc = round((len(exact) / valid_reviewed) * 100, 1) if valid_reviewed > 0 else 0.0

    return {
        "total": len(items),
        "reviewed": len(reviewed),
        "pending": len(items) - len(reviewed),
        "exact_matches": len(exact),
        "modified": len(modified),
        "discarded": len(discarded),
        "accuracy_pct": acc
    }

class UpdateCharRequest(BaseModel):
    filename: str
    char_index: int
    new_char: str

@app.post("/api/review/update_char")
def review_update_char(req: UpdateCharRequest):
    import json
    base_name = os.path.splitext(req.filename)[0]
    meta_path = os.path.join(AUTO_LABELED_DIR, "review_data.json")
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="Review data not found")

    with open(meta_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    target_item = None
    for item in items:
        if item["filename"] == req.filename:
            target_item = item
            break

    if not target_item or req.char_index >= len(target_item["chars"]):
        raise HTTPException(status_code=404, detail="Item or char index not found")

    engine = get_alpr_engine()
    name_to_id = {v: k for k, v in engine.recog_model.names.items()}
    
    from core.config import CHAR_MAP
    ar_to_name = {v: k for k, v in CHAR_MAP.items()}
    digit_fix = {'٠':'0', '١':'1', '٢':'2', '٣':'3', '٤':'4', '٥':'5', '٦':'6', '٧':'7', '٨':'8', '٩':'9', 'ه':'haa'}
    
    raw = req.new_char.strip()
    if raw in digit_fix:
        raw = digit_fix[raw]

    model_name = ar_to_name.get(raw, raw)
    if model_name not in name_to_id:
        for k, v in name_to_id.items():
            if k.lower() == model_name.lower():
                model_name = k
                break

    class_id = name_to_id.get(model_name, 0)
    arabic_display = CHAR_MAP.get(model_name, raw)

    target_item["chars"][req.char_index]["name"] = model_name
    target_item["chars"][req.char_index]["arabic"] = arabic_display
    target_item["chars"][req.char_index]["is_digit"] = model_name in [str(i) for i in range(10)]

    nums = [c["arabic"] for c in target_item["chars"] if c["is_digit"]]
    # الحروف مقصوصة من اليسار لليمين مكانياً، وقراءتها بالعربية تبدأ من اليمين لليسار
    raw_lets = [c["arabic"] for c in target_item["chars"] if not c["is_digit"]]
    arabic_lets = list(reversed(raw_lets))

    from core.verification import validate_egyptian_syntax
    syntax_info = validate_egyptian_syntax(nums, arabic_lets)
    target_item["plate_text"] = f"[{' '.join(nums)}] | [{' '.join(arabic_lets)}]"
    target_item["syntax_valid"] = syntax_info["is_valid"]
    target_item["governorate"] = syntax_info["governorate"]
    target_item["badge"] = syntax_info["badge"]
    target_item["is_modified"] = True
    target_item["accuracy_status"] = "modified"

    # إعادة كتابة ملف الـ YOLO .txt على القرص فوراً
    target_folder = target_item.get("folder", "verified")
    txt_path = os.path.join(AUTO_LABELED_DIR, target_folder, f"{base_name}.txt")
    img_path = os.path.join(AUTO_LABELED_DIR, target_folder, f"{base_name}.jpg")

    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        if img is not None:
            h, w = img.shape[:2]
            yolo_lines = []
            for c in target_item["chars"]:
                cid = name_to_id.get(c["name"], 0)
                lx1, ly1, lx2, ly2 = c["box"]
                cx = ((lx1 + lx2) / 2.0) / w
                cy = ((ly1 + ly2) / 2.0) / h
                bw = (lx2 - lx1) / w
                bh = (ly2 - ly1) / h
                yolo_lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            with open(txt_path, "w", encoding="utf-8") as f_txt:
                f_txt.write("\n".join(yolo_lines) + "\n")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    return {
        "success": True, 
        "updated_item": target_item,
        "new_class_id": class_id,
        "saved_txt": txt_path
    }

class DeleteCharRequest(BaseModel):
    filename: str
    char_index: int

@app.post("/api/review/delete_char")
def review_delete_char(req: DeleteCharRequest):
    import json
    base_name = os.path.splitext(req.filename)[0]
    meta_path = os.path.join(AUTO_LABELED_DIR, "review_data.json")
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="Review data not found")

    with open(meta_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    target_item = None
    for item in items:
        if item["filename"] == req.filename:
            target_item = item
            break

    if not target_item or req.char_index >= len(target_item["chars"]):
        raise HTTPException(status_code=404, detail="Item or char index not found")

    # حذف الخانة المحددة نهائياً
    del target_item["chars"][req.char_index]
    target_item["is_modified"] = True
    target_item["accuracy_status"] = "modified"

    nums = [c["arabic"] for c in target_item["chars"] if c["is_digit"]]
    raw_lets = [c["arabic"] for c in target_item["chars"] if not c["is_digit"]]
    arabic_lets = list(reversed(raw_lets))

    from core.verification import validate_egyptian_syntax
    syntax_info = validate_egyptian_syntax(nums, arabic_lets)
    target_item["plate_text"] = f"[{' '.join(nums)}] | [{' '.join(arabic_lets)}]"
    target_item["syntax_valid"] = syntax_info["is_valid"]
    target_item["governorate"] = syntax_info["governorate"]
    target_item["badge"] = syntax_info["badge"]

    # إعادة كتابة ملف الـ YOLO .txt على القرص فوراً
    target_folder = target_item.get("folder", "candidates")
    txt_path = os.path.join(AUTO_LABELED_DIR, target_folder, f"{base_name}.txt")
    img_path = os.path.join(AUTO_LABELED_DIR, target_folder, f"{base_name}.jpg")

    engine = get_alpr_engine()
    name_to_id = {v: k for k, v in engine.recog_model.names.items()}

    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        if img is not None:
            h, w = img.shape[:2]
            yolo_lines = []
            for c in target_item["chars"]:
                cid = name_to_id.get(c["name"], 0)
                lx1, ly1, lx2, ly2 = c["box"]
                cx = ((lx1 + lx2) / 2.0) / w
                cy = ((ly1 + ly2) / 2.0) / h
                bw = (lx2 - lx1) / w
                bh = (ly2 - ly1) / h
                yolo_lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            with open(txt_path, "w", encoding="utf-8") as f_txt:
                f_txt.write("\n".join(yolo_lines) + "\n")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    return {
        "success": True, 
        "updated_item": target_item,
        "saved_txt": txt_path
    }

class AddCharRequest(BaseModel):
    filename: str
    char_name: str
    is_digit: bool = False

@app.post("/api/review/add_char")
def review_add_char(req: AddCharRequest):
    import json
    base_name = os.path.splitext(req.filename)[0]
    meta_path = os.path.join(AUTO_LABELED_DIR, "review_data.json")
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="Review data not found")

    with open(meta_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    target_item = None
    for item in items:
        if item["filename"] == req.filename:
            target_item = item
            break

    if not target_item:
        raise HTTPException(status_code=404, detail="Item not found")

    engine = get_alpr_engine()
    name_to_id = {v: k for k, v in engine.recog_model.names.items()}
    from core.config import CHAR_MAP
    ar_to_name = {v: k for k, v in CHAR_MAP.items()}
    digit_fix = {'٠':'0', '١':'1', '٢':'2', '٣':'3', '٤':'4', '٥':'5', '٦':'6', '٧':'7', '٨':'8', '٩':'9', 'ه':'haa'}
    
    raw = req.char_name.strip()
    if raw in digit_fix:
        raw = digit_fix[raw]
    model_name = ar_to_name.get(raw, raw)
    arabic_display = CHAR_MAP.get(model_name, raw)
    is_digit = model_name in [str(i) for i in range(10)]

    chars = target_item["chars"]
    if chars:
        avg_w = float(np.mean([c["box"][2] - c["box"][0] for c in chars]))
        avg_h = float(np.mean([c["box"][3] - c["box"][1] for c in chars]))
        avg_y1 = float(np.mean([c["box"][1] for c in chars]))
        avg_y2 = float(np.mean([c["box"][3] for c in chars]))
        if is_digit:
            min_x = min(c["box"][0] for c in chars)
            new_box = [max(0, int(min_x - avg_w)), int(avg_y1), int(min_x), int(avg_y2)]
        else:
            max_x = max(c["box"][2] for c in chars)
            new_box = [int(max_x), int(avg_y1), int(max_x + avg_w), int(avg_y2)]
    else:
        new_box = [50, 50, 100, 150]

    chars.append({
        "name": model_name,
        "arabic": arabic_display,
        "is_digit": is_digit,
        "box": new_box
    })
    chars.sort(key=lambda c: c["box"][0])

    target_item["is_modified"] = True
    target_item["accuracy_status"] = "modified"

    nums = [c["arabic"] for c in target_item["chars"] if c["is_digit"]]
    raw_lets = [c["arabic"] for c in target_item["chars"] if not c["is_digit"]]
    arabic_lets = list(reversed(raw_lets))

    from core.verification import validate_egyptian_syntax
    syntax_info = validate_egyptian_syntax(nums, arabic_lets)
    target_item["plate_text"] = f"[{' '.join(nums)}] | [{' '.join(arabic_lets)}]"
    target_item["syntax_valid"] = syntax_info["is_valid"]
    target_item["governorate"] = syntax_info["governorate"]
    target_item["badge"] = syntax_info["badge"]

    # كتابة ملف YOLO .txt
    target_folder = target_item.get("folder", "candidates")
    txt_path = os.path.join(AUTO_LABELED_DIR, target_folder, f"{base_name}.txt")
    img_path = os.path.join(AUTO_LABELED_DIR, target_folder, f"{base_name}.jpg")

    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        if img is not None:
            h, w = img.shape[:2]
            yolo_lines = []
            for c in target_item["chars"]:
                cid = name_to_id.get(c["name"], 0)
                lx1, ly1, lx2, ly2 = c["box"]
                cx = ((lx1 + lx2) / 2.0) / w
                cy = ((ly1 + ly2) / 2.0) / h
                bw = (lx2 - lx1) / w
                bh = (ly2 - ly1) / h
                yolo_lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            with open(txt_path, "w", encoding="utf-8") as f_txt:
                f_txt.write("\n".join(yolo_lines) + "\n")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    return {
        "success": True,
        "updated_item": target_item,
        "saved_txt": txt_path
    }

@app.get("/api/review/export_zip")
def export_dataset_zip():
    import zipfile
    ver_dir = os.path.join(AUTO_LABELED_DIR, "verified")
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as z:
        engine = get_alpr_engine()
        names_list = [engine.recog_model.names[i] for i in sorted(engine.recog_model.names.keys())]
        yaml_content = f"""path: ./dataset
train: images/train
val: images/val
nc: {len(names_list)}
names: {names_list}
"""
        z.writestr("dataset/data.yaml", yaml_content)

        files = os.listdir(ver_dir) if os.path.exists(ver_dir) else []
        jpg_files = sorted([f for f in files if f.endswith(".jpg")])
        
        # تقسيم بنسبة 85% للتدريب و 15% للتحقق لضمان حساب دقيق للـ mAP في Colab
        split_idx = int(len(jpg_files) * 0.85)
        train_set = set(jpg_files[:split_idx])

        for f in jpg_files:
            base = os.path.splitext(f)[0]
            split = "train" if f in train_set else "val"
            img_path = os.path.join(ver_dir, f)
            txt_path = os.path.join(ver_dir, f"{base}.txt")
            if os.path.exists(img_path):
                z.write(img_path, f"dataset/images/{split}/{f}")
            if os.path.exists(txt_path):
                z.write(txt_path, f"dataset/labels/{split}/{base}.txt")

    zip_buffer.seek(0)
    return Response(
        content=zip_buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=egyptian_chars_verified_dataset_{int(time.time())}.zip"}
    )

# =============================================================
# مسارات إدارة قاعدة البيانات ولوحة التحكم
# =============================================================
class BlacklistAddRequest(BaseModel):
    digits: list
    letters: list
    reason: str
    severity: str = "danger"

@app.get("/api/stats")
def get_stats():
    db = get_alpr_database()
    return db.get_stats_summary()

@app.get("/api/logs")
def get_logs(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    q: str = Query("", description="البحث باللوحة أو المحافظة"),
    date: str = Query("", description="فلترة التاريخ YYYY-MM-DD")
):
    db = get_alpr_database()
    logs = db.get_logs(limit=limit, offset=offset, query=q, date_filter=date)
    return {"logs": logs}

@app.get("/api/blacklist")
def get_blacklist():
    db = get_alpr_database()
    return {"items": db.get_blacklist_items()}

@app.post("/api/blacklist")
def add_blacklist(item: BlacklistAddRequest):
    db = get_alpr_database()
    success = db.add_to_blacklist(
        digits=item.digits,
        letters=item.letters,
        reason=item.reason,
        severity=item.severity
    )
    return {"success": success}

@app.delete("/api/blacklist/{item_id}")
def delete_blacklist(item_id: int):
    db = get_alpr_database()
    success = db.remove_from_blacklist(item_id)
    return {"success": success}

@app.get("/api/export/csv")
def export_csv():
    db = get_alpr_database()
    logs = db.get_logs(limit=3000)
    output = io.StringIO()
    # UTF-8 BOM so Microsoft Excel renders Arabic text properly
    output.write('\ufeff')
    writer = csv.writer(output)
    writer.writerow(["المعرف", "الوقت والتاريخ", "رقم اللوحة", "المحافظة", "نسبة التأكد", "مطلوبة أمنياً", "سبب الإدراج الأمني"])
    for r in logs:
        writer.writerow([
            r["id"],
            r["timestamp"],
            r["plate_text"],
            r["governorate"],
            f"{round(r['confidence'] * 100)}%",
            "نعم 🚨" if r["is_blacklist"] else "لا",
            r["blacklist_reason"] or "-"
        ])
    output.seek(0)
    return Response(
        content=output.getvalue().encode('utf-8-sig'),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename=vehicle_logs_{int(time.time())}.csv"}
    )

# =============================================================
# نظام التثبيت الزمني للوحات عبر الفريمات المتتابعة
# =============================================================
class TemporalPlateStabilizer:
    """
    نظام التصويت والتثبيت الزمني التراكمي (Multi-Frame Temporal Voting)
    يجمع قراءات الفريمات المتتالية لنفس اللوحة لتثبيت الحروف ومنع الرعشة والالتباس العشوائي
    """
    def __init__(self, history_len: int = 6):
        self.history_len = history_len
        self.tracks = {}
        self.last_seen = {}

    def update(self, plates: list, curr_time: float) -> list:
        # تنظيف اللوحات التي اختفت لأكثر من 1.8 ثانية
        expired = [tid for tid, t in self.last_seen.items() if curr_time - t > 1.8]
        for tid in expired:
            self.tracks.pop(tid, None)
            self.last_seen.pop(tid, None)

        stabilized_plates = []
        for p in plates:
            matched_tid = None
            # إذا توفر Track ID مباشر من خوارزمية ByteTrack המعتمدة
            if p.get("track_id") is not None:
                matched_tid = f"byte_{p['track_id']}"
            else:
                bx1, by1, bx2, by2 = p["bbox"]
                bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                min_dist = float("inf")
                for tid, hist in self.tracks.items():
                    if not hist: continue
                    last_p = hist[-1]
                    lx1, ly1, lx2, ly2 = last_p["bbox"]
                    lcx, lcy = (lx1 + lx2) / 2.0, (ly1 + ly2) / 2.0
                    dist = ((bcx - lcx)**2 + (bcy - lcy)**2)**0.5
                    if dist < 85 and dist < min_dist:
                        min_dist = dist
                        matched_tid = tid

            if matched_tid is None:
                matched_tid = f"tr_{int(curr_time*1000)%10000}_{len(self.tracks)}"

            if matched_tid not in self.tracks:
                self.tracks[matched_tid] = deque(maxlen=self.history_len)

            self.last_seen[matched_tid] = curr_time
            self.tracks[matched_tid].append(p)

            # التصويت بالأغلبية مع إعطاء الأولوية للوحات المطابقة للقواعد المرورية المصرية
            hist = self.tracks[matched_tid]
            valid_hist = [h for h in hist if h.get("text") and "?" not in h.get("text")]
            
            # إذا وجدت قراءات مطابقة قانونياً (syntax_valid == True)، يتم التصويت ضمنها أولاً
            legal_hist = [h for h in valid_hist if h.get("syntax_valid") is True]
            vote_pool = legal_hist if legal_hist else valid_hist

            if vote_pool:
                text_counts = Counter(h["text"] for h in vote_pool)
                stable_text = text_counts.most_common(1)[0][0]
                
                # تجميع الأرقام والحروف الأكثر تكراراً
                all_digits = [h["digits"] for h in vote_pool if h["digits"]]
                all_letters = [h["letters"] for h in vote_pool if h["letters"]]
                stable_digits = Counter(tuple(d) for d in all_digits).most_common(1)[0][0] if all_digits else p["digits"]
                stable_letters = Counter(tuple(l) for l in all_letters).most_common(1)[0][0] if all_letters else p["letters"]
                
                sample_item = next((h for h in vote_pool if h["text"] == stable_text), vote_pool[0])
                stable_syntax = sample_item.get("syntax_valid", False)
                stable_gov = sample_item.get("governorate", "غير محدد")
                stable_badge = sample_item.get("badge", "")
            else:
                stable_text = p["text"]
                stable_digits = p["digits"]
                stable_letters = p["letters"]
                stable_syntax = p.get("syntax_valid", False)
                stable_gov = p.get("governorate", "غير محدد")
                stable_badge = p.get("badge", "")

            stabilized_plates.append({
                "text": stable_text if stable_text else p["text"],
                "digits": list(stable_digits),
                "letters": list(stable_letters),
                "bbox": p["bbox"],
                "conf": p["conf"],
                "is_stabilized": len(vote_pool) >= 2,
                "syntax_valid": stable_syntax,
                "governorate": stable_gov,
                "badge": stable_badge
            })

        return stabilized_plates

def process_live_frame_dict(frame: np.ndarray, stabilizer: TemporalPlateStabilizer, engine, db) -> dict:
    """معالجة فريم حي مباشر من الكاميرا مع التثبيت الزمني وفحص القائمة السوداء باستخدام ByteTrack"""
    h, w = frame.shape[:2]
    try:
        p_res = engine.plate_model.track(
            frame, conf=0.35, device=engine.device, persist=True, tracker="bytetrack.yaml", verbose=False
        )[0]
    except Exception:
        p_res = engine.plate_model(frame, conf=0.35, device=engine.device, verbose=False)[0]

    raw_plates = []
    for b in p_res.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        bw, bh = x2 - x1, y2 - y1
        if bw < 42 or bh < 14:
            continue
        pad_x = int(bw * 0.05)
        pad_y = int(bh * 0.05)
        crop = frame[max(0, y1 - pad_y):min(h, y2 + pad_y), max(0, x1 - pad_x):min(w, x2 + pad_x)]
        rec = engine.recognize_plate_crop(crop)
        tid = int(b.id[0]) if (b.id is not None) else None
        raw_plates.append({
            "text": rec["text"],
            "digits": rec["digits"],
            "letters": rec["letters"],
            "bbox": [x1, y1, x2, y2],
            "track_id": tid,
            "conf": round(float(b.conf[0]), 2),
            "syntax_valid": rec.get("syntax_valid", False),
            "governorate": rec.get("governorate", "غير محدد"),
            "badge": rec.get("badge", "")
        })

    curr_t = time.time()
    plates = stabilizer.update(raw_plates, curr_t)

    has_alert = False
    alert_data = None

    for plate in plates:
        bl_info = db.check_blacklist(plate.get("digits", []), plate.get("letters", []))
        if bl_info is not None:
            plate["is_blacklist"] = True
            plate["blacklist_reason"] = bl_info["reason"]
            plate["blacklist_severity"] = bl_info["severity"]
            has_alert = True
            alert_data = {
                "plate": plate["text"],
                "reason": bl_info["reason"],
                "severity": bl_info["severity"]
            }
        else:
            plate["is_blacklist"] = False
            plate["blacklist_reason"] = ""
            plate["blacklist_severity"] = ""

        # تسجيل المركبة في قاعدة البيانات وحفظ لقطة اللوحة عند ثبوت القراءة
        if plate.get("is_stabilized") or (plate.get("syntax_valid") and len(plate.get("digits", [])) >= 3):
            bx1, by1, bx2, by2 = plate["bbox"]
            plate_crop = frame[max(0, by1):min(h, by2), max(0, bx1):min(w, bx2)]
            if plate_crop.size > 0:
                db.log_vehicle(plate_data=plate, crop_img=plate_crop, cooldown_seconds=20.0)

    return {
        "success": len(plates) > 0,
        "plates": plates,
        "alert": has_alert,
        "alert_info": alert_data,
        "frame_w": w,
        "frame_h": h
    }

class LiveFrameRequest(BaseModel):
    image: str

global_http_stabilizer = TemporalPlateStabilizer(history_len=6)

@app.post("/api/scan/live_frame")
async def scan_live_frame(req: LiveFrameRequest):
    """نقطة نهاية سريعة للمسح اللحظي عبر HTTP لأجهزة الآيفون في حال تقييد الـ WebSocket"""
    data = req.image
    if "," in data:
        data = data.split(",")[1]
    try:
        img_bytes = base64.b64decode(data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data")

    if frame is None:
        raise HTTPException(status_code=400, detail="Failed to decode image")

    engine = get_alpr_engine()
    db = get_alpr_database()
    result = process_live_frame_dict(frame, global_http_stabilizer, engine, db)
    return result

# =============================================================
# نقطة نهاية الـ WebSocket للكاميرا الحية المباشرة الفورية
# =============================================================
@app.websocket("/ws/scanner")
async def websocket_scanner(websocket: WebSocket):
    await websocket.accept()
    engine = get_alpr_engine()
    db = get_alpr_database()
    stabilizer = TemporalPlateStabilizer(history_len=6)
    fps_counter = 0
    t0 = time.time()
    try:
        while True:
            data = await websocket.receive_text()
            if "," in data:
                data = data.split(",")[1]
            try:
                img_bytes = base64.b64decode(data)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            except Exception:
                await websocket.send_json({"success": False, "plates": []})
                continue

            if frame is None:
                await websocket.send_json({"success": False, "plates": []})
                continue

            result = process_live_frame_dict(frame, stabilizer, engine, db)

            fps_counter += 1
            elapsed = time.time() - t0
            fps = round(fps_counter / max(0.001, elapsed), 1)
            if elapsed > 2.0:
                fps_counter = 0
                t0 = time.time()

            result["fps"] = fps
            await websocket.send_json(result)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print("WS error:", e)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=False)
