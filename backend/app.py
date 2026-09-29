import os
import time
import base64
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, FileResponse

from core.pipeline import EgyptianALPR
from backend.schemas import ALPRResponse, PlateResult

app = FastAPI(
    title="Egyptian License Plate Recognition API",
    description="واجهة برمجية فائقة الدقة للتعرف على لوحات السيارات المصرية باستخدام الذكاء الاصطناعي متعدد المراحل",
    version="2.0.0"
)

# تمكين CORS لدعم الاتصال من Streamlit والتطبيقات المحمولة
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Egyptian ALPR API",
        "version": "2.0.0",
        "docs": "/docs"
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
# مسار صفحة الكاميرا الحية المباشرة (AR Scanner Client)
# =============================================================
@app.get("/scanner")
def get_scanner_page():
    scanner_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "scanner.html")
    if os.path.exists(scanner_path):
        return FileResponse(scanner_path)
    return Response(content="<h1>Scanner page not found</h1>", media_type="text/html", status_code=404)

from collections import deque, Counter

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
            del self.tracks[tid]
            del self.last_seen[tid]

        stabilized_plates = []
        for p in plates:
            bx1, by1, bx2, by2 = p["bbox"]
            bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0

            # ربط اللوحة بأقرب مسار سابق عبر المسافة المكانية
            matched_tid = None
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

# =============================================================
# نقطة نهاية الـ WebSocket للكاميرا الحية المباشرة الفورية
# =============================================================
@app.websocket("/ws/scanner")
async def websocket_scanner(websocket: WebSocket):
    await websocket.accept()
    engine = get_alpr_engine()
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

            h, w = frame.shape[:2]
            p_res = engine.plate_model(frame, conf=0.35, verbose=False)[0]
            raw_plates = []
            for b in p_res.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                bw, bh = x2 - x1, y2 - y1
                # تجاهل الصناديق متناهية الصغر التي تسبب تشويشاً
                if bw < 42 or bh < 14:
                    continue
                crop = frame[y1:y2, x1:x2]
                rec = engine.recognize_plate_crop(crop)
                raw_plates.append({
                    "text": rec["text"],
                    "digits": rec["digits"],
                    "letters": rec["letters"],
                    "bbox": [x1, y1, x2, y2],
                    "conf": round(float(b.conf[0]), 2),
                    "syntax_valid": rec.get("syntax_valid", False),
                    "governorate": rec.get("governorate", "غير محدد"),
                    "badge": rec.get("badge", "")
                })

            # تطبيق التثبيت والتصويت الزمني عبر الفريمات المتتالية
            curr_t = time.time()
            plates = stabilizer.update(raw_plates, curr_t)

            fps_counter += 1
            elapsed = time.time() - t0
            fps = round(fps_counter / max(0.001, elapsed), 1)
            if elapsed > 2.0:
                fps_counter = 0
                t0 = time.time()

            await websocket.send_json({
                "success": len(plates) > 0,
                "plates": plates,
                "frame_w": w,
                "frame_h": h,
                "fps": fps
            })
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print("WS error:", e)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=False)
