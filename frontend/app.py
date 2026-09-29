import io
import os
import base64
import requests
import streamlit as st
from PIL import Image

# =============================================================
# 1. إعداد الصفحة والواجهة المتجاوبة للموبايل
# =============================================================
st.set_page_config(
    page_title="كاشف لوحات السيارات المصرية الذكي",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# تخصيص CSS عصري ويدعم الموبايل واللغة العربية (RTL)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Cairo', sans-serif;
        direction: rtl;
        text-align: right;
    }
    
    /* رأس الصفحة المتدرج الأنيق */
    .hero-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 22px 18px;
        border-radius: 18px;
        color: white;
        text-align: center;
        margin-bottom: 22px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    .hero-header h1 {
        font-size: 1.7rem;
        margin-bottom: 6px;
        color: #ffffff !important;
        font-weight: 800;
    }
    .hero-header p {
        font-size: 0.95rem;
        color: #e0e7ff;
        margin-bottom: 0px;
    }
    
    /* بطاقة عرض النتيجة الكبيرة */
    .plate-card {
        background: #0f172a;
        border: 2px solid #38bdf8;
        border-radius: 16px;
        padding: 16px;
        text-align: center;
        margin: 15px 0;
        box-shadow: 0 4px 20px rgba(56, 189, 248, 0.2);
    }
    .plate-title {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .plate-text {
        font-size: 2.2rem;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: 2px;
        line-height: 1.3;
    }
    .meta-badge {
        display: inline-block;
        background: rgba(34, 197, 94, 0.2);
        color: #4ade80;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 6px;
    }
    
    /* تحسين الكاميرا ومدخلات الموبايل */
    .stCamera, .stFileUploader {
        border-radius: 14px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================
# 2. الشريط الجانبي (إعدادات الاتصال بالـ FastAPI Backend)
# =============================================================
with st.sidebar:
    st.header("⚙️ إعدادات النظام")
    backend_url = st.text_input("رابط خادم FastAPI:", value="http://localhost:8000")
    
    # فحص حالة الخادم
    server_online = False
    try:
        r = requests.get(f"{backend_url}/health", timeout=1.5)
        if r.status_code == 200:
            server_online = True
            st.success("🟢 متصل بالـ FastAPI Backend")
        else:
            st.warning("🟡 الخادم يرد برمز غير متوقع")
    except Exception:
        st.error("🔴 تعذر الاتصال بـ FastAPI (سيتم استخدام المعالج المحلي)")
        
    st.divider()
    st.caption("نظام كشف وتحقيق لوحات السيارات المصرية v2.0 - معتمد على الذكاء الاصطناعي متعدد المراحل.")

# =============================================================
# 3. واجهة الصفحة الرئيسية
# =============================================================
st.markdown("""
<div class="hero-header">
    <h1>🚗 كاشف لوحات السيارات الذكي</h1>
    <p>تعرف لحظي فائق الدقة على الحروف والأرقام بنظام التحقق المتعدد</p>
</div>
""", unsafe_allow_html=True)

tab_cam, tab_file, tab_demo = st.tabs(["📷 كاميرا الموبايل", "📁 رفع صورة", "✨ عينات تجريبية"])

# وظيفة مساعدة لإرسال الصورة واستقبال النتيجة
def analyze_image_bytes(image_bytes: bytes, file_name: str = "image.jpg"):
    """إرسال الصورة إلى FastAPI أو معالجتها محلياً إذا كان الخادم مغلقاً"""
    if server_online:
        try:
            files = {"file": (file_name, image_bytes, "image/jpeg")}
            resp = requests.post(f"{backend_url}/predict/image", files=files, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                return data, "api"
        except Exception as e:
            st.warning(f"حدث خطأ أثناء الاتصال بالـ API: {e}، جاري التحويل للمعالج المباشر...")

    # الوضع الاحتياطي المحلي في حال عدم تشغيل الـ FastAPI
    from core.pipeline import EgyptianALPR
    import cv2
    import numpy as np

    if "local_engine" not in st.session_state:
        st.session_state.local_engine = EgyptianALPR()

    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    res = st.session_state.local_engine.process_image(img, add_hud=True)

    # تحويل لـ Base64
    _, buf = cv2.imencode(".jpg", res["annotated_image"])
    b64_str = base64.b64encode(buf).decode("utf-8")

    formatted = {
        "success": res["success"],
        "plates_count": res["plates_count"],
        "plates": res["plates"],
        "annotated_image_base64": b64_str,
        "message": "تم الفحص بالمعالج الداخلي"
    }
    return formatted, "local"

def render_results(data: dict):
    """عرض النتائج بأسلوب بطاقات فخم يدعم الموبايل"""
    if not data or not data.get("success"):
        st.warning("⚠️ لم يتم رصد لوحة واضحة في الصورة، يرجى إعادة المحاولة من زاوية أفضل.")
        return

    plates = data.get("plates", [])
    for idx, p in enumerate(plates, 1):
        plate_str = p.get("text", "غير محدد")
        conf = p.get("confidence", 0.0)

        st.markdown(f"""
        <div class="plate-card">
            <div class="plate-title">لوحة سيارة رقم {idx}</div>
            <div class="plate-text">{plate_str}</div>
            <div class="meta-badge">دقة الكشف: {int(conf * 100)}%</div>
        </div>
        """, unsafe_allow_html=True)

    # عرض الصورة المعالجة مع صناديق الموديل الثاني ونافذة الزووم
    b64_img = data.get("annotated_image_base64")
    if b64_img:
        img_bytes = base64.b64decode(b64_img)
        st.image(img_bytes, caption="الصورة المعالجة (توضح صناديق الحروف والزووم الفائق)", use_container_width=True)

# -------------------------------------------------------------
# -------------------------------------------------------------
# تبويب 1: كاميرا الموبايل وتصوير اللوحات
# -------------------------------------------------------------
with tab_cam:
    st.markdown("### 🎥 الكاميرا الحية المباشرة (Live Viewfinder)")
    st.write("وجّه الكاميرا نحو لوحة السيارة والتقط الصورة مباشرة:")
    live_cam = st.camera_input("معاينة الكاميرا الحية")
    
    if live_cam is not None:
        with st.spinner("⏳ جاري الفحص واستخراج المحارف..."):
            bytes_data = live_cam.getvalue()
            res_data, mode = analyze_image_bytes(bytes_data, "live_cam.jpg")
            render_results(res_data)
            
    st.divider()
    with st.expander("📷 أو التقاط صورة عبر تطبيق كاميرا الآيفون العادي"):
        mobile_cam = st.file_uploader(
            "اضغط هنا لفتح تطبيق الكاميرا الأصلي:",
            type=["jpg", "jpeg", "png", "webp"],
            key="mobile_cam_uploader"
        )
        if mobile_cam is not None:
            with st.spinner("⏳ جاري الفحص..."):
                bytes_data = mobile_cam.getvalue()
                res_data, mode = analyze_image_bytes(bytes_data, mobile_cam.name)
                render_results(res_data)

# -------------------------------------------------------------
# تبويب 2: رفع صورة من المعرض
# -------------------------------------------------------------
with tab_file:
    uploaded = st.file_uploader("اختر صورة لوحة محفوظة:", type=["jpg", "jpeg", "png", "webp"], key="file_gallery")
    if uploaded is not None:
        with st.spinner("⏳ جاري الفحص..."):
            bytes_data = uploaded.getvalue()
            res_data, mode = analyze_image_bytes(bytes_data, uploaded.name)
            render_results(res_data)

# -------------------------------------------------------------
# تبويب 3: عينات تجريبية جاهزة للتحقق الفوري
# -------------------------------------------------------------
with tab_demo:
    st.write("🧪 جرب بنقرة واحدة على صور اللوحات التي تم تدقيقها:")
    cols = st.columns(3)
    sample_imgs = [
        ("صورة 1 (س ب)", "1.jpg"),
        ("صورة 3 (٧ ٢ ٧)", "3.jpg"),
        ("صورة 4 (١ ٥ ٤ ٧)", "4.jpg"),
        ("صورة 5 (ع ص ي)", "5.jpg"),
        ("صورة 7 (ل س س)", "7.jpg"),
        ("صورة 8 (ق ق)", "8.jpg"),
    ]
    for i, (title, path) in enumerate(sample_imgs):
        with cols[i % 3]:
            if st.button(title, key=f"btn_{path}", use_container_width=True):
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        b_data = f.read()
                    with st.spinner(f"جاري فحص {title}..."):
                        res_data, _ = analyze_image_bytes(b_data, path)
                        render_results(res_data)
                else:
                    st.error(f"الملف {path} غير موجود.")
