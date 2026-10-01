
"""
benchmark_gpu_vs_cpu.py
"""
import os, sys, time, random
sys.stdout.reconfigure(encoding='utf-8')
import cv2, numpy as np, torch

# ─── 0. معلومات الجهاز ────────────────────────────────
print("=" * 55)
print("  معلومات الجهاز")
print("=" * 55)
print(f"  PyTorch : {torch.__version__}")
print(f"  CUDA    : {torch.version.cuda or 'غير مفعّل'}")
print(f"  Device  : {'CUDA - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU فقط'}")
if torch.cuda.is_available():
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  VRAM    : {vram:.1f} GB")
print("=" * 55)

# ─── 1. تحميل الـ pipeline ───────────────────────────
from core.pipeline import EgyptianALPR

print("\n  جاري تحميل النماذج...")
t_load = time.perf_counter()
engine = EgyptianALPR()
load_time = (time.perf_counter() - t_load) * 1000
print(f"  ✅ تم التحميل في {load_time:.0f}ms\n")

# ─── 2. اختيار 10 صور عشوائية ────────────────────────
CROPS_DIR = os.path.join("data", "crops")
all_imgs = [f for f in os.listdir(CROPS_DIR) if f.endswith(".jpg")]
selected = random.sample(all_imgs, min(10, len(all_imgs)))

print("  الصور المختارة للاختبار:")
for i, f in enumerate(selected, 1):
    print(f"    {i:2}. {f}")

# ─── 3. Warmup run ───────────────────────────────────
print("\n  جاري الـ warmup...")
dummy = cv2.imread(os.path.join(CROPS_DIR, selected[0]))
engine.recognize_plate_crop(dummy)
print("  ✅ Warmup انتهى\n")

# ─── 4. Benchmark ────────────────────────────────────
print("=" * 55)
print("  نتائج البنشمارك")
print("=" * 55)
print(f"  {'#':<4} {'الصورة':<35} {'الوقت':>8}  {'النتيجة'}")
print("-" * 55)

times = []
results = []

for i, fname in enumerate(selected, 1):
    img_path = os.path.join(CROPS_DIR, fname)
    img = cv2.imread(img_path)
    if img is None:
        print(f"  {i:<4} {fname:<35} {'خطأ':>8}")
        continue

    t0 = time.perf_counter()
    res = engine.recognize_plate_crop(img)
    elapsed = (time.perf_counter() - t0) * 1000
    times.append(elapsed)

    plate_text = res.get("text", "---")
    valid = "✅" if res.get("syntax_valid") else "⚠️"
    short_name = fname[:33] + ".." if len(fname) > 35 else fname
    print(f"  {i:<4} {short_name:<35} {elapsed:>6.0f}ms  {valid} {plate_text}")

# ─── 5. الملخص الإحصائي ─────────────────────────────
print("=" * 55)
if times:
    avg   = sum(times) / len(times)
    mn    = min(times)
    mx    = max(times)
    fps   = 1000 / avg

    device_label = "CUDA GPU" if torch.cuda.is_available() else "CPU"
    print(f"  الجهاز         : {device_label}")
    print(f"  وقت التحميل   : {load_time:.0f}ms")
    print(f"  متوسط الوقت   : {avg:.1f}ms/صورة")
    print(f"  أسرع صورة     : {mn:.1f}ms")
    print(f"  أبطأ صورة     : {mx:.1f}ms")
    print(f"  FPS المقدّر   : {fps:.1f} frame/sec")
    print("=" * 55)

    # حفظ النتيجة في ملف للمقارنة لاحقاً
    mode = "cuda" if torch.cuda.is_available() else "cpu"
    with open(f"benchmark_result_{mode}.txt", "w", encoding="utf-8") as f:
        f.write(f"Device: {device_label}\n")
        f.write(f"Load time: {load_time:.0f}ms\n")
        f.write(f"Avg: {avg:.1f}ms\n")
        f.write(f"Min: {mn:.1f}ms\n")
        f.write(f"Max: {mx:.1f}ms\n")
        f.write(f"FPS: {fps:.1f}\n")
    print(f"\n  ✅ النتيجة محفوظة في: benchmark_result_{mode}.txt")
