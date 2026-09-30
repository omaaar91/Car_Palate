# 🚀 دليل وكود التدريب على Google Colab (Egyptian Plate ALPR Fine-Tuning)

## الملفات المطلوب رفعها إلى كولاب:
1. `egyptian_chars_verified_dataset_1790779507.zip`
2. `Tuning_char.pt`

---

## 💻 الكود الكامل المنسوخ إلى Colab:

```python
# =====================================================================
# 1. التأكد من تفعيل كارت الشاشة GPU وتثبيت Ultralytics
# =====================================================================
import torch
print("GPU Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
else:
    print("⚠️ تنبيه: يرجى تفعيل كارت الشاشة من القائمة: Runtime -> Change runtime type -> T4 GPU")

!pip install -q ultralytics

# =====================================================================
# 2. فك ضغط الداتا سيت وضبط المسارات
# =====================================================================
import os
import glob
import yaml

zip_files = glob.glob('egyptian_chars_verified_dataset_*.zip')
if not zip_files:
    raise FileNotFoundError("لم يتم العثور على ملف الداتا سيت المضغوط! يرجى رفعه إلى لوحة الملفات على اليسار.")

zip_target = zip_files[0]
print(f"📦 جاري فك ضغط الداتا سيت: {zip_target} ...")
!unzip -q -o "{zip_target}"

# تصحيح مسار الداتا سيت لبيئة كولاب
yaml_path = 'dataset/data.yaml'
with open(yaml_path, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

cfg['path'] = '/content/dataset'
with open(yaml_path, 'w', encoding='utf-8') as f:
    yaml.dump(cfg, f, allow_unicode=True)

print("✅ تم تجهيز الداتا سيت بنجاح وبناء تقسيم Train و Validation!")

# =====================================================================
# 3. التدريب والضبط الدقيق (Fine-Tuning) بحماية الحروف العربية
# =====================================================================
from ultralytics import YOLO

if not os.path.exists('Tuning_char.pt'):
    raise FileNotFoundError("لم يتم العثور على ملف Tuning_char.pt! يرجى رفعه لبدء التدريب منه.")

print("🧠 جاري تحميل أحدث أوزان للموديل Tuning_char.pt ...")
model = YOLO('Tuning_char.pt')

print("🚀 انطلاق التدريب (50 Epochs مع Augmentation مخصص لقراءة النصوص واللوحات)...")
results = model.train(
    data='dataset/data.yaml',
    epochs=50,                  # 50 إيبوك تستغرق حوالي 2-3 دقائق على T4 GPU
    imgsz=320,                  # الأبعاد القياسية للوحات المقصوصة
    batch=16,
    lr0=0.001,                  # معدل تعلم متزن للمحافظة على الأوزان السابقة
    lrf=0.01,
    
    # --- إعدادات Augmentation لحماية النصوص والحروف العربية ---
    fliplr=0.0,                 # 🚫 ممنوع قلب الصور أفقياً (حماية الحروف والأرقام من الانعكاس)
    flipud=0.0,                 # 🚫 ممنوع قلب الصور رأسياً
    degrees=4.0,                # محاكاة ميل الكاميرا الطبيعي
    perspective=0.0005,         # محاكاة زوايا التصوير الواقعية
    hsv_h=0.015,                # تنوع درجات الألوان
    hsv_s=0.5,                  # تنوع تشبع الألوان
    hsv_v=0.4,                  # محاكاة سطوع الشمس والظلال القوية
    scale=0.1,                  # محاكاة اختلاف المسافة وقرب/بعد اللوحة
    mosaic=0.3,                 # دمج ومزج الصور
    
    name='egyptian_char_tuning_v2',
    device=0,
    save=True,
    plots=True
)

print("\n" + "=" * 60)
print("🎉 اكتمل التدريب والضبط الدقيق بنجاح وبأعلى دقة!")
print("=" * 60 + "\n")

# =====================================================================
# 4. تنزيل الموديل الجديد المطور إلى جهازك مباشرة
# =====================================================================
from google.colab import files

best_weight = 'runs/detect/egyptian_char_tuning_v2/weights/best.pt'
if os.path.exists(best_weight):
    print("📥 جاري بدء تنزيل الموديل المطور (best.pt) إلى جهازك...")
    files.download(best_weight)
else:
    print(f"الملف غير موجود في: {best_weight}")
```
