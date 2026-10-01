# 🚀 دليل وكود التدريب على Google Colab (Egyptian ALPR Phase 3 - Real Plates Fine-Tuning)

## 📁 الملفات المطلوب رفعها إلى Google Colab:
ارفع هذين الملفين فقط إلى لوحة الملفات (Files 📁) على يسار شاشة Colab:
1. **ملف الداتا سيت المضغوط** الذي قمت بتحميله للتو:  
   `egyptian_chars_verified_dataset_*.zip` (يحتوي على الـ 748 لوحة المعتمدة بشرياً).  
   *(ملاحظة: انتظر حتى تكتمل دائرة الرفع الدائرية وتختفي بجانب اسم الملف).*
2. **ملف الموديل الحالي** من مجلد المشروع في جهازك:  
   `best_char_afterTuning.pt` (الموجود في مجلد المشروع الرئيسي: `e:\tik tok\Car_Palate\best_char_afterTuning.pt`).

---

## 💻 الكود الكامل والمعدل: انسخه وضعه في خلية واحدة (Cell) في Colab واضغط تشغيل:

```python
# =====================================================================
# 1. التأكد من كارت الشاشة GPU وتثبيت مكتبة Ultralytics
# =====================================================================
import torch
print("GPU Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
else:
    print("⚠️ تنبيه: يرجى تفعيل كارت الشاشة أولاً من القائمة العلوية:")
    print("Runtime -> Change runtime type -> T4 GPU")

!pip install -q ultralytics

# =====================================================================
# 2. فك ضغط الداتا سيت ومعالجة الملفات تلقائياً (بأمان تام)
# =====================================================================
import os
import glob
import zipfile
import yaml

# البحث التلقائي عن أي ملف zip تم رفعه (حتى لو كان باسم فيه مسافات أو أقواس مثل (1).zip)
zip_candidates = sorted(glob.glob('*.zip'), key=os.path.getmtime, reverse=True)
if not zip_candidates:
    raise FileNotFoundError("❌ لم يتم العثور على أي ملف zip! يرجى سحب ملف الداتا سيت وإفلاته في لوحة الملفات على اليسار.")

zip_target = zip_candidates[0]
print(f"📦 جاري فك ضغط: {zip_target} عبر مكتبة zipfile...")

with zipfile.ZipFile(zip_target, 'r') as zip_ref:
    zip_ref.extractall('.')

print("✅ تم فك الضغط بنجاح!")

# تحديد مكان مجلد الصور تلقائياً أينما استقر
images_train_candidates = glob.glob('**/images/train', recursive=True)
if images_train_candidates:
    dataset_root = os.path.dirname(os.path.dirname(os.path.abspath(images_train_candidates[0])))
else:
    dataset_root = os.path.abspath('dataset')

train_path = os.path.join(dataset_root, 'images', 'train')
val_path = os.path.join(dataset_root, 'images', 'val')

# قائمة الفئات الـ 38 الرسمية لأحرف وأرقام اللوحات المصرية
classes_38 = [
    '0', '1', '2', '3', '4', '5', '6', '7', '7aa', '8', '9', 
    'Taa', 'Thaa', 'ain', 'alif', 'baa', 'daad', 'daal', 'faa', 'ghayn', 
    'haa', 'jeem', 'kaaf', 'khaa', 'laam', 'meem', 'noon', 'qaaf', 'raa', 
    'saad', 'seen', 'sheen', 'taa', 'thaa', 'waw', 'yaa', 'zaal', 'zay'
]

# إنشاء وضمان ملف data.yaml بمسارات مطلقة صحيحة 100%
data_yaml_path = os.path.join(dataset_root, 'data.yaml')
yaml_content = {
    'path': dataset_root,
    'train': 'images/train',
    'val': 'images/val',
    'nc': len(classes_38),
    'names': classes_38
}

with open(data_yaml_path, 'w', encoding='utf-8') as f:
    yaml.dump(yaml_content, f, allow_unicode=True)

train_imgs = glob.glob(os.path.join(train_path, '*.jpg'))
val_imgs = glob.glob(os.path.join(val_path, '*.jpg'))

print(f"📄 مسار ملف الإعدادات: {data_yaml_path}")
print(f"   - صور التدريب (Train): {len(train_imgs)}")
print(f"   - صور التحقق (Val): {len(val_imgs)}")

if len(train_imgs) == 0:
    raise RuntimeError("⚠️ لم يتم العثور على صور تدريب! قد يكون الملف قيد الرفع في كولاب، انتظر حتى يكتمل رفعه ثم أعد التشغيل.")

# =====================================================================
# 3. تحميل الموديل وانطلاق التدريب والضبط الدقيق (Fine-Tuning)
# =====================================================================
from ultralytics import YOLO

# البحث عن ملف الموديل
weights_candidates = ['best_char_afterTuning.pt', 'Tuning_char.pt', 'best_char.pt']
model_path = None
for w in weights_candidates:
    if os.path.exists(w):
        model_path = w
        break

if not model_path:
    pts = sorted(glob.glob('*.pt'), key=os.path.getmtime, reverse=True)
    if pts:
        model_path = pts[0]
    else:
        raise FileNotFoundError("❌ لم يتم العثور على ملف الموديل (.pt)! يرجى رفع best_char_afterTuning.pt إلى Colab.")

print(f"🧠 جاري تحميل الأوزان من: {model_path} ...")
model = YOLO(model_path)

print("🚀 انطلاق الضبط الدقيق (40 Epochs لتعلم الحالات الصعبة والمشوشة)...")
results = model.train(
    data=data_yaml_path,
    epochs=40,                  # 40 إيبوك تستغرق حوالي 2.5 دقيقة على T4 GPU
    imgsz=320,                  # الأبعاد المثالية للوحات المقصوصة
    batch=16,
    lr0=0.0008,                 # معدل تعلم متزن يحافظ على الأساسيات ويعالج الحالات الضعيفة
    lrf=0.01,
    
    # --- إعدادات الحفاظ على اتجاه وهندسة الحروف العربية ---
    fliplr=0.0,                 # 🚫 ممنوع قلب الصور أفقياً (حماية الحروف والأرقام من الانعكاس)
    flipud=0.0,                 # 🚫 ممنوع قلب الصور رأسياً
    degrees=3.0,                # محاكاة زوايا التصوير المائلة
    perspective=0.0004,         # محاكاة منظور الكاميرا الواقعي
    hsv_h=0.015,                # تنوع ألوان خفيف
    hsv_s=0.5,                  # تنوع تشبع الألوان
    hsv_v=0.4,                  # محاكاة الظلال القوية والشمس الساطعة
    scale=0.15,                 # محاكاة اختلاف حجم اللوحة
    mosaic=0.2,                 # دمج الصور لزيادة التركيز
    close_mosaic=10,            # إيقاف الـ mosaic في آخر 10 إيبوكس لدقة التفاصيل
    
    name='egyptian_char_real_tuning_v3',
    device=0,
    save=True,
    plots=True
)

print("\n" + "=" * 65)
print("🎉 اكتمل التدريب بنجاح تام وتم تحديث قدرات الموديل على اللوحات الصعبة!")
print("=" * 65 + "\n")

# =====================================================================
# 4. تنزيل الموديل الجديد المطور إلى جهازك مباشرة
# =====================================================================
from google.colab import files

best_weight = 'runs/detect/egyptian_char_real_tuning_v3/weights/best.pt'
if os.path.exists(best_weight):
    print("📥 جاري بدء تنزيل الموديل الجديد المطور (best.pt) إلى جهازك...")
    files.download(best_weight)
else:
    print(f"الملف موجود في: {best_weight}")
```
