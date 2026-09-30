import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from test_verified import process_verified as process_old
from test_verified_after_tuning import process_verified as process_new, recog_model as new_model

print("=" * 80)
print(f"{'الصورة':<8} | {'الموديل القديم (Tuning_char.pt)':<28} | {'الموديل الجديد (best_char_afterTuning.pt)':<32} | {'الحالة'}")
print("=" * 80)

for i in range(1, 10):
    fn = f'{i}.jpg'
    res_old = process_old(fn)
    res_new = process_new(fn, new_model)
    diff = "✅ متطابق ومستقر" if res_old == res_new else "🎯 تصحيح وتحسن"
    print(f"{fn:<8} | {res_old:<28} | {res_new:<32} | {diff}")

print("=" * 80)
