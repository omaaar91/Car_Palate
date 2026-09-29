import os
import sys
import cv2

sys.stdout.reconfigure(encoding='utf-8')
from core.pipeline import EgyptianALPR

alpr = EgyptianALPR()

test_images = sorted([
    f for f in os.listdir('.') 
    if (f.endswith('.jpg') or f.endswith('.png')) 
    and ('test' in f or f[:-4].isdigit())
])

print(f"=== Testing on {len(test_images)} images ===")
for img_name in test_images:
    img = cv2.imread(img_name)
    if img is None:
        continue
    res = alpr.process_image(img, add_hud=False)
    if res['plates']:
        p = res['plates'][0]
        digits_str = " ".join(p['digits'])
        letters_str = " ".join(p['letters'])
        print(f"{img_name:12s} -> [{digits_str}] | [{letters_str}]  (conf: {p['confidence']:.2f})")
    else:
        print(f"{img_name:12s} -> NO PLATE DETECTED")
