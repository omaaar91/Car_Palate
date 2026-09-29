import os
import sys
import cv2

sys.stdout.reconfigure(encoding='utf-8')
from core.pipeline import EgyptianALPR

alpr = EgyptianALPR()

images = ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', '6.jpg', '7.jpg', '8.jpg', '9.jpg', 'test1.jpg', 'test2.jpg', 'test3.jpg', 'test4.png']

print(f"{'Image':10s} | {'Digits':12s} | {'Letters':10s} | {'Total':5s} | {'Gov Category'}")
print("-" * 65)

for img_name in images:
    img = cv2.imread(img_name)
    if img is None: continue
    res = alpr.process_image(img, add_hud=False)
    p = res['plates'][0]
    d_len = len(p['digits'])
    l_len = len(p['letters'])
    tot = d_len + l_len
    
    if d_len == 3 and l_len == 3:
        gov = "القاهرة (Cairo)"
    elif d_len == 4 and l_len == 2:
        gov = "الجيزة (Giza)"
    elif d_len == 4 and l_len == 3:
        gov = "المحافظات (Other)"
    else:
        gov = f"غير قياسي ({d_len}D + {l_len}L)"
        
    d_str = " ".join(p['digits'])
    l_str = " ".join(p['letters'])
    print(f"{img_name:10s} | {d_str:12s} | {l_str:10s} | {tot:<5d} | {gov}")
