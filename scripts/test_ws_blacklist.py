import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import asyncio
import ssl
import websockets
import json
import base64
import cv2
import numpy as np

from generator.plate_engine import EgyptianPlateGenerator

async def test_websocket_blacklist():
    print("🚗 1. Generating plate matching blacklist: 8429 س ب...")
    generator = EgyptianPlateGenerator()
    # digits: ٨ ٤ ٢ ٩, letters: س ب
    plate_img, char_boxes = generator.draw_base_plate(
        digits=['٨', '٤', '٢', '٩'],
        letters=['س', 'ب'],
        style_key='private'
    )

    # وضع اللوحة داخل صورة فريم كاميرا بحجم 720x480
    frame = np.full((480, 720, 3), 45, dtype=np.uint8)
    ph, pw = plate_img.shape[:2]
    target_w = 340
    target_h = int(ph * (target_w / pw))
    resized_plate = cv2.resize(plate_img, (target_w, target_h))
    
    # لصقها في منتصف الفريم
    sy = (480 - target_h) // 2
    sx = (720 - target_w) // 2
    frame[sy:sy+target_h, sx:sx+target_w] = resized_plate

    # تحويل الفريم إلى Base64
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64_frame = "data:image/jpeg;base64," + base64.b64encode(buf).decode('utf-8')

    print("🔌 2. Connecting to WebSocket wss://127.0.0.1:8000/ws/scanner...")
    ssl_context = ssl._create_unverified_context()

    async with websockets.connect("wss://127.0.0.1:8000/ws/scanner", ssl=ssl_context) as ws:
        print(" Connected! Sending 4 consecutive frames for stabilization & blacklist verification...")
        for i in range(4):
            await ws.send(b64_frame)
            resp = await ws.recv()
            data = json.loads(resp)
            print(f"\n--- Frame {i+1} Response ---")
            print("   Success:", data.get("success"))
            print("   Plates found:", len(data.get("plates", [])))
            print("   Alert triggered:", data.get("alert"))
            if data.get("plates"):
                p = data["plates"][0]
                print(f"   Plate Text: {p.get('text')}")
                print(f"   Stabilized: {p.get('is_stabilized')}")
                print(f"   Is Blacklist: {p.get('is_blacklist')}")
                print(f"   Blacklist Reason: {p.get('blacklist_reason')}")
            if data.get("alert_info"):
                print("   🚨 Alert Info:", data.get("alert_info"))
            await asyncio.sleep(0.15)

    print("\n✅ WEBSOCKET BLACKLIST TEST COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    asyncio.run(test_websocket_blacklist())
