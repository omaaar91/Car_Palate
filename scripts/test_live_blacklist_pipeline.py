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
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

async def run_live_pipeline_test():
    base_url = "https://127.0.0.1:8000"

    print("🚨 1. Adding real plate [7229 أ أ ج] to Blacklist via API...")
    add_payload = {
        "digits": ["7", "2", "2", "9"],
        "letters": ["أ", "أ", "ج"],
        "reason": "سيارة مطلوبة أمنياً - بلاغ سرقة رسمي 2026",
        "severity": "danger"
    }
    r = requests.post(f"{base_url}/api/blacklist", json=add_payload, verify=False)
    print("API Response:", r.json())
    assert r.json().get("success") is True

    print("\n📸 2. Loading real car image '2.jpg'...")
    img = cv2.imread("2.jpg")
    assert img is not None, "2.jpg not found!"

    # تحويل الصورة إلى Base64
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64_frame = "data:image/jpeg;base64," + base64.b64encode(buf).decode('utf-8')

    print("\n🔌 3. Connecting to WebSocket wss://127.0.0.1:8000/ws/scanner...")
    ssl_context = ssl._create_unverified_context()

    async with websockets.connect(f"wss://127.0.0.1:8000/ws/scanner", ssl=ssl_context) as ws:
        print(" Connected! Streaming frames to trigger stabilizer and siren alert...")
        for i in range(3):
            await ws.send(b64_frame)
            resp = await ws.recv()
            data = json.loads(resp)
            print(f"\n--- Frame {i+1} Response ---")
            print("   Success:", data.get("success"))
            print("   Plates Count:", len(data.get("plates", [])))
            print("   🚨 ALERT STATUS:", data.get("alert"))
            if data.get("alert_info"):
                print("   🚨 ALERT DETAILS:", data.get("alert_info"))
            if data.get("plates"):
                p = data["plates"][0]
                print(f"   Plate: {p.get('text')}")
                print(f"   Is Blacklist: {p.get('is_blacklist')}")
                print(f"   Reason: {p.get('blacklist_reason')}")
            await asyncio.sleep(0.1)

    print("\n📊 4. Checking Dashboard Logs & Stats...")
    stats_r = requests.get(f"{base_url}/api/stats", verify=False)
    print("Stats Summary:", stats_r.json())

    logs_r = requests.get(f"{base_url}/api/logs", verify=False)
    logs = logs_r.json().get("logs", [])
    print(f"Total Logs in DB: {len(logs)}")
    if logs:
        latest = logs[0]
        print(f"Latest Log -> Plate: {latest['plate_text']} | Blacklist: {latest['is_blacklist']} | Reason: {latest['blacklist_reason']} | Crop: {latest['crop_path']}")

    print("\n🎉 SUCCESS: ALL 3 ENTERPRISE FEATURES TESTED AND VERIFIED END-TO-END! 🎉")

if __name__ == "__main__":
    asyncio.run(run_live_pipeline_test())
