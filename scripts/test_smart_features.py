import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import requests
import json
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_URL = "https://127.0.0.1:8000"

def test_endpoints():
    print("--- 1. Testing /health ---")
    r = requests.get(f"{BASE_URL}/health", verify=False)
    print(r.status_code, r.json())
    assert r.status_code == 200

    print("--- 2. Testing /dashboard ---")
    r = requests.get(f"{BASE_URL}/dashboard", verify=False)
    print(r.status_code, "Length:", len(r.text))
    assert r.status_code == 200
    assert "لوحة التحكم" in r.text

    print("--- 3. Testing /scanner ---")
    r = requests.get(f"{BASE_URL}/scanner", verify=False)
    print(r.status_code, "Length:", len(r.text))
    assert r.status_code == 200
    assert "manifest.json" in r.text
    assert "triggerSiren" in r.text

    print("--- 4. Testing /manifest.json ---")
    r = requests.get(f"{BASE_URL}/manifest.json", verify=False)
    print(r.status_code, r.headers.get("content-type"))
    assert r.status_code == 200
    m_json = r.json()
    print("App Name:", m_json.get("name"))

    print("--- 5. Testing /sw.js ---")
    r = requests.get(f"{BASE_URL}/sw.js", verify=False)
    print(r.status_code, r.headers.get("content-type"))
    assert r.status_code == 200

    print("--- 6. Testing /api/stats ---")
    r = requests.get(f"{BASE_URL}/api/stats", verify=False)
    print(r.status_code, r.json())
    assert r.status_code == 200

    print("--- 7. Testing /api/blacklist (GET) ---")
    r = requests.get(f"{BASE_URL}/api/blacklist", verify=False)
    print(r.status_code, "Items count:", len(r.json().get("items", [])))
    assert r.status_code == 200

    print("--- 8. Testing /api/blacklist (POST) ---")
    payload = {
        "digits": ["7", "7", "7"],
        "letters": ["س", "س", "س"],
        "reason": "تجربة حية لوحة مشبوهة",
        "severity": "danger"
    }
    r = requests.post(f"{BASE_URL}/api/blacklist", json=payload, verify=False)
    print(r.status_code, r.json())
    assert r.status_code == 200
    assert r.json().get("success") is True

    print("--- 9. Testing /api/logs ---")
    r = requests.get(f"{BASE_URL}/api/logs", verify=False)
    print(r.status_code, "Logs count:", len(r.json().get("logs", [])))
    assert r.status_code == 200

    print("--- 10. Testing /api/export/csv ---")
    r = requests.get(f"{BASE_URL}/api/export/csv", verify=False)
    print(r.status_code, r.headers.get("content-type"), "Bytes:", len(r.content))
    assert r.status_code == 200

    print("\n ALL ENDPOINT TESTS PASSED SUCCESSFULLY! ")

if __name__ == "__main__":
    test_endpoints()
