import os
import sys
import time
import subprocess

TARGET_WHEEL = r"E:\pip_temp\torch-2.6.0+cu124-cp311-cp311-win_amd64.whl"
TORCHVISION_WHEEL = r"E:\pip_temp\torchvision-0.21.0+cu124-cp311-cp311-win_amd64.whl"
EXPECTED_SIZE = 2532350702
VENV_PIP = r"E:\car_palate_venv\Scripts\pip.exe"
VENV_PYTHON = r"E:\car_palate_venv\Scripts\python.exe"

print("⏳ Waiting for PyTorch CUDA wheel download to finish...")
last_mb = 0
while True:
    if os.path.exists(TARGET_WHEEL):
        sz = os.path.getsize(TARGET_WHEEL)
        mb = sz / (1024 * 1024)
        pct = (sz / EXPECTED_SIZE) * 100
        if int(mb) != int(last_mb) and int(mb) % 100 == 0:
            print(f"📥 Progress: {mb:.1f} MB / 2415 MB ({pct:.1f}%)")
            last_mb = mb
        if sz >= EXPECTED_SIZE:
            print(f"✅ Download 100% complete! ({sz} bytes)")
            break
    time.sleep(3)

print("\n🚀 Installing PyTorch + CUDA 12.4 + Torchvision into E:\\car_palate_venv ...")
cmd = [VENV_PIP, "install", TARGET_WHEEL, TORCHVISION_WHEEL, "--no-deps"]
res = subprocess.run(cmd, capture_output=True, text=True)
print(res.stdout)
if res.stderr:
    print(res.stderr)

if res.returncode != 0:
    print("❌ Installation failed!")
    sys.exit(1)

print("✅ Installation complete! Now running benchmark on 10 random images...")
bench_cmd = [VENV_PYTHON, r"E:\tik tok\Car_Palate\benchmark_performance.py"]
bench_res = subprocess.run(bench_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")

print(bench_res.stdout)
if bench_res.stderr:
    print(bench_res.stderr)

with open(r"E:\tik tok\Car_Palate\benchmark_results.txt", "w", encoding="utf-8") as f:
    f.write(bench_res.stdout)
    if bench_res.stderr:
        f.write("\nSTDERR:\n" + bench_res.stderr)

print("🏁 Finished benchmark! Results saved to benchmark_results.txt")
