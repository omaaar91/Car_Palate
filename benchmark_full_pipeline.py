import os
import sys

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import time
import random
import glob
import cv2
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.pipeline import EgyptianALPR

print("=" * 65)
print(f"Device: {torch.cuda.get_device_name(0)} | CUDA: {torch.cuda.is_available()}")
print("=" * 65)

# Find 10 full vehicle images
vehicle_images = glob.glob(os.path.join(BASE_DIR, "data", "**", "vehicle_*.jpg"), recursive=True)
if not vehicle_images:
    vehicle_images = glob.glob(os.path.join(BASE_DIR, "data", "**", "*.jpg"), recursive=True)

random.seed(42)
selected_imgs = random.sample(vehicle_images, min(10, len(vehicle_images)))

print(f"Running Full End-to-End Pipeline on {len(selected_imgs)} Vehicle Images:")
for i, p in enumerate(selected_imgs, 1):
    print(f"  {i}. {os.path.basename(p)}")

# 1. CPU Pipeline
print("\n" + "-" * 65)
print("1. Running on CPU...")
print("-" * 65)
alpr_cpu = EgyptianALPR()
alpr_cpu.plate_model.to('cpu')
alpr_cpu.recog_model.to('cpu')
alpr_cpu.loc_model.to('cpu')

# Warmup
dummy = cv2.imread(selected_imgs[0])
if dummy is not None:
    _ = alpr_cpu.process_image(dummy)

cpu_times = []
for idx, path in enumerate(selected_imgs, 1):
    img = cv2.imread(path)
    if img is None:
        continue
    t0 = time.perf_counter()
    res = alpr_cpu.process_image(img)
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000
    cpu_times.append(ms)
    plates = [p.get('text', '') for p in res.get('plates', [])]
    print(f"  [{idx:02d}/10] CPU: {ms:6.1f} ms | Plates Detected: {plates or 'No plate'}")

avg_cpu = sum(cpu_times) / len(cpu_times)

# 2. GPU Pipeline
print("\n" + "-" * 65)
print("2. Running on GPU (NVIDIA CUDA)...")
print("-" * 65)
alpr_gpu = EgyptianALPR()
alpr_gpu.plate_model.to('cuda:0')
alpr_gpu.recog_model.to('cuda:0')
alpr_gpu.loc_model.to('cuda:0')

# Warmup
if dummy is not None:
    _ = alpr_gpu.process_image(dummy)
    torch.cuda.synchronize()

gpu_times = []
for idx, path in enumerate(selected_imgs, 1):
    img = cv2.imread(path)
    if img is None:
        continue
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    res = alpr_gpu.process_image(img)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000
    gpu_times.append(ms)
    plates = [p.get('text', '') for p in res.get('plates', [])]
    print(f"  [{idx:02d}/10] GPU: {ms:6.1f} ms | Plates Detected: {plates or 'No plate'}")

avg_gpu = sum(gpu_times) / len(gpu_times)
speedup = avg_cpu / avg_gpu

print("\n" + "=" * 65)
print("FINAL BENCHMARK REPORT (Full Vehicle Pipeline - 10 Images):")
print("=" * 65)
print(f"{'Metric':<28} | {'CPU (Before)':<15} | {'GPU CUDA (After)':<15}")
print("-" * 65)
print(f"{'Average Latency per Frame':<28} | {avg_cpu:7.1f} ms     | {avg_gpu:7.1f} ms")
print(f"{'Throughput (FPS)':<28} | {1000/avg_cpu:7.1f} FPS    | {1000/avg_gpu:7.1f} FPS")
print(f"{'Total Time (10 frames)':<28} | {sum(cpu_times):7.1f} ms     | {sum(gpu_times):7.1f} ms")
print("-" * 65)
print(f">> SPEEDUP: {speedup:.2f}x FASTER WITH CUDA ON QUADRO T1000!")
print("=" * 65)
