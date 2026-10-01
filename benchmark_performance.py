import os
import sys

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import time
import random
import glob

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import torch
print("=" * 60)
print("Checking PyTorch & CUDA status:")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device:      {torch.cuda.get_device_name(0)}")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Total VRAM:      {vram_gb:.2f} GB")
print("=" * 60)

# Import the ALPR pipeline
from core.pipeline import EgyptianALPR
import cv2

# Pick 10 random images from dataset
all_images = glob.glob(os.path.join(BASE_DIR, "data", "**", "*.jpg"), recursive=True)
if not all_images:
    all_images = glob.glob(os.path.join(BASE_DIR, "synthetic_samples", "**", "*.jpg"), recursive=True)

if not all_images:
    print("No test images found!")
    sys.exit(1)

random.seed(42)
test_images = random.sample(all_images, min(10, len(all_images)))
print(f"Selected {len(test_images)} test images for performance comparison:")
for i, p in enumerate(test_images, 1):
    print(f"  {i}. {os.path.basename(p)}")

print("\n" + "=" * 60)
print("1. Benchmarking on CPU...")
print("=" * 60)

alpr_cpu = EgyptianALPR()
alpr_cpu.plate_model.to('cpu')
alpr_cpu.recog_model.to('cpu')
alpr_cpu.loc_model.to('cpu')

# Warmup CPU
dummy = cv2.imread(test_images[0])
if dummy is not None:
    _ = alpr_cpu.recognize_plate_crop(dummy)

cpu_times = []
results_text = []

for idx, img_path in enumerate(test_images, 1):
    img = cv2.imread(img_path)
    if img is None:
        continue
    t0 = time.perf_counter()
    res = alpr_cpu.recognize_plate_crop(img)
    t1 = time.perf_counter()
    elapsed_ms = (t1 - t0) * 1000
    cpu_times.append(elapsed_ms)
    plate_txt = f"{res.get('text', '')}"
    results_text.append(plate_txt)
    print(f"  [{idx:02d}/10] CPU: {elapsed_ms:6.1f} ms | Text: {plate_txt}")

avg_cpu = sum(cpu_times) / len(cpu_times) if cpu_times else 0
print(f"\n>> CPU Average Latency: {avg_cpu:.1f} ms per image (~{1000/avg_cpu:.1f} FPS)")

if torch.cuda.is_available():
    print("\n" + "=" * 60)
    print("2. Benchmarking on GPU (NVIDIA CUDA)...")
    print("=" * 60)
    
    alpr_gpu = EgyptianALPR()
    alpr_gpu.plate_model.to('cuda:0')
    alpr_gpu.recog_model.to('cuda:0')
    alpr_gpu.loc_model.to('cuda:0')
    
    # Warmup GPU
    _ = alpr_gpu.recognize_plate_crop(dummy)
    torch.cuda.synchronize()
    
    gpu_times = []
    for idx, img_path in enumerate(test_images, 1):
        img = cv2.imread(img_path)
        if img is None:
            continue
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = alpr_gpu.recognize_plate_crop(img)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000
        gpu_times.append(elapsed_ms)
        print(f"  [{idx:02d}/10] GPU: {elapsed_ms:6.1f} ms | Text: {res.get('text', '')}")

    avg_gpu = sum(gpu_times) / len(gpu_times) if gpu_times else 1
    speedup = avg_cpu / avg_gpu if avg_gpu > 0 else 0
    
    print("\n" + "=" * 60)
    print("FINAL PERFORMANCE COMPARISON (10 IMAGES):")
    print("=" * 60)
    print(f"{'Metric':<25} | {'CPU (Before)':<15} | {'GPU (After)':<15}")
    print("-" * 60)
    print(f"{'Average Latency':<25} | {avg_cpu:7.1f} ms     | {avg_gpu:7.1f} ms")
    print(f"{'Throughput (FPS)':<25} | {1000/avg_cpu:7.1f} FPS    | {1000/avg_gpu:7.1f} FPS")
    print(f"{'Total Time (10 imgs)':<25} | {sum(cpu_times):7.1f} ms     | {sum(gpu_times):7.1f} ms")
    print("-" * 60)
    print(f">> GPU SPEEDUP FACTOR: {speedup:.2f}x FASTER!")
    print("=" * 60)
