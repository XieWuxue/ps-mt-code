import subprocess
import sys

# 安装依赖包
deps = [
    'opencv-python==4.5.1.48',
    'numpy==1.19.5',
    'matplotlib==3.4.2',
    'tqdm==4.59.0',
    'wandb>=0.12.14'
]

print("Installing dependencies...")
for dep in deps:
    print(f"Installing {dep}...")
    result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-i', 'https://pypi.tuna.tsinghua.edu.cn/simple', dep], 
                          capture_output=True, text=True)
    print(f"Exit code: {result.returncode}")
    if result.stdout:
        print(f"Output: {result.stdout[:500]}...")
    if result.stderr:
        print(f"Error: {result.stderr[:500]}...")

print("Dependency installation complete!")

# 验证依赖
print("\nVerifying dependencies...")
try:
    import cv2
    print("✓ cv2 imported successfully")
except ImportError:
    print("✗ cv2 import failed")

try:
    import numpy
    print("✓ numpy imported successfully")
except ImportError:
    print("✗ numpy import failed")

try:
    import matplotlib
    print("✓ matplotlib imported successfully")
except ImportError:
    print("✗ matplotlib import failed")

try:
    import tqdm
    print("✓ tqdm imported successfully")
except ImportError:
    print("✗ tqdm import failed")

try:
    import wandb
    print("✓ wandb imported successfully")
except ImportError:
    print("✗ wandb import failed")
