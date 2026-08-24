# ============================================================
# HUONG DAN DAY DU: Benchmark NAFNet tren Kaggle
# ============================================================

"""
=================================================================
BUOC 0: TAO KAGGLE NOTEBOOK
=================================================================
1. Vao https://www.kaggle.com/code
2. Click "New Notebook" (or "New" -> "Notebook")
3. Chon GPU: Menu "..." -> "Change runtime type" -> GPU (P100 or T4 x2)
4. Dat ten: "NAFNet-Benchmark"

=================================================================
BUOC 1: CLONE REPO VA INSTALL
=================================================================
Copy CELL 1 vao cell dau tien:

```
!git clone https://github.com/megvii-research/NAFNet.git
%cd NAFNet
!pip install -r requirements.txt
!pip install ptflops thop
!python setup.py develop --no_cuda_ext
```

Ky luat:
- `--no_cuda_ext`: khong can compile CUDA extension (chi can cho training)
- `ptflops`: de do FLOPs/MACs

=================================================================
BUOC 2: DOWNLOAD PRETRAINED MODELS
=================================================================
Copy CELL 2 vao cell thu 2:

```
import os
os.makedirs("experiments/pretrained_models", exist_ok=True)

# GoPro width32
!gdown "https://drive.google.com/uc?id=1Fr2QadtDCEXg6iwWX8OzeZLbHOx2t5Bj" -O experiments/pretrained_models/NAFNet-GoPro-width32.pth

# GoPro width64
!gdown "https://drive.google.com/uc?id=1S0PVRbyTakYY9a82kujgZLbMihfNBLfC" -O experiments/pretrained_models/NAFNet-GoPro-width64.pth

# SIDD width32
!gdown "https://drive.google.com/uc?id=1lsByk21Xw-6aW7epCwOQxvm6HYCQZPHZ" -O experiments/pretrained_models/NAFNet-SIDD-width32.pth

# SIDD width64
!gdown "https://drive.google.com/uc?id=14Fht1QQJ2gMlk4N1ERCRuElg8JfjrWWR" -O experiments/pretrained_models/NAFNet-SIDD-width64.pth

!ls -lh experiments/pretrained_models/
```

Neu gdown loi (Google Drive block), dung cach 2:
```
# Cach 2: Download thu cong tu Google Drive links
# Mo tung link ben duoi trong browser, download, upload len Kaggle Dataset
# Roi copy vao folder experiments/pretrained_models/

# Hoac dung gdown voi id:
!pip install gdown
!gdown 1Fr2QadtDCEXg6iwWX8OzeZLbHOx2t5Bj -O experiments/pretrained_models/NAFNet-GoPro-width32.pth
!gdown 1S0PVRbyTakYY9a82kujgZLbMihfNBLfC -O experiments/pretrained_models/NAFNet-GoPro-width64.pth
!gdown 1lsByk21Xw-6aW7epCwOQxvm6HYCQZPHZ -O experiments/pretrained_models/NAFNet-SIDD-width32.pth
!gdown 14Fht1QQJ2gMlk4N1ERCRuElg8JfjrWWR -O experiments/pretrained_models/NAFNet-SIDD-width64.pth
```

=================================================================
BUOC 3: CHECK GPU
=================================================================
Copy CELL 3:

```
!nvidia-smi
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
```

Kiem tra:
- GPU name: T4, P100, V100, A100, etc.
- VRAM: >= 16GB la tot (A100 40GB ly tuong)

=================================================================
BUOC 4: CHAY BENCHMARK
=================================================================
Copy CA HET CELL 4 + CELL 5 vao (noi dung benchmark script).

Cell 4: Define functions + MODELS config
Cell 5: Chay benchmark

Ket qua se hien ra:
- Parameters (M): so luong tham so
- MACs (G): FLOPs tinh theo Multiply-Accum
- Latency (ms): thoi gian inference 1 anh
- FPS: frames per second
- VRAM (MB): bo nho GPU su dung

=================================================================
BUOC 5: XEM BANG KET QUA (DEP)
=================================================================
Copy CELL 6 de hien bang HTML dep tren Kaggle output.

=================================================================
PHIEN BAN GPU KHAC NHAU - KY LUAT
=================================================================
Kaggle chi cho P100 (16GB) hoac T4 (16GB).

Thoi gian du kien tren Kaggle P100 (batch=1, 256x256):
- NAFNet-w32: ~3-5 ms (200-300 FPS)
- NAFNet-w64: ~8-15 ms (65-125 FPS)
- NAFNet-SIDD-w32: ~3-5 ms
- NAFNet-SIDD-w64: ~10-18 ms

Thoi gian du kien tren Kaggle T4 (batch=1, 720x1280):
- NAFNet-w32: ~25-40 ms
- NAFNet-w64: ~60-100 ms

Luu y:
- FLOPs/Khong thay doi voi GPU
- Latency/VRAM thay doi tuy GPU
- Benchmark 100 lan la du on dinh

=================================================================
TRONG SOAT KET QUA voi BAI BAO
=================================================================
Bai bao (Table 6,7) tren NVIDIA 2080Ti:
- NAFNet-w64 (GoPro): MACs = 65G, PSNR = 33.71
- NAFNet-w32 (GoPro): MACs = 16G, PSNR = 32.87
- NAFNet-w64 (SIDD):  MACs = 65G, PSNR = 40.30
- NAFNet-w32 (SIDD):  MACs = 16G, PSNR = 39.97

Neu FLOPs cua ban gan 65G/16G la dung.
Latency tren 2080Ti (Table 3 trong paper):
- 36 blocks, 256x256: 39.1 ms
- 36 blocks, 720x1280: 177.1 ms

=================================================================
XUAT KET QUA
=================================================================
Sau khi chay xong, export Kaggle notebook:
1. Click "Save Version" -> "Save and Run All"
2. Khi xong, click "..." -> "Download" -> .ipynb
3. Hoac copy output bang thanh text

De xuat CSV:
```python
import pandas as pd
df = pd.DataFrame(results)
df.to_csv("nafnet_benchmark.csv", index=False)
print(df.to_markdown(index=False))
```
"""
