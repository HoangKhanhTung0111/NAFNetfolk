# ============================================================
# Kaggle NAFNet Benchmark - Copy-Paste Notebook
# ============================================================
# TAO BUC CELLS BEN DUOI VAO KAGGLE NOTEBOOK
# Moi "# --- CELL X ---" la mot cell rieng biet
# ============================================================


# --- CELL 1: Clone repo va install dependencies ---
# ============================================================
!git clone https://github.com/megvii-research/NAFNet.git
%cd NAFNet
!pip install -r requirements.txt
!pip install ptflops thop
!python setup.py develop --no_cuda_ext


# --- CELL 2: Download pretrained models ---
# ============================================================
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


# --- CELL 3: Check GPU info ---
# ============================================================
!nvidia-smi

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")


# --- CELL 4: Benchmark script (copy vao day) ---
# ============================================================
# === DANH SACH MODEL CAN DO ===
import os, sys, time, warnings
import torch
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())

from basicsr.models.archs.NAFNet_arch import NAFNet, NAFNetLocal

MODELS = {
    "NAFNet-GoPro-w32": {
        "arch": "NAFNetLocal",
        "width": 32,
        "enc_blk_nums": [1, 1, 1, 28],
        "middle_blk_num": 1,
        "dec_blk_nums": [1, 1, 1, 1],
        "ckpt": "experiments/pretrained_models/NAFNet-GoPro-width32.pth",
    },
    "NAFNet-GoPro-w64": {
        "arch": "NAFNetLocal",
        "width": 64,
        "enc_blk_nums": [1, 1, 1, 28],
        "middle_blk_num": 1,
        "dec_blk_nums": [1, 1, 1, 1],
        "ckpt": "experiments/pretrained_models/NAFNet-GoPro-width64.pth",
    },
    "NAFNet-SIDD-w32": {
        "arch": "NAFNet",
        "width": 32,
        "enc_blk_nums": [2, 2, 4, 8],
        "middle_blk_num": 12,
        "dec_blk_nums": [2, 2, 2, 2],
        "ckpt": "experiments/pretrained_models/NAFNet-SIDD-width32.pth",
    },
    "NAFNet-SIDD-w64": {
        "arch": "NAFNet",
        "width": 64,
        "enc_blk_nums": [2, 2, 4, 8],
        "middle_blk_num": 12,
        "dec_blk_nums": [2, 2, 2, 2],
        "ckpt": "experiments/pretrained_models/NAFNet-SIDD-width64.pth",
    },
}

INPUT_SIZES = {
    "256x256": (1, 3, 256, 256),
    "720x1280": (1, 3, 720, 1280),
}

WARMUP_ITERS = 10
BENCH_ITERS = 100


def build_model(cfg):
    if cfg["arch"] == "NAFNetLocal":
        return NAFNetLocal(
            img_channel=3, width=cfg["width"],
            middle_blk_num=cfg["middle_blk_num"],
            enc_blk_nums=cfg["enc_blk_nums"],
            dec_blk_nums=cfg["dec_blk_nums"],
        )
    else:
        return NAFNet(
            img_channel=3, width=cfg["width"],
            middle_blk_num=cfg["middle_blk_num"],
            enc_blk_nums=cfg["enc_blk_nums"],
            dec_blk_nums=cfg["dec_blk_nums"],
        )


def load_weights(model, ckpt_path):
    if not os.path.exists(ckpt_path):
        print(f"  [SKIP] Not found: {ckpt_path}")
        return model, False
    sd = torch.load(ckpt_path, map_location="cpu")
    if "params_ema" in sd: sd = sd["params_ema"]
    elif "params" in sd: sd = sd["params"]
    model.load_state_dict(sd, strict=True)
    return model, True


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    return total


def measure_flops(model, input_size):
    try:
        from ptflops import get_model_complexity_info
        macs, params = get_model_complexity_info(
            model, input_size[1:], verbose=False, print_per_layer_stat=False
        )
        macs_val = float(macs.replace(" GMACs", "").replace(" MACs", ""))
        if "M" in macs and "G" not in macs:
            macs_val /= 1000
        return macs_val
    except:
        return None


def measure_latency_vram(model, input_size, device="cuda"):
    model = model.to(device).eval()
    x = torch.randn(*input_size).to(device)

    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            _ = model(x)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(BENCH_ITERS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    vram = torch.cuda.max_memory_allocated(device) / 1024**2
    return np.mean(times), np.std(times), np.min(times), vram


# --- CELL 5: Chay benchmark ---
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

results = []

for name, cfg in MODELS.items():
    print(f"--- {name} ---")
    model = build_model(cfg)
    model, loaded = load_weights(model, cfg["ckpt"])
    if loaded:
        print(f"  Weights loaded.")
    model = model.to(device).eval()

    params = count_params(model)
    print(f"  Params: {params:,} ({params/1e6:.2f} M)")

    flops_256 = measure_flops(model, INPUT_SIZES["256x256"])
    flops_720 = measure_flops(model, INPUT_SIZES["720x1280"])
    if flops_256: print(f"  MACs (256x256):  {flops_256:.2f} G")
    if flops_720: print(f"  MACs (720x1280): {flops_720:.2f} G")

    if device == "cuda":
        torch.cuda.empty_cache()
        lat256, _, _, vram256 = measure_latency_vram(model, INPUT_SIZES["256x256"], device)
        torch.cuda.empty_cache()
        lat720, _, _, vram720 = measure_latency_vram(model, INPUT_SIZES["720x1280"], device)
        print(f"  Latency (256x256):  {lat256:.2f} ms ({1000/lat256:.1f} FPS) | VRAM: {vram256:.0f} MB")
        print(f"  Latency (720x1280): {lat720:.2f} ms ({1000/lat720:.1f} FPS) | VRAM: {vram720:.0f} MB")
    else:
        lat256 = lat720 = vram256 = vram720 = None

    results.append({
        "name": name, "params_M": params/1e6,
        "flops_256": flops_256, "flops_720": flops_720,
        "lat_256": lat256, "lat_720": lat720,
        "vram_256": vram256, "vram_720": vram720,
    })

    del model
    torch.cuda.empty_cache()
    print()


# --- CELL 6: In bang ket qua ---
# ============================================================
from IPython.display import HTML, display

html = """
<style>
  .bench-table { border-collapse: collapse; font-family: monospace; font-size: 13px; }
  .bench-table th { background: #2d2d2d; color: #0f0; padding: 8px 12px; border: 1px solid #555; }
  .bench-table td { padding: 6px 12px; border: 1px solid #444; text-align: right; }
  .bench-table tr:nth-child(even) { background: #1a1a2e; }
  .bench-table tr:nth-child(odd) { background: #16213e; }
  .bench-table td:first-child { text-align: left; color: #0ff; font-weight: bold; }
</style>
<table class="bench-table">
<tr>
  <th>Model</th><th>Params (M)</th>
  <th>MACs 256 (G)</th><th>MACs 720 (G)</th>
  <th>Lat 256 (ms)</th><th>Lat 720 (ms)</th>
  <th>FPS 720</th><th>VRAM 720 (MB)</th>
</tr>
"""

for r in results:
    html += f"<tr><td>{r['name']}</td><td>{r['params_M']:.2f}</td>"
    html += f"<td>{r['flops_256']:.2f}</td>" if r['flops_256'] else "<td>N/A</td>"
    html += f"<td>{r['flops_720']:.2f}</td>" if r['flops_720'] else "<td>N/A</td>"
    if r['lat_256']:
        html += f"<td>{r['lat_256']:.2f}</td><td>{r['lat_720']:.2f}</td>"
        html += f"<td>{1000/r['lat_720']:.1f}</td><td>{r['vram_720']:.0f}</td>"
    else:
        html += "<td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td>"
    html += "</tr>"

html += "</table>"
display(HTML(html))

# Also print plain text
print()
print(f"{'Model':<22} | {'Params(M)':>10} | {'MACs-256(G)':>11} | {'MACs-720(G)':>11} | {'Lat-256(ms)':>11} | {'Lat-720(ms)':>11} | {'FPS-720':>8} | {'VRAM-720(MB)':>12}")
print("-" * 130)
for r in results:
    row = f"{r['name']:<22} | {r['params_M']:>10.2f}"
    row += f" | {r['flops_256']:>11.2f}" if r['flops_256'] else f" | {'N/A':>11}"
    row += f" | {r['flops_720']:>11.2f}" if r['flops_720'] else f" | {'N/A':>11}"
    if r['lat_256']:
        row += f" | {r['lat_256']:>11.2f} | {r['lat_720']:>11.2f} | {1000/r['lat_720']:>8.1f} | {r['vram_720']:>12.0f}"
    else:
        row += f" | {'N/A':>11} | {'N/A':>11} | {'N/A':>8} | {'N/A':>12}"
    print(row)
