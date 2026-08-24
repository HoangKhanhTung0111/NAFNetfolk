"""
NAFNet Benchmark Script
Measure: Parameters, FLOPs (MACs), Latency, VRAM usage
"""
import os
import sys
import time
import torch
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIG - Chon model can benchmark
# ============================================================
MODELS = {
    # --- GoPro (deblur) ---
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
    # --- SIDD (denoise) ---
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

# Kich thuoc anh test
INPUT_SIZES = {
    "256x256": (1, 3, 256, 256),
    "720x1280": (1, 3, 720, 1280),  # anh GoPro full size
}

# So lan chay de tinh latency (bo 5 lan warmup)
WARMUP_ITERS = 10
BENCH_ITERS = 100


# ============================================================
# 2. IMPORT MODEL
# ============================================================
sys.path.insert(0, os.getcwd())
from basicsr.models.archs.NAFNet_arch import NAFNet, NAFNetLocal


def build_model(cfg):
    """Xay dung model tu config."""
    if cfg["arch"] == "NAFNetLocal":
        model = NAFNetLocal(
            img_channel=3,
            width=cfg["width"],
            middle_blk_num=cfg["middle_blk_num"],
            enc_blk_nums=cfg["enc_blk_nums"],
            dec_blk_nums=cfg["dec_blk_nums"],
        )
    else:
        model = NAFNet(
            img_channel=3,
            width=cfg["width"],
            middle_blk_num=cfg["middle_blk_num"],
            enc_blk_nums=cfg["enc_blk_nums"],
            dec_blk_nums=cfg["dec_blk_nums"],
        )
    return model


def load_weights(model, ckpt_path):
    """Load pretrained weights."""
    if not os.path.exists(ckpt_path):
        print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
        return model, False
    state_dict = torch.load(ckpt_path, map_location="cpu")
    if "params_ema" in state_dict:
        state_dict = state_dict["params_ema"]
    elif "params" in state_dict:
        state_dict = state_dict["params"]
    model.load_state_dict(state_dict, strict=True)
    return model, True


# ============================================================
# 3. MEASURE FUNCTIONS
# ============================================================
def count_parameters(model):
    """Dem so luong parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_flops(model, input_size):
    """Do FLOPs (MACs) bang ptflops."""
    try:
        from ptflops import get_model_complexity_info

        macs, params = get_model_complexity_info(
            model,
            input_size[1:],  # bo batch dim
            verbose=False,
            print_per_layer_stat=False,
        )
        # macs: "XX.XX GMACs", params: "XX.XX M"
        macs_val = float(macs.replace(" GMACs", "").replace(" MACs", ""))
        if "G" in macs:
            macs_val = macs_val
        elif "M" in macs:
            macs_val = macs_val / 1000
        return macs_val
    except ImportError:
        print("  [WARN] ptflops not installed. Install: pip install ptflops")
        return None


def measure_latency_vram(model, input_size, device="cuda", warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    """Do latency (ms) va VRAM (MB)."""
    model = model.to(device)
    model.eval()
    x = torch.randn(*input_size).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

    torch.cuda.synchronize()

    # Reset VRAM tracking
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

    # Benchmark latency
    times = []
    with torch.no_grad():
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

    # VRAM
    vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    avg_latency = np.mean(times)
    std_latency = np.std(times)
    min_latency = np.min(times)
    max_latency = np.max(times)

    return avg_latency, std_latency, min_latency, max_latency, vram_mb


# ============================================================
# 4. MAIN
# ============================================================
def main():
    print("=" * 80)
    print("NAFNet BENCHMARK")
    print("=" * 80)

    # Check GPU
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        device = "cpu"
        print("WARNING: No GPU found. Latency/VRAM will be CPU-only.")

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    print()

    results = []

    for name, cfg in MODELS.items():
        print("-" * 60)
        print(f"[{name}]")
        print(f"  arch={cfg['arch']}, width={cfg['width']}")
        print(f"  enc={cfg['enc_blk_nums']}, mid={cfg['middle_blk_num']}, dec={cfg['dec_blk_nums']}")

        # Build
        model = build_model(cfg)

        # Load weights
        model, loaded = load_weights(model, cfg["ckpt"])
        if loaded:
            print(f"  Loaded weights from: {cfg['ckpt']}")
        model = model.to(device)
        model.eval()

        # Params
        total_params, trainable_params = count_parameters(model)
        print(f"  Parameters: {total_params:,} ({total_params/1e6:.2f} M)")
        print(f"  Trainable:  {trainable_params:,} ({trainable_params/1e6:.2f} M)")

        # FLOPs cho tung input size
        flops_results = {}
        for size_name, size in INPUT_SIZES.items():
            flops = measure_flops(model, size)
            if flops is not None:
                flops_results[size_name] = flops
                print(f"  MACs ({size_name}): {flops:.2f} G")

        # Latency & VRAM cho tung input size
        latency_results = {}
        for size_name, size in INPUT_SIZES.items():
            if device == "cuda":
                # Clear cache
                torch.cuda.empty_cache()
                avg, std, mn, mx, vram = measure_latency_vram(model, size, device)
                latency_results[size_name] = {
                    "avg_ms": avg,
                    "std_ms": std,
                    "min_ms": mn,
                    "max_ms": mx,
                    "vram_mb": vram,
                }
                fps = 1000.0 / avg
                print(f"  Latency ({size_name}): {avg:.2f} +/- {std:.2f} ms ({fps:.1f} FPS)")
                print(f"  VRAM   ({size_name}): {vram:.1f} MB")
            else:
                latency_results[size_name] = None

        results.append({
            "name": name,
            "params_M": total_params / 1e6,
            "flops": flops_results,
            "latency": latency_results,
        })

        del model
        torch.cuda.empty_cache()

    # ============================================================
    # 5. PRINT SUMMARY TABLE
    # ============================================================
    print()
    print("=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)

    header = f"{'Model':<22} | {'Params(M)':>10} | {'MACs-256(G)':>11} | {'MACs-720(G)':>11}"
    if device == "cuda":
        header += f" | {'Lat-256(ms)':>11} | {'Lat-720(ms)':>11} | {'FPS-720':>8} | {'VRAM-720(MB)':>12}"
    print(header)
    print("-" * len(header))

    for r in results:
        row = f"{r['name']:<22} | {r['params_M']:>10.2f}"

        # FLOPs
        f256 = r["flops"].get("256x256", None)
        f720 = r["flops"].get("720x1280", None)
        row += f" | {f256:>11.2f}" if f256 else f" | {'N/A':>11}"
        row += f" | {f720:>11.2f}" if f720 else f" | {'N/A':>11}"

        # Latency & VRAM
        if device == "cuda":
            l256 = r["latency"].get("256x256", None)
            l720 = r["latency"].get("720x1280", None)
            if l256:
                row += f" | {l256['avg_ms']:>11.2f}"
            else:
                row += f" | {'N/A':>11}"
            if l720:
                row += f" | {l720['avg_ms']:>11.2f}"
                row += f" | {1000/l720['avg_ms']:>8.1f}"
                row += f" | {l720['vram_mb']:>12.1f}"
            else:
                row += f" | {'N/A':>11} | {'N/A':>8} | {'N/A':>12}"

        print(row)

    print("=" * 100)
    print()
    print("Done!")


if __name__ == "__main__":
    main()
