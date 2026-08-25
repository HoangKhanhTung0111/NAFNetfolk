"""
NAFNet Benchmark + Evaluation Script
Usage:
  python benchmark_and_eval.py --mode all          # benchmark + evaluate
  python benchmark_and_eval.py --mode benchmark     # chi do params/flops/latency/vram
  python benchmark_and_eval.py --mode eval          # chi evaluate PSNR/SSIM
"""
import os
import sys
import re
import time
import gc
import argparse
import warnings
import numpy as np
import torch
import yaml

warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())

from basicsr.models.archs.NAFNet_arch import NAFNet, NAFNetLocal

# ============================================================
# CONFIGS
# ============================================================
MODEL_CONFIGS = {
    "NAFNet-GoPro-w32": {
        "cls": "NAFNetLocal", "width": 32,
        "enc": [1, 1, 1, 28], "mid": 1, "dec": [1, 1, 1, 1],
        "ckpt": "experiments/pretrained_models/NAFNet-GoPro-width32.pth",
        "task": "deblur", "dataset": "GoPro",
        "yaml_test": "options/test/GoPro/NAFNet-width32.yml",
        "paper_psnr": 32.87, "paper_ssim": 0.9606,
    },
    "NAFNet-GoPro-w64": {
        "cls": "NAFNetLocal", "width": 64,
        "enc": [1, 1, 1, 28], "mid": 1, "dec": [1, 1, 1, 1],
        "ckpt": "experiments/pretrained_models/NAFNet-GoPro-width64.pth",
        "task": "deblur", "dataset": "GoPro",
        "yaml_test": "options/test/GoPro/NAFNet-width64.yml",
        "paper_psnr": 33.69, "paper_ssim": 0.9668,
    },
    "NAFNet-SIDD-w32": {
        "cls": "NAFNet", "width": 32,
        "enc": [2, 2, 4, 8], "mid": 12, "dec": [2, 2, 2, 2],
        "ckpt": "experiments/pretrained_models/NAFNet-SIDD-width32.pth",
        "task": "denoise", "dataset": "SIDD",
        "yaml_test": "options/test/SIDD/NAFNet-width32.yml",
        "paper_psnr": 39.97, "paper_ssim": 0.9599,
    },
    "NAFNet-SIDD-w64": {
        "cls": "NAFNet", "width": 64,
        "enc": [2, 2, 4, 8], "mid": 12, "dec": [2, 2, 2, 2],
        "ckpt": "experiments/pretrained_models/NAFNet-SIDD-width64.pth",
        "task": "denoise", "dataset": "SIDD",
        "yaml_test": "options/test/SIDD/NAFNet-width64.yml",
        "paper_psnr": 40.30, "paper_ssim": 0.9614,
    },
}


def build_model(cfg):
    cls = NAFNetLocal if cfg["cls"] == "NAFNetLocal" else NAFNet
    return cls(img_channel=3, width=cfg["width"],
               middle_blk_num=cfg["mid"],
               enc_blk_nums=cfg["enc"], dec_blk_nums=cfg["dec"])


def load_weights(model, path):
    if not os.path.exists(path):
        return model, False
    sd = torch.load(path, map_location="cpu")
    sd = sd.get("params_ema", sd.get("params", sd))
    model.load_state_dict(sd, strict=True)
    return model, True


def compute_macs_analytical(width, enc_blks, mid_blk, dec_blks, h, w):
    """
    Tinh FLOPs (MACs) thu cong theo cong thuc trong paper.
    Moi NAFBlock tai spatial HxW voi channel c: ~6*H*W*c MACs
    (xem paper Appendix A.1)
    """
    total_macs = 0
    c = width
    spatial_areas = []  # (h, w, c) tai moi level

    # Encoder
    for num in enc_blks:
        macs_per_block = 6 * h * w * c * c  # 6*H*W*c^2
        total_macs += macs_per_block * num
        spatial_areas.append((h, w, c))
        h, w = h // 2, w // 2
        c = c * 2

    # Middle
    macs_per_block = 6 * h * w * c * c
    total_macs += macs_per_block * mid_blk
    spatial_areas.append((h, w, c))

    # Decoder
    for num in dec_blks:
        h, w = h * 2, w * 2
        c = c // 2
        macs_per_block = 6 * h * w * c * c
        total_macs += macs_per_block * num

    # intro conv: 3x3, 3->width
    total_macs += h * w * 3 * width * 9  # 3x3 conv
    # ending conv: 3x3, width->3
    total_macs += h * w * width * 3 * 9

    return total_macs / 1e9  # return in GMACs


# ============================================================
# PART 1: BENCHMARK (Params, FLOPs, Latency, VRAM)
# ============================================================
def run_benchmark():
    print("=" * 90)
    print("  BENCHMARK: Parameters, FLOPs, Latency, VRAM")
    print("=" * 90)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    print(f"PyTorch: {torch.__version__}\n")

    INPUT_SIZES = [(1, 3, 256, 256), (1, 3, 720, 1280)]
    WARMUP, ITERS = 10, 100
    results = []

    for name, cfg in MODEL_CONFIGS.items():
        print(f"--- {name} ---")
        model = build_model(cfg)
        model, ok = load_weights(model, cfg["ckpt"])
        print(f"  Weights: {'loaded' if ok else 'random (no ckpt)'}")
        model.eval()

        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  Params: {params:.2f} M")

        # FLOPs: tinh thu cong, khong dung ptflops
        flops = {}
        for sz in INPUT_SIZES:
            _, _, h, w = sz
            macs = compute_macs_analytical(cfg["width"], cfg["enc"], cfg["mid"], cfg["dec"], h, w)
            flops[sz] = macs
            print(f"  MACs {sz[2]}x{sz[3]}: {macs:.2f} G")

        # Latency + VRAM: on GPU
        lat = {}
        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            model.to(device)

            # Measure 256x256 first (always fits)
            try:
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.reset_peak_memory_stats()
                x = torch.randn(1, 3, 256, 256, device=device)
                with torch.no_grad():
                    for _ in range(WARMUP): model(x)
                torch.cuda.synchronize()
                times = []
                with torch.no_grad():
                    for _ in range(ITERS):
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        model(x)
                        torch.cuda.synchronize()
                        times.append((time.perf_counter()-t0)*1000)
                vram = torch.cuda.max_memory_allocated()/1024**2
                avg = np.mean(times)
                lat[(1,3,256,256)] = {"ms": avg, "fps": 1000/avg, "vram_mb": vram}
                print(f"  Lat 256x256: {avg:.2f} ms ({1000/avg:.1f} FPS) | VRAM: {vram:.0f} MB")
                del x
            except Exception as e:
                lat[(1,3,256,256)] = None
                print(f"  Lat 256x256: error ({e})")

            # Measure 720x1280 only if enough VRAM (>4GB free)
            free_mem = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3
            if free_mem > 4.0:
                try:
                    torch.cuda.empty_cache()
                    gc.collect()
                    torch.cuda.reset_peak_memory_stats()
                    x = torch.randn(1, 3, 720, 1280, device=device)
                    with torch.no_grad():
                        for _ in range(WARMUP): model(x)
                    torch.cuda.synchronize()
                    times = []
                    with torch.no_grad():
                        for _ in range(ITERS):
                            torch.cuda.synchronize()
                            t0 = time.perf_counter()
                            model(x)
                            torch.cuda.synchronize()
                            times.append((time.perf_counter()-t0)*1000)
                    vram = torch.cuda.max_memory_allocated()/1024**2
                    avg = np.mean(times)
                    lat[(1,3,720,1280)] = {"ms": avg, "fps": 1000/avg, "vram_mb": vram}
                    print(f"  Lat 720x1280: {avg:.2f} ms ({1000/avg:.1f} FPS) | VRAM: {vram:.0f} MB")
                    del x
                except torch.cuda.OutOfMemoryError:
                    lat[(1,3,720,1280)] = None
                    print(f"  Lat 720x1280: OOM (skipped)")
                    torch.cuda.empty_cache()
                    gc.collect()
                except Exception as e:
                    lat[(1,3,720,1280)] = None
                    print(f"  Lat 720x1280: error ({e})")
            else:
                lat[(1,3,720,1280)] = None
                print(f"  Lat 720x1280: skipped (only {free_mem:.1f}GB free)")

            model.cpu()
            torch.cuda.empty_cache()
            gc.collect()
        else:
            for sz in INPUT_SIZES:
                lat[sz] = None

        results.append({"name": name, "params": params, "flops": flops, "lat": lat})
        del model; torch.cuda.empty_cache(); gc.collect()
        print()

    # Summary table
    print("=" * 120)
    print(f"{'Model':<22}|{'Task':<8}|{'Params(M)':>10}|{'MACs-256(G)':>11}|{'MACs-720(G)':>11}|{'Lat-256(ms)':>11}|{'Lat-720(ms)':>11}|{'FPS-720':>8}|{'VRAM-720(MB)':>12}")
    print("-" * 120)
    for i, r in enumerate(results):
        cfg = list(MODEL_CONFIGS.values())[i]
        l256 = r["lat"][(1,3,256,256)]
        l720 = r["lat"][(1,3,720,1280)]
        f256 = r["flops"].get((1,3,256,256), 0)
        f720 = r["flops"].get((1,3,720,1280), 0)
        row = (f"{r['name']:<22}|{cfg['task']:<8}"
               f"|{r['params']:>10.2f}"
               f"|{f256:>11.2f}|{f720:>11.2f}")
        if l256:
            row += f"|{l256['ms']:>11.2f}|{l720['ms']:>11.2f}|{l720['fps']:>8.1f}|{l720['vram_mb']:>12.0f}"
        else:
            row += f"|{'N/A':>11}|{'N/A':>11}|{'N/A':>8}|{'N/A':>12}"
        print(row)
    print("=" * 120)
    return results


# ============================================================
# PART 2: EVALUATE (PSNR/SSIM on test set)
# ============================================================
def run_eval():
    print("\n" + "=" * 90)
    print("  EVALUATE: PSNR / SSIM on test sets")
    print("=" * 90)

    from basicsr.data import create_dataloader, create_dataset
    from basicsr.models import create_model
    from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim
    from basicsr.utils import tensor2img, get_root_logger, get_time_str
    import tempfile
    from tqdm import tqdm

    eval_results = []

    for name, cfg in MODEL_CONFIGS.items():
        print(f"\n--- {name} ---")

        with open(cfg["yaml_test"]) as f:
            opt_raw = yaml.safe_load(f)

        opt_raw["path"]["pretrain_network_g"] = cfg["ckpt"]
        opt_raw["val"]["save_img"] = False

        out_dir = f"results/{name}"
        os.makedirs(out_dir, exist_ok=True)
        opt_raw["path"]["log"] = out_dir
        opt_raw["path"]["visualization"] = out_dir
        opt_raw["path"]["models"] = out_dir

        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False, dir=".")
        yaml.dump(opt_raw, tmp)
        tmp.close()

        from basicsr.utils.options import parse
        opt = parse(tmp.name, is_train=False)
        opt["path"]["pretrain_network_g"] = cfg["ckpt"]
        opt["val"]["save_img"] = False
        opt["dist"] = False
        opt["rank"] = 0
        opt["world_size"] = 1

        try:
            torch.backends.cudnn.benchmark = True
            from basicsr.utils import get_env_info

            # Create dataloader
            for phase, dataset_opt in sorted(opt["datasets"].items()):
                test_set = create_dataset(dataset_opt)
                max_imgs = min(len(test_set), 50)
                class LimitedDataset:
                    def __init__(self, ds, n):
                        self.ds = ds
                        self.n = n
                        self.opt = ds.opt
                    def __len__(self):
                        return self.n
                    def __getitem__(self, i):
                        return self.ds[i]
                test_set_limited = LimitedDataset(test_set, max_imgs)
                test_loader = create_dataloader(test_set_limited, dataset_opt, num_gpu=opt["num_gpu"],
                                                dist=opt["dist"], sampler=None, seed=opt["manual_seed"])

            # Create model
            model = create_model(opt)

            # Custom validation loop - no distributed
            print(f"  Evaluating {max_imgs} images...")
            psnr_total = 0.0
            ssim_total = 0.0
            count = 0

            model.net_g.eval()
            with torch.no_grad():
                for data in tqdm(test_loader, desc=f"  {name}", ncols=80):
                    model.feed_data(data, is_val=True)
                    model.test()
                    visuals = model.get_current_visuals()

                    sr_img = tensor2img([visuals['result']], rgb2bgr=True)
                    gt_img = tensor2img([visuals['gt']], rgb2bgr=True)

                    psnr_total += calculate_psnr(sr_img, gt_img, crop_border=0, test_y_channel=False)
                    ssim_total += calculate_ssim(sr_img, gt_img, crop_border=0, test_y_channel=False)
                    count += 1

                    del model.lq, model.output
                    if hasattr(model, 'gt'):
                        del model.gt
                    torch.cuda.empty_cache()

            psnr_avg = psnr_total / count if count > 0 else 0
            ssim_avg = ssim_total / count if count > 0 else 0

            print(f"  PSNR: {psnr_avg:.4f} dB")
            print(f"  SSIM: {ssim_avg:.4f}")
            print(f"  Paper: {cfg['paper_psnr']} dB / {cfg['paper_ssim']}")

            diff = psnr_avg - cfg["paper_psnr"]
            tag = "OK" if abs(diff) < 0.5 else "CHECK"
            print(f"  Delta: {diff:+.4f} dB [{tag}]")

            eval_results.append({"name": name, "psnr": psnr_avg, "ssim": ssim_avg})

        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback; traceback.print_exc()
            eval_results.append({"name": name, "psnr": None, "ssim": None})

        finally:
            os.unlink(tmp.name)
            if 'model' in dir():
                del model
            torch.cuda.empty_cache()
            gc.collect()

    return eval_results


# ============================================================
# PART 3: COMBINED SUMMARY
# ============================================================
def print_final_summary(bench_results, eval_results):
    print("\n\n")
    print("=" * 130)
    print("  FINAL SUMMARY TABLE")
    print("=" * 130)
    print(f"{'Model':<22}|{'Task':<8}|{'Params(M)':>10}|{'MACs(G)':>9}|{'Lat-720(ms)':>11}|{'FPS':>7}|{'VRAM(MB)':>9}|{'PSNR':>8}|{'SSIM':>8}|{'Paper':>8}")
    print("-" * 130)

    for i, (b, e) in enumerate(zip(bench_results, eval_results)):
        cfg = list(MODEL_CONFIGS.values())[i]
        l720 = b["lat"][(1,3,720,1280)]
        psnr_str = f"{e['psnr']:.2f}" if e["psnr"] else "N/A"
        ssim_str = f"{e['ssim']:.4f}" if e["ssim"] else "N/A"

        f720 = b["flops"].get((1,3,720,1280), 0)
        row = (f"{b['name']:<22}|{cfg['task']:<8}"
               f"|{b['params']:>10.2f}"
               f"|{f720:>9.2f}")
        if l720:
            row += f"|{l720['ms']:>11.2f}|{l720['fps']:>7.1f}|{l720['vram_mb']:>9.0f}"
        else:
            row += f"|{'N/A':>11}|{'N/A':>7}|{'N/A':>9}"
        row += f"|{psnr_str:>8}|{ssim_str:>8}|{cfg['paper_psnr']:>8.2f}"
        print(row)

    print("=" * 130)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all", choices=["all", "benchmark", "eval"])
    args = parser.parse_args()

    bench_results = None
    eval_results = None

    if args.mode in ("all", "benchmark"):
        bench_results = run_benchmark()

    if args.mode in ("all", "eval"):
        eval_results = run_eval()

    if args.mode == "all" and bench_results and eval_results:
        print_final_summary(bench_results, eval_results)

    print("\nDone!")
