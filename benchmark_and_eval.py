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

    from ptflops import get_model_complexity_info

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

        # FLOPs: always on CPU to avoid GPU OOM
        flops = {}
        model_cpu = model.cpu()
        for sz in INPUT_SIZES:
            try:
                macs, _ = get_model_complexity_info(model_cpu, sz[1:], verbose=False, print_per_layer_stat=False)
                if macs is not None:
                    macs_val = float(macs.replace(" GMACs", "").replace(" GMac", "").replace(" MACs", ""))
                    flops[sz] = macs_val
                    print(f"  MACs {sz[2]}x{sz[3]}: {macs_val:.2f} G")
                else:
                    flops[sz] = 0.0
                    print(f"  MACs {sz[2]}x{sz[3]}: N/A")
            except Exception as e:
                flops[sz] = 0.0
                print(f"  MACs {sz[2]}x{sz[3]}: error ({e})")
        del model_cpu

        # Latency + VRAM: on GPU
        lat = {}
        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            model.to(device)
            for sz in INPUT_SIZES:
                try:
                    torch.cuda.empty_cache()
                    gc.collect()
                    torch.cuda.reset_peak_memory_stats()
                    x = torch.randn(*sz, device=device)
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
                    lat[sz] = {"ms": avg, "fps": 1000/avg, "vram_mb": vram}
                    print(f"  Lat {sz[2]}x{sz[3]}: {avg:.2f} ms ({1000/avg:.1f} FPS) | VRAM: {vram:.0f} MB")
                    del x
                except torch.cuda.OutOfMemoryError:
                    lat[sz] = None
                    print(f"  Lat {sz[2]}x{sz[3]}: OOM (skipped)")
                    torch.cuda.empty_cache()
                    gc.collect()
                except Exception as e:
                    lat[sz] = None
                    print(f"  Lat {sz[2]}x{sz[3]}: error ({e})")
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
        f256_str = f"{f256:.2f}" if f256 else "N/A"
        f720_str = f"{f720:.2f}" if f720 else "N/A"
        row = (f"{r['name']:<22}|{cfg['task']:<8}"
               f"|{r['params']:>10.2f}"
               f"|{f256_str:>11}|{f720_str:>11}")
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

    from basicsr.train import parse_options
    from basicsr.data import create_dataloader, create_dataset
    from basicsr.models import create_model
    from basicsr.utils import get_root_logger, make_exp_dirs, get_time_str
    import tempfile

    eval_results = []

    for name, cfg in MODEL_CONFIGS.items():
        print(f"\n--- {name} ---")

        with open(cfg["yaml_test"]) as f:
            opt = yaml.safe_load(f)

        opt["path"]["pretrain_network_g"] = cfg["ckpt"]
        opt["val"]["save_img"] = False
        opt["dist"] = False
        opt["rank"] = 0
        opt["world_size"] = 1

        out_dir = f"results/{name}"
        os.makedirs(out_dir, exist_ok=True)
        opt["path"]["log"] = out_dir
        opt["path"]["visualization"] = out_dir
        opt["path"]["models"] = out_dir

        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False, dir=".")
        yaml.dump(opt, tmp)
        tmp.close()

        # Redirect stdout to capture output
        import io
        old_stdout = sys.stdout
        sys.stdout = mystdout = io.StringIO()

        try:
            torch.backends.cudnn.benchmark = True
            from basicsr.utils.options import dict2str
            from basicsr.utils import get_env_info
            import logging

            log_file = os.path.join(out_dir, f"test_{name}_{get_time_str()}.log")
            logger = get_root_logger(logger_name="basicsr", log_level=logging.INFO, log_file=log_file)
            logger.info(get_env_info())

            test_loaders = []
            for phase, dataset_opt in sorted(opt["datasets"].items()):
                if "test" in phase:
                    dataset_opt["phase"] = "test"
                test_set = create_dataset(dataset_opt)
                test_loader = create_dataloader(test_set, dataset_opt, num_gpu=opt["num_gpu"],
                                                dist=opt["dist"], sampler=None, seed=opt["manual_seed"])
                logger.info(f"Number of test images: {len(test_set)}")
                test_loaders.append(test_loader)

            model = create_model(opt)

            for test_loader in test_loaders:
                model.validation(test_loader, current_iter=opt["name"], tb_logger=None,
                                 save_img=opt["val"]["save_img"], rgb2bgr=opt["val"].get("rgb2bgr", True),
                                 use_image=opt["val"].get("use_image", True))

        except Exception as e:
            sys.stdout = old_stdout
            print(f"  [ERROR] {e}")
            import traceback; traceback.print_exc()
            os.unlink(tmp.name)
            eval_results.append({"name": name, "psnr": None, "ssim": None})
            continue

        sys.stdout = old_stdout
        os.unlink(tmp.name)

        # Parse metrics from captured output
        output = mystdout.getvalue()
        psnr_match = re.search(r"psnr:\s*([\d.]+)", output)
        ssim_match = re.search(r"ssim:\s*([\d.]+)", output)

        psnr = float(psnr_match.group(1)) if psnr_match else None
        ssim = float(ssim_match.group(1)) if ssim_match else None

        print(f"  PSNR: {psnr:.4f} dB" if psnr else "  PSNR: N/A")
        print(f"  SSIM: {ssim:.4f}" if ssim else "  SSIM: N/A")
        print(f"  Paper: {cfg['paper_psnr']} dB / {cfg['paper_ssim']}")

        diff = psnr - cfg["paper_psnr"] if psnr else None
        if diff is not None:
            tag = "OK" if abs(diff) < 0.5 else "CHECK"
            print(f"  Delta: {diff:+.4f} dB [{tag}]")

        eval_results.append({"name": name, "psnr": psnr, "ssim": ssim})

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
        f720_str = f"{f720:.2f}" if f720 else "N/A"
        psnr_str = f"{e['psnr']:.2f}" if e["psnr"] else "N/A"
        ssim_str = f"{e['ssim']:.4f}" if e["ssim"] else "N/A"

        row = (f"{b['name']:<22}|{cfg['task']:<8}"
               f"|{b['params']:>10.2f}"
               f"|{f720_str:>9}")
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
