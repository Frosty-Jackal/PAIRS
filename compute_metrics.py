"""
compute_metrics.py — Evaluate all metrics on CelebRefHQHR test set for HrRestore paper.

Usage:
    python compute_metrics.py --checkpoint path/to/final_model_ckpt.pt [--data-root ./CelebRefHQHR/test] [--output-dir ./metrics]
    python compute_metrics.py --checkpoint ckpt.pt --metrics lpips,psnr,ssim
    python compute_metrics.py --checkpoint ckpt.pt --metrics all   (default)

Supported metrics:
    lpips   — Learned Perceptual Image Patch Similarity (AlexNet backbone, community standard)
    psnr    — Peak Signal-to-Noise Ratio
    ssim    — Structural Similarity Index Measure
    id      — ArcFace Identity cosine similarity
    params  — Model parameter count (computed once at startup, not per-sample)

Output files (written under --output-dir):
    metrics_detailed.csv       per-sample scores for all metrics
    metrics_summary.csv         per-scale mean / std / median for each metric
    metrics_overall.txt         single-line overall metrics for paper table
"""

import argparse
import csv
import sys
from glob import glob
from pathlib import Path

import numpy as np
import torch

from natsort import natsorted
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms as T

# Allow imports from the HRFR codebase (sibling of this script)
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import lpips
import pyrallis
from face_replace.configs.train_config import TrainConfig
from face_replace.models.face_replace_model import FaceReplaceModel
from face_replace.training.criteria.id_loss import IDLoss

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMG_SIZE = 512
MAX_COND_IMAGES = 3
NORMALIZE_MEAN = [0.5, 0.5, 0.5]
NORMALIZE_STD = [0.5, 0.5, 0.5]

# ArcFace backbone path (relative to HRFR/ — same as training)
ARCFACE_PATH = str(SCRIPT_DIR / "external_models" / "model_ir_se50.pth")

ALL_METRICS = ["lpips", "psnr", "ssim", "id", "params"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_transform():
    return T.Compose([
        T.Resize(IMG_SIZE, interpolation=T.InterpolationMode.LANCZOS),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def preprocess_input(pil_img: Image.Image, transform) -> torch.Tensor:
    """Return a (1, 3, 512, 512) tensor in [-1, 1]."""
    return transform(pil_img).unsqueeze(0)


def preprocess_conds(ref_list, transform) -> torch.Tensor:
    """Return (1, N, 3, 512, 512) tensor in [-1, 1]."""
    conds = torch.stack([transform(ref) for ref in ref_list], dim=0)
    return conds.unsqueeze(0)


def tensor_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    """Convert a (1, 3, H, W) tensor in [-1, 1] → (H, W, 3) uint8 numpy [0, 255]."""
    arr = tensor.squeeze(0).cpu().float().numpy()          # (3, H, W)
    arr = np.transpose(arr, (1, 2, 0))                      # (H, W, 3)
    arr = ((arr + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return arr


# ---------------------------------------------------------------------------
# Metric initializers
# ---------------------------------------------------------------------------
def init_lpips(device):
    """LPIPS with AlexNet backbone — community standard for face restoration."""
    print("Initializing LPIPS (AlexNet) ...")
    return lpips.LPIPS(net="alex", verbose=False).to(device)


def init_id_loss(device):
    """ArcFace ID loss — same model as training."""
    print(f"Initializing ArcFace ID loss from {ARCFACE_PATH} ...")
    id_loss = IDLoss(
        pretrained_arcface_path=ARCFACE_PATH,
        device=device,
        dtype=torch.float32,
    )
    id_loss.eval()
    return id_loss



def count_params(model) -> dict:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


# ---------------------------------------------------------------------------
# Per-sample metric functions
# ---------------------------------------------------------------------------
def compute_psnr(pred_tensor: torch.Tensor, gt_tensor: torch.Tensor) -> float:
    """PSNR between two (1, 3, H, W) tensors in [-1, 1]."""
    pred_uint8 = tensor_to_uint8(pred_tensor)
    gt_uint8 = tensor_to_uint8(gt_tensor)
    return psnr(gt_uint8, pred_uint8, data_range=255)


def compute_ssim(pred_tensor: torch.Tensor, gt_tensor: torch.Tensor) -> float:
    """SSIM between two (1, 3, H, W) tensors in [-1, 1].
    Returns MS-SSIM compatible value (channel-wise mean SSIM).
    """
    pred_uint8 = tensor_to_uint8(pred_tensor)
    gt_uint8 = tensor_to_uint8(gt_tensor)
    # Compute SSIM per channel, then average (standard for color images)
    return ssim(gt_uint8, pred_uint8, channel_axis=2, data_range=255)


def compute_lpips_score(pred_tensor: torch.Tensor, gt_tensor: torch.Tensor, lpips_fn) -> float:
    """LPIPS between two tensors in [-1, 1]."""
    return lpips_fn(pred_tensor.float(), gt_tensor.float()).item()


def compute_id_similarity(pred_tensor: torch.Tensor, gt_tensor: torch.Tensor, id_loss) -> float:
    """ArcFace cosine similarity between predicted and GT faces.
    Returns similarity in [0, 1] (higher = same identity).
    """
    _, sim = id_loss(
        predicted_pixel_values=pred_tensor.float(),
        target_pixel_values=gt_tensor.float(),
    )
    return sim.item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Multi-metric evaluation for HrRestore")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the model checkpoint, e.g. final_model_ckpt.pt")
    parser.add_argument("--data-root", type=str, default="./CelebRefHQHR/test",
                        help="Root of the CelebRefHQHR test folder (contains x4, x8, x16)")
    parser.add_argument("--output-dir", type=str, default="./metrics",
                        help="Directory to write results into")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on (cuda / cpu)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit total samples for quick testing (omit for full eval)")
    parser.add_argument("--metrics", type=str, default="all",
                        help="Comma-separated metrics to compute, e.g. 'lpips,psnr,ssim,id,params' "
                             "or 'psnr,ssim'. Use 'all' for everything (default). "
                             f"Available: {', '.join(ALL_METRICS)}")
    parser.add_argument("--arcface-path", type=str, default=ARCFACE_PATH,
                        help="Path to ArcFace backbone model_ir_se50.pth")
    args = parser.parse_args()

    # --- Parse metrics selection ---
    if args.metrics.strip().lower() == "all":
        enabled = set(ALL_METRICS)
    else:
        enabled = set(m.strip().lower() for m in args.metrics.split(","))
        unknown = enabled - set(ALL_METRICS)
        if unknown:
            print(f"ERROR: Unknown metric(s): {', '.join(unknown)}")
            print(f"Available: {', '.join(ALL_METRICS)}")
            sys.exit(1)

    print(f"Metrics enabled: {', '.join(sorted(enabled))}")
    print()

    # ------------------------------------------------------------------
    # 0. Setup
    # ------------------------------------------------------------------
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"Data root   : {data_root}")
    print(f"Output dir  : {output_dir}")
    print(f"Device      : {device}")
    print(f"Checkpoint  : {args.checkpoint}")
    print()

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print("Loading model ...")
    checkpoint_dict = torch.load(args.checkpoint, map_location="cpu")
    cfg = pyrallis.decode(TrainConfig, checkpoint_dict["cfg"])
    model = FaceReplaceModel(cfg=cfg.model, full_cfg=cfg, evaluating=True)

    try:
        model.load_state_dict(checkpoint_dict["state_dict"], strict=True)
    except Exception:
        fixed_sd = {k.replace(".module.", "."): v
                    for k, v in checkpoint_dict["state_dict"].items()}
        model.load_state_dict(fixed_sd, strict=True)

    model.eval()
    model.net.noise_timesteps = [249]
    model = model.to(device)
    print("Model loaded.\n")

    # ------------------------------------------------------------------
    # 2. Params (once — only depends on model structure)
    # ------------------------------------------------------------------
    params_info = None
    if "params" in enabled:
        params_info = count_params(model)
        print(f"Model params — Total: {params_info['total']:,}  "
              f"Trainable: {params_info['trainable']:,}\n")

    # ------------------------------------------------------------------
    # 3. Initialize selected metrics
    # ------------------------------------------------------------------
    lpips_fn = init_lpips(device) if "lpips" in enabled else None
    id_loss = init_id_loss(device) if "id" in enabled else None

    # ------------------------------------------------------------------
    # 4. Discover test samples
    # ------------------------------------------------------------------
    scales = ["x4", "x8", "x16"]
    transform = build_transform()

    samples = []
    for scale in scales:
        scale_dir = data_root / scale
        if not scale_dir.is_dir():
            print(f"WARNING: {scale_dir} not found — skipping")
            continue
        for identity_dir in sorted(scale_dir.iterdir()):
            if not identity_dir.is_dir():
                continue
            for img_dir in sorted(identity_dir.iterdir()):
                if img_dir.is_dir():
                    samples.append((scale, identity_dir.name, img_dir))

    print(f"Found {len(samples)} total samples across {len(scales)} scales.")
    if args.max_samples:
        samples = samples[:args.max_samples]
        print(f"Limited to {len(samples)} samples (--max-samples={args.max_samples}).")
    print()

    # ------------------------------------------------------------------
    # 5. Evaluate
    # ------------------------------------------------------------------
    all_results = []                                          # list of dicts (per-sample)
    scale_scores = {m: {s: [] for s in scales} for m in enabled if m != "params"}

    for idx, (scale, identity, img_dir) in enumerate(samples):
        degraded_path = img_dir / "degraded.png"
        gt_path = img_dir / "gt.png"
        cond_dir = img_dir / "conditioning"

        if not degraded_path.exists() or not gt_path.exists():
            print(f"[{idx+1}/{len(samples)}] SKIP {scale}/{identity}/{img_dir.name} — missing degraded or gt")
            continue

        # --- conditioning images ---
        cond_paths = natsorted(glob(str(cond_dir / "*.png")))[:MAX_COND_IMAGES]
        if len(cond_paths) == 0:
            print(f"[{idx+1}/{len(samples)}] SKIP {scale}/{identity}/{img_dir.name} — no conditioning images")
            continue

        while len(cond_paths) < MAX_COND_IMAGES:
            cond_paths.append(cond_paths[-1])

        ref_pils = [load_image(p) for p in cond_paths]

        # --- preprocess ---
        inp_t = preprocess_input(load_image(degraded_path), transform)
        conds_t = preprocess_conds(ref_pils, transform)
        valid = torch.tensor([MAX_COND_IMAGES], dtype=torch.int)

        # --- inference ---
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                x_pred, _, _ = model.net.forward(
                    inp_t.to(device, dtype),
                    conditioning_images=conds_t.to(device, dtype),
                    valid_indices=valid,
                )

        # x_pred in [-1, 1], (1, 3, 512, 512)
        gt_pil = load_image(gt_path)
        gt_tensor_512 = pil_to_metric_tensor(gt_pil).to(device)   # for LPIPS
        pred_float = x_pred.float()
        gt_float = gt_tensor_512.float()

        # --- compute per-sample metrics ---
        row = {"scale": scale, "identity": identity, "sample": img_dir.name}

        if lpips_fn is not None:
            row["lpips"] = compute_lpips_score(pred_float, gt_float, lpips_fn)

        if "psnr" in enabled:
            row["psnr"] = compute_psnr(x_pred.cpu(), gt_tensor_512.cpu())

        if "ssim" in enabled:
            row["ssim"] = compute_ssim(x_pred.cpu(), gt_tensor_512.cpu())

        if id_loss is not None:
            id_sim = compute_id_similarity(pred_float, gt_float, id_loss)
            row["id"] = id_sim

        all_results.append(row)

        # Update per-scale accumulators
        for metric_name in row:
            if metric_name in ("scale", "identity", "sample"):
                continue
            if metric_name in scale_scores:
                scale_scores[metric_name][scale].append(row[metric_name])
            else:
                # Only happens if metric not in enabled set
                pass

        # Progress
        if (idx + 1) % 50 == 0 or (idx + 1) == len(samples):
            parts = [f"{scale}/{identity}/{img_dir.name}"]
            for m in sorted(row.keys()):
                if m in ("scale", "identity", "sample"):
                    continue
                parts.append(f"{m.upper()}={row[m]:.4f}")
            print(f"[{idx+1}/{len(samples)}]  " + "  ".join(parts))

    # ------------------------------------------------------------------
    # 6. Write outputs
    # ------------------------------------------------------------------
    if len(all_results) == 0:
        print("ERROR: No valid samples found — aborting.")
        sys.exit(1)

    # Determine which metric columns to write (exclude params — it's per-model not per-sample)
    metric_cols = [m for m in ALL_METRICS if m != "params" and m in enabled]

    # 7a. Detailed CSV
    detailed_csv = output_dir / "metrics_detailed.csv"
    with open(detailed_csv, "w", newline="") as f:
        fieldnames = ["scale", "identity", "sample"] + metric_cols
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nDetailed results → {detailed_csv}")

    # 7b. Per-scale summary CSV
    summary_csv = output_dir / "metrics_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["metric", "scale", "num_samples", "mean", "std", "median", "min", "max"]
        writer.writerow(header)
        for metric_name in metric_cols:
            for scale in scales:
                vals = scale_scores.get(metric_name, {}).get(scale, [])
                if len(vals) == 0:
                    continue
                writer.writerow([
                    metric_name,
                    scale,
                    len(vals),
                    f"{np.mean(vals):.6f}",
                    f"{np.std(vals):.6f}",
                    f"{np.median(vals):.6f}",
                    f"{np.min(vals):.6f}",
                    f"{np.max(vals):.6f}",
                ])
        # Params row
        if params_info:
            writer.writerow(["params_total", "—", 1, f"{params_info['total']}", "", "", "", ""])
            writer.writerow(["params_trainable", "—", 1, f"{params_info['trainable']}", "", "", "", ""])
    print(f"Per-scale summary → {summary_csv}")

    # 7c. Overall summary (paper-ready)
    overall_txt = output_dir / "metrics_overall.txt"
    with open(overall_txt, "w") as f:
        f.write("=" * 65 + "\n")
        f.write("HrRestore Evaluation — CelebRefHQHR test\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Total samples evaluated: {len(all_results)}\n\n")

        # Per-metric overall
        for metric_name in metric_cols:
            all_vals = [r[metric_name] for r in all_results if metric_name in r]
            if len(all_vals) == 0:
                continue
            f.write(f"Overall {metric_name.upper()} (mean ± std):  "
                    f"{np.mean(all_vals):.6f} ± {np.std(all_vals):.6f}\n")
            f.write(f"Overall {metric_name.upper()} (median):      "
                    f"{np.median(all_vals):.6f}\n\n")

        if params_info:
            f.write(f"Model Params — Total: {params_info['total']:,}  "
                    f"Trainable: {params_info['trainable']:,}\n\n")

        # Per-scale breakdown
        f.write("Per-scale breakdown:\n")
        f.write("-" * 45 + "\n")
        for metric_name in metric_cols:
            f.write(f"\n  {metric_name.upper()}:\n")
            for scale in scales:
                vals = scale_scores.get(metric_name, {}).get(scale, [])
                if len(vals) == 0:
                    continue
                f.write(f"    {scale}:  {np.mean(vals):.6f} ± {np.std(vals):.6f}  (n={len(vals)})\n")

        # Table-ready one-liners
        f.write("\n" + "=" * 65 + "\n")
        f.write("Table-ready one-liners:\n")
        for metric_name in metric_cols:
            all_vals = [r[metric_name] for r in all_results if metric_name in r]
            if len(all_vals) == 0:
                continue
            fmt = ".4f" if metric_name in ("lpips", "ssim") else ".2f"
            f.write(f"  {metric_name.upper()} = {np.mean(all_vals):{fmt}}\n")
        if params_info:
            f.write(f"  Params = {params_info['total']:,} (trainable: {params_info['trainable']:,})\n")

    print(f"Overall summary  → {overall_txt}")

    # Also print to stdout
    print()
    print("=" * 65)
    for metric_name in metric_cols:
        all_vals = [r[metric_name] for r in all_results if metric_name in r]
        if len(all_vals) == 0:
            continue
        print(f"Overall {metric_name.upper()}:  {np.mean(all_vals):.6f} ± {np.std(all_vals):.6f}")
    if params_info:
        print(f"Params — Total: {params_info['total']:,}  Trainable: {params_info['trainable']:,}")
    print("=" * 65)
    print("Done.")


# ---------------------------------------------------------------------------
# Helper: pil_to_tensor for GT (same as pil_to_lpips_tensor in old script)
# ---------------------------------------------------------------------------
def pil_to_metric_tensor(pil_img: Image.Image) -> torch.Tensor:
    """Convert a PIL image to (1, 3, 512, 512) tensor in [-1, 1]."""
    t = T.Compose([
        T.Resize(IMG_SIZE, interpolation=T.InterpolationMode.LANCZOS),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])(pil_img)
    return t.unsqueeze(0)


if __name__ == "__main__":
    main()
