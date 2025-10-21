import argparse, os, glob, csv
from typing import List, Tuple, Dict
from PIL import Image
import numpy as np
import torch
from torchvision import transforms as T
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
import lpips

# TorchMetrics: FID & KID
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def _list_images(root: str) -> Dict[str, str]:
    files = {}
    for ext in IMG_EXTS:
        for p in glob.glob(os.path.join(root, f'**/*{ext}'), recursive=True):
            key = os.path.splitext(os.path.basename(p))[0]
            files[key] = p
    return files

def _pair_by_name(pred_dir: str, gt_dir: str) -> List[Tuple[str, str]]:
    pred_map = _list_images(pred_dir)
    gt_map = _list_images(gt_dir)
    keys = sorted(set(pred_map.keys()) & set(gt_map.keys()))
    return [(pred_map[k], gt_map[k]) for k in keys]

def _psnr_ssim_pair(pred_img: np.ndarray, gt_img: np.ndarray) -> Tuple[float, float]:
    # to uint8 if necessary
    if pred_img.dtype != np.uint8:
        pred_img = (np.clip(pred_img, 0, 1) * 255).astype(np.uint8)
    if gt_img.dtype != np.uint8:
        gt_img = (np.clip(gt_img, 0, 1) * 255).astype(np.uint8)

    psnr = sk_psnr(gt_img, pred_img, data_range=255)
    # SSIM 分通道求平均（与你现有实现一致）
    ssim_c = []
    for c in range(3):
        ssim_c.append(sk_ssim(gt_img[..., c], pred_img[..., c], data_range=255))
    ssim = float(np.mean(ssim_c))
    return float(psnr), float(ssim)

@torch.no_grad()
def evaluate(pred_dir: str, gt_dir: str, device: str = None, csv_out: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] device={device}")

    pairs = _pair_by_name(pred_dir, gt_dir)
    if len(pairs) == 0:
        raise FileNotFoundError(f"No matched image pairs between {pred_dir} and {gt_dir} by base filename.")

    # LPIPS (VGG)
    lpips_fn = lpips.LPIPS(net='vgg').to(device).eval()
    to_tensor = T.ToTensor()

    # TorchMetrics
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    kid = KernelInceptionDistance(subset_size=50, normalize=True).to(device)

    per_image_rows = []  # slide/fov不一定有，用文件名代替
    psnr_list, ssim_list, lpips_list = [], [], []

    for pred_path, gt_path in pairs:
        pred = Image.open(pred_path).convert('RGB')
        gt = Image.open(gt_path).convert('RGB')

        # === PSNR & SSIM ===
        pred_np = np.array(pred)
        gt_np = np.array(gt)
        psnr, ssim = _psnr_ssim_pair(pred_np, gt_np)
        psnr_list.append(psnr); ssim_list.append(ssim)

        # === LPIPS ===
        pred_t = to_tensor(pred).unsqueeze(0).to(device)
        gt_t = to_tensor(gt).unsqueeze(0).to(device)
        lp = lpips_fn(pred_t, gt_t).mean().item()
        lpips_list.append(lp)

        # === FID / KID 累加 ===
        # TorchMetrics 需要 [0,255] uint8 BCHW 或 float with [0,1], normalize=True 时传 [0,1]
        fid.update(gt_t, real=True)
        fid.update(pred_t, real=False)
        kid.update(gt_t, real=True)
        kid.update(pred_t, real=False)

        base = os.path.splitext(os.path.basename(pred_path))[0]
        per_image_rows.append([base, psnr, ssim, lp])

    fid_score = float(fid.compute().cpu().item())
    kid_mean, kid_std = kid.compute()
    kid_mean = float(kid_mean.cpu().item())
    kid_std = float(kid_std.cpu().item())

    summary = {
        "N": len(pairs),
        "FID": fid_score,
        "KID_mean": kid_mean,
        "KID_std": kid_std,
        "PSNR_mean": float(np.mean(psnr_list)),
        "PSNR_std": float(np.std(psnr_list)),
        "SSIM_mean": float(np.mean(ssim_list)),
        "SSIM_std": float(np.std(ssim_list)),
        "LPIPS_mean": float(np.mean(lpips_list)),
        "LPIPS_std": float(np.std(lpips_list)),
    }

    print("\n===== Metrics (dataset level) =====")
    for k, v in summary.items():
        print(f"{k}: {v}")

    if csv_out:
        os.makedirs(os.path.dirname(csv_out), exist_ok=True)
        with open(csv_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "PSNR", "SSIM", "LPIPS"])
            w.writerows(per_image_rows)
        print(f"[Eval] Per-image metrics saved to: {csv_out}")

    return summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True, help="directory of predicted images")
    ap.add_argument("--gt_dir", required=True, help="directory of ground-truth images")
    ap.add_argument("--csv_out", default="metrics/per_image_metrics.csv")
    args = ap.parse_args()
    evaluate(args.pred_dir, args.gt_dir, csv_out=args.csv_out)

if __name__ == "__main__":
    main()
