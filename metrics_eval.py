import argparse, os, glob, csv, time
from typing import List, Tuple, Dict
from PIL import Image
import numpy as np
import torch
from torchvision import transforms as T
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
import lpips
import pandas as pd
# TorchMetrics: FID & KID
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

# ==== 软依赖：torchmetrics.image.*（FID/KID需要torch-fidelity） ====
_HAS_TM = True
_TM_ERR = None
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
except Exception as e:
    _HAS_TM = False
    _TM_ERR = e

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
def evaluate(pred_dir: str,
             gt_dir: str,
             device: str = None,
             csv_out: str = None,
             xlsx_out: str = "metrics/metrics.xlsx",
             exp_meta: dict = None):
    """
    统一将所有指标写入一个 Excel 的同一张表（sheet='metrics'，长表格式）。
    每次评估会追加：1行summary + 若干行per_image。
    列：['type','name','stamp','N','FID','KID_mean','KID_std','PSNR','SSIM','LPIPS', ...exp_meta]
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] device={device}")
    stamp = time.strftime("%Y%m%d-%H%M%S")

    pairs = _pair_by_name(pred_dir, gt_dir)
    if len(pairs) == 0:
        raise FileNotFoundError(f"No matched image pairs between {pred_dir} and {gt_dir} by base filename.")

    # LPIPS (VGG)
    lpips_fn = lpips.LPIPS(net='vgg').to(device).eval()
    to_tensor = T.ToTensor()

    # FID/KID（可能因缺依赖而禁用）
    fid = kid = None
    fidkid_enabled = False
    if _HAS_TM:
        try:
            fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
            kid = KernelInceptionDistance(subset_size=50, normalize=True).to(device)
            fidkid_enabled = True
        except Exception as e:
            print(f"[Eval] ⚠️ FID/KID disabled: {e}")
    else:
        print(f"[Eval] ⚠️ FID/KID disabled: {_TM_ERR}")

    per_image_rows = []  # ["name", PSNR, SSIM, LPIPS]
    psnr_list, ssim_list, lpips_list = [], [], []

    for pred_path, gt_path in pairs:
        pred = Image.open(pred_path).convert('RGB')
        gt = Image.open(gt_path).convert('RGB')

        # PSNR & SSIM
        pred_np = np.array(pred)
        gt_np = np.array(gt)
        psnr, ssim = _psnr_ssim_pair(pred_np, gt_np)
        psnr_list.append(psnr); ssim_list.append(ssim)

        # LPIPS
        pred_t = to_tensor(pred).unsqueeze(0).to(device)
        gt_t = to_tensor(gt).unsqueeze(0).to(device)
        lp = lpips_fn(pred_t, gt_t).mean().item()
        lpips_list.append(lp)

        # FID/KID 累加
        if fidkid_enabled:
            fid.update(gt_t, real=True);  fid.update(pred_t, real=False)
            kid.update(gt_t, real=True);  kid.update(pred_t, real=False)

        base = os.path.splitext(os.path.basename(pred_path))[0]
        per_image_rows.append([base, psnr, ssim, lp])

    # 汇总
    if fidkid_enabled:
        fid_score = float(fid.compute().cpu().item())
        kid_mean, kid_std = kid.compute()
        kid_mean = float(kid_mean.cpu().item())
        kid_std  = float(kid_std.cpu().item())
    else:
        fid_score = None
        kid_mean = None
        kid_std  = None

    summary = {
        "N": len(pairs),
        "FID": fid_score,
        "KID_mean": kid_mean,
        "KID_std": kid_std,
        "PSNR_mean": float(np.mean(psnr_list)),
        "PSNR_std":  float(np.std(psnr_list)),
        "SSIM_mean": float(np.mean(ssim_list)),
        "SSIM_std":  float(np.std(ssim_list)),
        "LPIPS_mean": float(np.mean(lpips_list)),
        "LPIPS_std":  float(np.std(lpips_list)),
    }

    # 控制台打印
    print("\n===== Metrics (dataset level) =====")
    def _fmt(v): return "NA" if v is None else v
    for k, v in summary.items():
        print(f"{k}: {_fmt(v)}")
    if not fidkid_enabled:
        print("⚠️ 提示：未安装 `torch-fidelity`（或 `pip install torchmetrics[image]`），已跳过 FID/KID。")

    # 可选：逐图 CSV（保留你原先的功能）
    if csv_out:
        os.makedirs(os.path.dirname(csv_out), exist_ok=True)
        with open(csv_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "PSNR", "SSIM", "LPIPS"])
            w.writerows(per_image_rows)
        print(f"[Eval] Per-image metrics saved to: {csv_out}")

    # —— 统一 Excel：一个文件，一个 sheet（长表），每次评估追加多行 ——
    if xlsx_out:
        os.makedirs(os.path.dirname(xlsx_out), exist_ok=True)

        # 1) 组装 summary 行（type='summary'）
        row_sum = {
            "type": "summary", "name": "", "stamp": stamp, "N": summary["N"],
            "FID": summary["FID"], "KID_mean": summary["KID_mean"], "KID_std": summary["KID_std"],
            "PSNR": summary["PSNR_mean"], "SSIM": summary["SSIM_mean"], "LPIPS": summary["LPIPS_mean"],
        }
        if exp_meta:
            for k, v in exp_meta.items():
                row_sum[str(k)] = v

        # 2) 逐图行（type='per_image'）
        rows_per = []
        for name, psnr, ssim, lp in per_image_rows:
            r = {"type": "per_image", "name": name, "stamp": stamp, "N": None,
                 "FID": None, "KID_mean": None, "KID_std": None,
                 "PSNR": psnr, "SSIM": ssim, "LPIPS": lp}
            if exp_meta:
                for k, v in exp_meta.items():
                    r[str(k)] = v
            rows_per.append(r)

        df_all = pd.DataFrame([row_sum] + rows_per)
        sheet = "metrics"

        if os.path.exists(xlsx_out):
            # 读取已存在sheet以对齐列并追加行
            try:
                existing = pd.read_excel(xlsx_out, sheet_name=sheet)
                all_cols = list(dict.fromkeys(list(existing.columns) + list(df_all.columns)))
                existing = existing.reindex(columns=all_cols)
                df_all   = df_all.reindex(columns=all_cols)
                with pd.ExcelWriter(xlsx_out, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
                    startrow = len(existing) + 1
                    df_all.to_excel(writer, sheet_name=sheet, index=False, header=False, startrow=startrow)
            except Exception:
                # 没有该sheet则创建
                with pd.ExcelWriter(xlsx_out, engine="openpyxl", mode="a") as writer:
                    df_all.to_excel(writer, sheet_name=sheet, index=False)
        else:
            with pd.ExcelWriter(xlsx_out, engine="openpyxl", mode="w") as writer:
                df_all.to_excel(writer, sheet_name=sheet, index=False)

        print(f"[Eval] Excel appended to: {xlsx_out}")

    return summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--csv_out", default=None)
    ap.add_argument("--xlsx_out", default="metrics/metrics.xlsx")
    ap.add_argument("--exp_meta", default=None, help="JSON string for extra columns")
    args = ap.parse_args()
    meta = json.loads(args.exp_meta) if args.exp_meta else None
    evaluate(args.pred_dir, args.gt_dir,
             csv_out=args.csv_out,
             xlsx_out=args.xlsx_out,
             exp_meta=meta)


if __name__ == "__main__":
    main()
