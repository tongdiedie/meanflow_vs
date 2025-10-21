import os
from pathlib import Path
from PIL import Image
import pandas as pd
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}

def prepare_paired_dataset(
    root_dir='/root/autodl-tmp/Ki67/TrainValAB',
    src_sub='valA',
    tgt_sub='valB',
    out_root='/root/autodl-tmp/Ki67/data_paired',
    out_ext='.png',
    verify_same_size=False,
    recursive=False,
    overwrite=True
):
    src_dir = Path(root_dir) / src_sub
    tgt_dir = Path(root_dir) / tgt_sub
    out_src = Path(out_root) / 'source'
    out_tgt = Path(out_root) / 'target'
    out_src.mkdir(parents=True, exist_ok=True)
    out_tgt.mkdir(parents=True, exist_ok=True)

    if not src_dir.exists():
        raise FileNotFoundError(f"Source目录不存在: {src_dir}")
    if not tgt_dir.exists():
        raise FileNotFoundError(f"Target目录不存在: {tgt_dir}")

    def list_files(base: Path):
        if recursive:
            it = base.rglob('*')
        else:
            it = (base / f for f in os.listdir(base))
        files = {}
        for p in it:
            p = Path(p)
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                # 以“去掉扩展名”的名字作为键（用于配对）
                files[p.stem] = p
        return files

    src_files = list_files(src_dir)
    tgt_files = list_files(tgt_dir)

    common = sorted(set(src_files.keys()) & set(tgt_files.keys()))
    only_src = sorted(set(src_files.keys()) - set(tgt_files.keys()))
    only_tgt = sorted(set(tgt_files.keys()) - set(src_files.keys()))

    print("="*60)
    print("[prepare] 文件统计")
    print(f"  - Source文件数: {len(src_files)}")
    print(f"  - Target文件数: {len(tgt_files)}")
    print(f"  - 成功配对数  : {len(common)}")
    if only_src:
        print(f"  - 仅在source出现(未配对): {len(only_src)}（示例：{only_src[:3]}）")
    if only_tgt:
        print(f"  - 仅在target出现(未配对): {len(only_tgt)}（示例：{only_tgt[:3]}）")
    if len(common) == 0:
        raise RuntimeError("没有找到任何可配对的文件（按去扩展名的文件名匹配）。")

    # 自动确定零填充宽度：img001, img002 ... 或 img0001 ...
    pad_width = max(3, len(str(len(common))))

    records = []
    for i, stem in enumerate(common, start=1):
        idx = str(i).zfill(pad_width)
        new_name = f"img{idx}{out_ext}"

        src_path = src_files[stem]
        tgt_path = tgt_files[stem]

        with Image.open(src_path) as im_src, Image.open(tgt_path) as im_tgt:
            im_src = im_src.convert('RGB')
            im_tgt = im_tgt.convert('RGB')

            if verify_same_size and (im_src.size != im_tgt.size):
                print(f"  ! 尺寸不一致：{stem}  src={im_src.size}, tgt={im_tgt.size}")

            w, h = im_src.size

            dst_src = out_src / new_name
            dst_tgt = out_tgt / new_name

            if (not overwrite) and (dst_src.exists() or dst_tgt.exists()):
                raise FileExistsError(f"目标文件已存在：{dst_src} 或 {dst_tgt}")

            im_src.save(dst_src)
            im_tgt.save(dst_tgt)

        records.append({
            "index": idx,
            "new_name": new_name,
            "source_original": str(src_path),
            "target_original": str(tgt_path),
            "width": w,
            "height": h
        })

        if i <= 3:
            print(f"  示例映射 {i}: {stem} -> {new_name}")

    # 保存为 CSV 和 Excel
    df = pd.DataFrame(records)
    csv_path = Path(out_root) / 'index.csv'
    xlsx_path = Path(out_root) / 'index.xlsx'
    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)

    print("="*60)
    print("✓ 输出完成：")
    print(f"  - Source: {out_src}")
    print(f"  - Target: {out_tgt}")
    print(f"  - 映射CSV: {csv_path}")
    print(f"  - 映射Excel: {xlsx_path}")
    print("="*60)

if __name__ == "__main__":
    prepare_paired_dataset(
        root_dir='/root/autodl-tmp/Ki67/TrainValAB',
        src_sub='valA',
        tgt_sub='valB',
        out_root='/root/autodl-tmp/Ki67/data_paired',
        out_ext='.png',            # 统一转为 PNG
        verify_same_size=False,    # 如需强校验可改 True
        recursive=False,           # 如需递归子目录可改 True
        overwrite=True             # 已存在时覆盖
    )
