import os, re
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

# 和你一致，不改
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def _natural_key(name: str):
    # 自然排序：img2 < img10；提取数字块按数值比较
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', name)]

class PairedDataset(Dataset):
    """
    预切块配对数据集：
    root_dir/
        source_patch/  (源域patch)   e.g. image0001_01.png
        target_patch/  (目标域patch) e.g. image0001_01.png
    要求两侧文件名（含扩展名）完全一致，一一对应。
    """
    def __init__(self, root_dir, patch_size=256, transform=None, stride=None, recursive=False, return_name=False):
        """
        Args:
            root_dir: 数据集根目录
            patch_size: patch尺寸（预切块模式下仅用于日志）
            transform: 数据增强（会同时作用于source/target）
            recursive: 是否递归（一般False）
        """
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.transform = transform
        self.precut = True  # 预切块模式

        # 强制使用你提供的目录结构（此处改为 target_patch）
        self.source_dir = os.path.join(root_dir, 'source_patch')
        self.target_dir = os.path.join(root_dir, 'target_patch')

        if not os.path.isdir(self.source_dir):
            raise FileNotFoundError(f"Source目录不存在: {self.source_dir}")
        if not os.path.isdir(self.target_dir):
            raise FileNotFoundError(f"Target目录不存在: {self.target_dir}")

        print(f"\n{'='*60}")
        print(f"[PairedDataset] 初始化（预切块模式）")
        print(f"  - 根目录: {root_dir}")
        print(f"  - 源染色目录: {self.source_dir}")
        print(f"  - 目标染色目录: {self.target_dir}")
        print(f"  - 预期Patch大小: {patch_size}x{patch_size}")
        print(f"{'='*60}\n")

        # 收集配对文件（非递归：按完整文件名交集）
        self.paired_paths = []
        self._collect_paired_images(recursive=False)

        if len(self.paired_paths) == 0:
            raise FileNotFoundError(f"No paired images found in {root_dir}")

        # 预切块：每对文件就是一个样本，占位(row,col)为0
        self.patch_indices = [(i, 0, 0) for i in range(len(self.paired_paths))]

        print(f"✓ 找到 {len(self.paired_paths)} 对配对patch（每个文件即一个样本）")
        print(f"{'='*60}\n")

    def _collect_paired_images(self, recursive=False):
        print("[PairedDataset] 正在收集配对图像...")

        if recursive:
            # 递归：使用相对路径键进行匹配
            src_map, tgt_map = {}, {}
            for root, _, files in os.walk(self.source_dir):
                for f in files:
                    ext = os.path.splitext(f)[1].lower()
                    if ext in IMG_EXTS:
                        rel = os.path.relpath(os.path.join(root, f), self.source_dir)
                        src_map[rel] = os.path.join(root, f)
            for root, _, files in os.walk(self.target_dir):
                for f in files:
                    ext = os.path.splitext(f)[1].lower()
                    if ext in IMG_EXTS:
                        rel = os.path.relpath(os.path.join(root, f), self.target_dir)
                        tgt_map[rel] = os.path.join(root, f)
            keys = sorted(set(src_map.keys()) & set(tgt_map.keys()), key=_natural_key)
            self.paired_paths = [(src_map[k], tgt_map[k]) for k in keys]
            print(f"  - Source文件数: {len(src_map)}")
            print(f"  - Target文件数: {len(tgt_map)}")
        else:
            # 非递归：按完整文件名（含扩展名）做交集
            src_files = {}
            tgt_files = {}
            for f in os.listdir(self.source_dir):
                fp = os.path.join(self.source_dir, f)
                if os.path.isfile(fp) and os.path.splitext(f)[1].lower() in IMG_EXTS:
                    src_files[f] = fp
            for f in os.listdir(self.target_dir):
                fp = os.path.join(self.target_dir, f)
                if os.path.isfile(fp) and os.path.splitext(f)[1].lower() in IMG_EXTS:
                    tgt_files[f] = fp
            common = sorted(set(src_files.keys()) & set(tgt_files.keys()), key=_natural_key)
            self.paired_paths = [(src_files[name], tgt_files[name]) for name in common]
            print(f"  - Source文件数: {len(src_files)}")
            print(f"  - Target文件数: {len(tgt_files)}")

        print(f"  - 配对成功数: {len(self.paired_paths)}")
        if len(self.paired_paths) > 0:
            print("\n  配对示例（前3对）:")
            for i, (s, t) in enumerate(self.paired_paths[:3], 1):
                print(f"    {i}. {os.path.basename(s)} <-> {os.path.basename(t)}")

    def __len__(self):
        return len(self.patch_indices)

    def __getitem__(self, idx):
        pair_idx, _, _ = self.patch_indices[idx]
        src_path, tgt_path = self.paired_paths[pair_idx]

        # 读取配对patch（每个文件就是一个patch）
        src_img = Image.open(src_path).convert('RGB')
        tgt_img = Image.open(tgt_path).convert('RGB')

        if self.transform:
            src_img = self.transform(src_img)
            tgt_img = self.transform(tgt_img)

        return src_img, tgt_img
