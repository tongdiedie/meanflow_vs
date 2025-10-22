import os, re
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
# 在计算行列数与真正裁剪之前，先把两张配对图中心裁成同尺寸，再pad 到 patch_size 的整数倍，这样上/下两排一定像素对齐。
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def _natural_key(name: str):
    # 自然排序：img2 < img10；提取数字块按数值比较
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', name)]

# ========================= 新增：对齐与填充工具函数 =========================
def _center_crop_to_same(src_img, tgt_img):
    """将两张图中心裁成相同尺寸（取各自宽高的最小值）"""
    w = min(src_img.width, tgt_img.width)
    h = min(src_img.height, tgt_img.height)
    def _c(img, w, h):
        left = (img.width  - w) // 2
        top  = (img.height - h) // 2
        return img.crop((left, top, left + w, top + h))
    return _c(src_img, w, h), _c(tgt_img, w, h)

def _pad_to_multiple(img, patch):
    # pad 到 patch_size 的整数倍（右/下方向），避免最后一行/列尺寸不足
    pw = ((img.width  + patch - 1) // patch) * patch
    ph = ((img.height + patch - 1) // patch) * patch
    pad_w = pw - img.width
    pad_h = ph - img.height
    if pad_w == 0 and pad_h == 0:
        return img
    # 用边缘延拓，避免硬填纯色
    img = ImageOps.expand(img, border=(0, 0, pad_w, pad_h), fill=None)
    # 补丁：Pillow 的边缘延拓对 RGB 不总是生效，这里把最后一列/行复制过去
    if pad_w > 0:
        right = img.crop((img.width - 1 - pad_w, 0, img.width - 1, img.height))
        img.paste(right, (img.width - pad_w, 0))
    if pad_h > 0:
        bottom = img.crop((0, img.height - 1 - pad_h, img.width, img.height - 1))
        img.paste(bottom, (0, img.height - pad_h))
    return img

class PairedDataset(Dataset):
    """
    配对数据集，用于虚拟染色任务
    
    数据结构应该是：
    root_dir/
        source/  (源染色，如HE)
            img0001.png
            img0002.png
        target/  (目标染色，如IHC)
            img0001.png
            img0002.png
    
    关键：source和target文件夹中的图像必须按文件名一一对应
    """
    def __init__(self, root_dir, patch_size=256, transform=None, stride=None, recursive=False, return_name=False):
        """
        Args:
            root_dir: 数据集根目录
            patch_size: 切分块的大小 (256, 512, etc.)
            transform: 数据增强变换（会同时应用到source和target）
            recursive: 是否递归搜索子文件夹
        """
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.transform = transform
        
        # 定义source和target文件夹路径
        self.source_dir = os.path.join(root_dir, 'source')  
        self.target_dir = os.path.join(root_dir, 'target')  
        
        self.paired_paths = []
        self._collect_paired_images(recursive)

        if len(self.paired_paths) == 0:
            raise FileNotFoundError(f"No paired images found in {root_dir}")

        # 预计算每对图像的 patch 索引：[(pair_idx, r, c), ...]
        self.patch_indices = []
        self._precompute_patch_info()

        print(f"\n{'='*60}")
        print(f"[PairedDataset] 初始化配对数据集")
        print(f"  - 根目录: {root_dir}")
        print(f"  - 源染色目录: {self.source_dir}")
        print(f"  - 目标染色目录: {self.target_dir}")
        print(f"  - Patch大小: {patch_size}x{patch_size}")
        print(f"{'='*60}\n")

        # 检查文件夹是否存在
        if not os.path.exists(self.source_dir):
            raise FileNotFoundError(f"Source目录不存在: {self.source_dir}")
        if not os.path.exists(self.target_dir):
            raise FileNotFoundError(f"Target目录不存在: {self.target_dir}")
        
        # 收集配对的图像路径
        self.paired_paths = []
        self._collect_paired_images(recursive)
        
        if len(self.paired_paths) == 0:
            raise FileNotFoundError(f"在 {root_dir} 中没有找到配对图像")
        
        print(f"✓ 找到 {len(self.paired_paths)} 对配对图像")
        
        # 预计算每对图像的patch信息
        self.patch_indices = []  # [(pair_idx, row_idx, col_idx), ...]
        self._precompute_patch_info()
        
        print(f"✓ 总共生成 {len(self.patch_indices)} 个配对patch")
        print(f"{'='*60}\n")

    def _collect_paired_images(self, recursive):
        """收集配对的图像路径"""
        print("[PairedDataset] 正在收集配对图像...")
        
        if recursive:
            # 递归搜索
            source_files = {}
            target_files = {}
            
            for root, _, files in os.walk(self.source_dir):
                for f in files:
                    if os.path.splitext(f)[1].lower() in IMG_EXTS:
                        rel = os.path.relpath(os.path.join(root, f), self.source_dir)
                        source_files[rel] = os.path.join(root, f)
            for root, _, files in os.walk(self.target_dir):
                for f in files:
                    if os.path.splitext(f)[1].lower() in IMG_EXTS:
                        rel = os.path.relpath(os.path.join(root, f), self.target_dir)
                        target_files[rel] = os.path.join(root, f)
            common_files = sorted(set(source_files.keys()) & set(target_files.keys()), key=_natural_key)
            self.paired_paths = [(source_files[k], target_files[k], os.path.splitext(os.path.basename(k))[0]) for k in common_files]
            for rel in common_files:
                self.paired_paths.append((source_files[rel], target_files[rel]))
        else:
            src = {f: os.path.join(self.source_dir, f)
                   for f in os.listdir(self.source_dir)
                   if os.path.splitext(f)[1].lower() in IMG_EXTS}
            tgt = {f: os.path.join(self.target_dir, f)
                   for f in os.listdir(self.target_dir)
                   if os.path.splitext(f)[1].lower() in IMG_EXTS}
            common_files = sorted(set(src.keys()) & set(tgt.keys()), key=_natural_key)
            for fname in common_files:
                self.paired_paths.append((src[fname], tgt[fname]))
        
        print(f"  - Source文件数: {len(self.paired_paths) if not recursive else '见上'}")
        print(f"  - Target文件数: {len(self.paired_paths) if not recursive else '见上'}")
        print(f"  - 配对成功数: {len(self.paired_paths)}")
        
        # 显示前3对作为示例
        if len(self.paired_paths) > 0:
            print("\n  配对示例（前3对）:")
            for i, (s, t) in enumerate(self.paired_paths[:3], 1):
                print(f"    {i}. {os.path.basename(s)} <-> {os.path.basename(t)}")

    def _precompute_patch_info(self):
        """预计算每对图像能生成的patch数量和索引"""
        print("\n[PairedDataset] 正在预计算patch索引...")
        
        for pair_idx, (src_path, tgt_path) in enumerate(self.paired_paths):
            try:
                # 读取两张配对图像
                src_img = Image.open(src_path).convert('RGB')
                tgt_img = Image.open(tgt_path).convert('RGB')

                # ========== 关键改动：先中心对齐，再pad到patch整数倍 ==========
                src_img, tgt_img = _center_crop_to_same(src_img, tgt_img)
                src_img = _pad_to_multiple(src_img, self.patch_size)
                tgt_img = _pad_to_multiple(tgt_img, self.patch_size)
                # ==========================================================

                w, h = src_img.size  # PIL返回(width, height)
                
                # 计算能切多少个patch
                # 这里使用滑动窗口，步长=patch_size（无重叠）
                num_rows = (h - self.patch_size) // self.patch_size + 1 if h >= self.patch_size else 1
                num_cols = (w - self.patch_size) // self.patch_size + 1 if w >= self.patch_size else 1
                
                for row in range(num_rows):
                    for col in range(num_cols):
                        self.patch_indices.append((pair_idx, row, col))
                
                if pair_idx == 0:
                    print(f"  示例图像尺寸: {w}x{h} -> 生成 {num_rows}x{num_cols}={num_rows*num_cols} 个patch")
                        
            except Exception as e:
                print(f"  警告: 无法处理 {src_path}: {e}")

    def __len__(self):
        """返回总patch数量"""
        return len(self.patch_indices)

    def __getitem__(self, idx):
        """
        返回一对配对的patch
        
        Returns:
            source_patch: 源染色patch, shape=(3, patch_size, patch_size)
            target_patch: 目标染色patch, shape=(3, patch_size, patch_size)
        """
        pair_idx, row, col = self.patch_indices[idx]
        src_path, tgt_path = self.paired_paths[pair_idx]
        
        # 读取配对图像
        src_image = Image.open(src_path).convert('RGB')
        tgt_image = Image.open(tgt_path).convert('RGB')
        
        # 提取相同位置的patch
        top = row * self.patch_size
        left = col * self.patch_size
        bottom = top + self.patch_size
        right = left + self.patch_size
        
        src_patch = src_image.crop((left, top, right, bottom))
        tgt_patch = tgt_image.crop((left, top, right, bottom))
        
        # 如果patch太小，用pad补齐到patch_size
        if src_patch.size != (self.patch_size, self.patch_size):
            padded_src = Image.new('RGB', (self.patch_size, self.patch_size))
            padded_src.paste(src_patch, (0, 0))
            src_patch = padded_src
            
            padded_tgt = Image.new('RGB', (self.patch_size, self.patch_size))
            padded_tgt.paste(tgt_patch, (0, 0))
            tgt_patch = padded_tgt
        
        # 应用相同的变换（注意：要确保随机变换在两张图上保持一致）
        # 这里简单处理，只用ToTensor
        if self.transform:
            # 为了保证source和target的随机变换一致，需要使用相同的随机种子
            # 这里暂时简化，不做随机变换
            src_patch = self.transform(src_patch)
            tgt_patch = self.transform(tgt_patch)
        
        return src_patch, tgt_patch


# 测试代码
if __name__ == '__main__':
    """
    测试数据集加载
    请确保你的数据结构如下：
    ./data_paired/
        source/
            img0001.png
            img0002.png
        target/
            img0001.png
            img0002.png
    """
    transform = T.Compose([T.ToTensor()])
    
    try:
        dataset = PairedDataset(
            root_dir='/root/autodl-tmp/Ki67/TrainValAB',
            patch_size=256,
            transform=transform
        )
        
        print(f"\n{'='*60}")
        print(f"数据集测试")
        print(f"  - 总patch数: {len(dataset)}")
        
        # 获取第一个样本
        src, tgt = dataset[0]
        print(f"\n样本测试:")
        print(f"  - Source patch shape: {src.shape}")  # 应该是 (3, 256, 256)
        print(f"  - Target patch shape: {tgt.shape}")  # 应该是 (3, 256, 256)
        print(f"  - Source value range: [{src.min():.3f}, {src.max():.3f}]")
        print(f"  - Target value range: [{tgt.min():.3f}, {tgt.max():.3f}]")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\n错误: {e}")
        print("\n请确保你的数据目录结构正确：")
        print("  ./data_paired/")
        print("      source/")
        print("          img0001.png")
        print("      target/")
        print("          img0001.png")