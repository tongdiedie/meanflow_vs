import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


class CustomDataset(Dataset):
    def __init__(self, root_dir, patch_size=256, transform=None, recursive=True):
        """
        Args:
            root_dir: 图像文件夹路径
            patch_size: 切分块的大小 (256, 512, etc.)
            transform: 数据增强变换
            recursive: 是否递归搜索子文件夹
        """
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.transform = transform
        self.image_paths = []
        self.patches_per_image = []  # 记录每张图像对应的patch数量

        if recursive:
            for root, _, files in os.walk(root_dir):
                for f in files:
                    ext = os.path.splitext(f)[1].lower()
                    if ext in IMG_EXTS:
                        self.image_paths.append(os.path.join(root, f))
        else:
            self.image_paths = [
                os.path.join(root_dir, f)
                for f in os.listdir(root_dir)
                if os.path.splitext(f)[1].lower() in IMG_EXTS
            ]

        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in {root_dir}")

        # 预计算每张图像能生成的patch数量
        self._precompute_patch_info()

    def _precompute_patch_info(self):
        """预计算patch数量和索引映射"""
        self.patch_indices = []  # [(img_idx, row_idx, col_idx), ...]

        for img_idx, img_path in enumerate(self.image_paths):
            try:
                image = Image.open(img_path).convert('RGB')
                h, w = image.size[::-1]  # PIL返回(w, h)，转为(h, w)

                # 计算能切多少个patch
                num_rows = (h - self.patch_size) // self.patch_size + 1 if h >= self.patch_size else 1
                num_cols = (w - self.patch_size) // self.patch_size + 1 if w >= self.patch_size else 1

                for row in range(num_rows):
                    for col in range(num_cols):
                        self.patch_indices.append((img_idx, row, col))
            except Exception as e:
                print(f"Warning: Could not process {img_path}: {e}")

    def __len__(self):
        return len(self.patch_indices)

    def __getitem__(self, idx):
        img_idx, row, col = self.patch_indices[idx]
        img_path = self.image_paths[img_idx]

        image = Image.open(img_path).convert('RGB')

        # 提取patch区域
        top = row * self.patch_size
        left = col * self.patch_size
        bottom = top + self.patch_size
        right = left + self.patch_size

        # 处理边界情况（如果原图不能被patch_size整除）
        patch = image.crop((left, top, right, bottom))

        # 如果patch太小，用pad补齐到patch_size
        if patch.size != (self.patch_size, self.patch_size):
            padded_patch = Image.new('RGB', (self.patch_size, self.patch_size))
            padded_patch.paste(patch, (0, 0))
            patch = padded_patch

        if self.transform:
            patch = self.transform(patch)

        return patch
