import os
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, recursive=True):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []

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

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image