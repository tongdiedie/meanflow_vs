import os
import time
import argparse
import numpy as np
from PIL import Image

import torch
import torchvision
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from accelerate import Accelerator
from diffusers import AutoencoderKL

from meanflow import MeanFlow
from models.dit import MFDiT

def center_crop_arr(pil_image, image_size):
    # 来自你原脚本的中心裁剪实现
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True,
                        help="ImageFolder 根目录（如 /path/to/your_dataset/train）")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=min(8, (os.cpu_count() or 4)))
    parser.add_argument("--vae_name", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--num_classes", type=int, default=-1, help="类别数；-1 表示自动")
    parser.add_argument("--sample_classes", type=int, nargs="*", default=None,
                        help="采样可视化时要显示的类索引列表；默认自动取前若干类")
    args = parser.parse_args()

    accelerator = Accelerator(mixed_precision='fp16')
    device = accelerator.device

    os.makedirs('images', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    # ---------- 数据集 ----------
    transform = T.Compose([
        T.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(0.5, 0.5),  # 与 SD VAE 一致：[-1,1]
    ])
    trainset = ImageFolder(args.train_path, transform=transform)
    num_classes = len(trainset.classes) if args.num_classes == -1 else args.num_classes
    assert num_classes >= 1, "ImageFolder 必须至少包含 1 个子目录作为类别"

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0, pin_memory=True
    )

    # ---------- VAE ----------
    vae = AutoencoderKL.from_pretrained(args.vae_name).to(device)
    latent_factor = 0.18215  # 与 Stable Diffusion 一致的缩放

    # ---------- 模型 ----------
    model = MFDiT(
        input_size=32,        # 256 / patch_size(=2)^4 = 16x16 token -> 输出 32x32 latent 特征图
        patch_size=2,
        in_channels=4,        # VAE latent 通道=4
        dim=384,
        depth=8,
        num_heads=6,
        num_classes=num_classes,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    meanflow = MeanFlow(
        channels=4,
        image_size=32,
        num_classes=num_classes,
        normalizer=['mean_std', 0.0, 1/latent_factor],  # 关键：latent 的标准化
        flow_ratio=0.50,
        time_dist=['lognorm', -0.4, 1.0],
        cfg_ratio=0.10,
        cfg_scale=2.0,
        cfg_uncond='u'
    )

    model, vae, optimizer, train_loader = accelerator.prepare(model, vae, optimizer, train_loader)

    # 确保随时可保存
    model_module = model.module if hasattr(model, 'module') else model

    global_step = 0
    losses_accum = 0.0
    mse_accum = 0.0

    for e in range(args.epochs):
        model.train()
        for x, c in tqdm(train_loader, dynamic_ncols=True, desc=f"Epoch {e+1}/{args.epochs}"):
            x = x.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)

            with torch.no_grad():
                # 编码到 latent 域
                x = vae.encode(x).latent_dist.sample()

            loss, mse_val = meanflow.loss(model, x, c)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            losses_accum += loss.item()
            mse_accum += mse_val.item()

            if accelerator.is_main_process and (global_step % 1000 == 0):
                now = time.asctime(time.localtime(time.time()))
                lr = optimizer.param_groups[0]['lr']
                log = (f"{now}\nGlobal Step: {global_step}  "
                       f"Loss: {losses_accum/1000:.6f}  "
                       f"MSE_Loss: {mse_accum/1000:.6f}  "
                       f"LR: {lr:.6f}\n")
                with open('log.txt', 'a') as f:
                    f.write(log)
                losses_accum = 0.0
                mse_accum = 0.0

            if global_step % 1000 == 0:
                if accelerator.is_main_process:
                    model_module = model.module if hasattr(model, 'module') else model
                    with torch.no_grad():
                        # 选择可视化的类别
                        if args.sample_classes is not None and len(args.sample_classes) > 0:
                            classes = args.sample_classes
                        else:
                            classes = list(range(min(5, num_classes)))
                        z = meanflow.sample_each_class(model_module,
                                                       n_per_class=1,
                                                       classes=classes,
                                                       sample_steps=5,
                                                       device=device)
                        # 解码回像素域 [0,1]
                        z = vae.decode(z).sample
                        z = z * 0.5 + 0.5
                        grid_cols = len(classes)
                        log_img = make_grid(z, nrow=grid_cols)
                        save_image(log_img, f"images/step_{global_step}.png")
                accelerator.wait_for_everyone()
                model.train()

    if accelerator.is_main_process:
        model_module = model.module if hasattr(model, 'module') else model
        accelerator.save(model_module.state_dict(), f"checkpoints/step_{global_step}.pt")

if __name__ == "__main__":
    main()