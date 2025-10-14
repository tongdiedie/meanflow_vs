import os
import time
import argparse

import torch
import torchvision
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator

from meanflow import MeanFlow
from models.dit import MFDiT

def build_transforms(image_size: int, in_channels: int):
    # 基础增强：随机翻转 + Resize/CenterCrop + ToTensor
    # 对灰度图自动转为 1 通道；对彩色图为 3 通道
    ops = [
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.RandomHorizontalFlip(),
    ]
    if in_channels == 1:
        ops.append(T.Grayscale(num_output_channels=1))
    else:
        ops.append(T.Lambda(lambda x: x.convert("RGB")))
    ops.append(T.ToTensor())
    return T.Compose(ops)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="./data",
                        help="选择数据集来源")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="imagefolder/cifar/mnist 的根目录")
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--num_classes", type=int, default=-1,
                        help="类别数；-1 表示自动从数据集中推断")
    parser.add_argument("--in_channels", type=int, default=-1,
                        help="输入通道；-1 表示自动（MNIST=1，其它=3）")
    parser.add_argument("--n_steps", type=int, default=200_000)
    parser.add_argument("--log_step", type=int, default=500)
    parser.add_argument("--sample_step", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=min(8, (os.cpu_count() or 4)))
    args = parser.parse_args()

    accelerator = Accelerator(mixed_precision='fp16')
    device = accelerator.device

    os.makedirs('images', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    # ---------- 构建数据集 ----------
    if args.dataset == "cifar10":
        t = T.Compose([T.Resize(args.image_size),
                       T.RandomHorizontalFlip(),
                       T.ToTensor()])
        trainset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, "cifar"),
                                                train=True, download=True, transform=t)
        inferred_nc = 3
        inferred_classes = 10

    elif args.dataset == "mnist":
        t = T.Compose([T.Resize((args.image_size, args.image_size)),
                       T.ToTensor()])
        trainset = torchvision.datasets.MNIST(root=os.path.join(args.data_dir, "mnist"),
                                              train=True, download=True, transform=t)
        inferred_nc = 1
        inferred_classes = 10

    else:  # imagefolder（自定义数据集）
        # auto infer channels by peeking one sample: 默认当作 RGB（3 通道），也支持通过 --in_channels 指定
        tmp_in_channels = 3 if args.in_channels == -1 else args.in_channels
        t = build_transforms(args.image_size, tmp_in_channels)
        trainset = ImageFolder(args.data_dir, transform=t)
        inferred_nc = tmp_in_channels
        inferred_classes = len(trainset.classes)

    in_channels = inferred_nc if args.in_channels == -1 else args.in_channels
    num_classes = inferred_classes if args.num_classes == -1 else args.num_classes

    assert num_classes >= 1, "类别数必须 >= 1（ImageFolder 至少需要一个子文件夹作为类别）"
    assert in_channels in (1, 3), "仅支持 in_channels 为 1 或 3"

    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True
    )

    # ---------- 构建模型 ----------
    model = MFDiT(
        input_size=args.image_size,
        patch_size=2,
        in_channels=in_channels,
        dim=384,
        depth=12,
        num_heads=6,
        num_classes=num_classes,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    # 像素域：minmax 归一化（[-1,1] 之间）
    meanflow = MeanFlow(
        channels=in_channels,
        image_size=args.image_size,
        num_classes=num_classes,
        normalizer=['minmax', None, None],
        flow_ratio=0.50,
        time_dist=['lognorm', -0.4, 1.0],
        cfg_ratio=0.10,
        cfg_scale=2.0,
        cfg_uncond='u'
    )

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    # 确保随时可保存
    model_module = model.module if hasattr(model, 'module') else model

    global_step = 0
    losses_accum = 0.0
    mse_accum = 0.0

    with tqdm(range(args.n_steps), dynamic_ncols=True) as pbar:
        pbar.set_description("Training")
        model.train()
        loader_iter = iter(train_loader)

        for _ in pbar:
            try:
                x, c = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                x, c = next(loader_iter)

            x = x.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)

            loss, mse_val = meanflow.loss(model, x, c)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            losses_accum += loss.item()
            mse_accum += mse_val.item()

            if accelerator.is_main_process and (global_step % args.log_step == 0):
                now = time.asctime(time.localtime(time.time()))
                lr = optimizer.param_groups[0]['lr']
                log = (f"{now}\nGlobal Step: {global_step}  "
                       f"Loss: {losses_accum/args.log_step:.6f}  "
                       f"MSE_Loss: {mse_accum/args.log_step:.6f}  "
                       f"LR: {lr:.6f}\n")
                with open('log.txt', 'a') as f:
                    f.write(log)
                losses_accum = 0.0
                mse_accum = 0.0

            if global_step % args.sample_step == 0:
                if accelerator.is_main_process:
                    model_module = model.module if hasattr(model, 'module') else model
                    n_per_class = 1
                    # 若类别数 < 10，则使用实际类别数；否则显示前 10 个类别的样本
                    classes_to_show = list(range(min(10, num_classes)))
                    with torch.no_grad():
                        z = meanflow.sample_each_class(model_module,
                                                       n_per_class=n_per_class,
                                                       classes=classes_to_show,
                                                       sample_steps=5,
                                                       device=device)
                    grid_cols = len(classes_to_show)
                    log_img = make_grid(z, nrow=grid_cols)
                    save_image(log_img, f"images/step_{global_step}.png")
                accelerator.wait_for_everyone()
                model.train()

    if accelerator.is_main_process:
        # 再次保证已定义
        model_module = model.module if hasattr(model, 'module') else model
        accelerator.save(model_module.state_dict(), f"checkpoints/step_{global_step}.pt")

if __name__ == '__main__':
    main()