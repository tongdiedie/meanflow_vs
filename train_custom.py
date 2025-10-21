from models.dit import MFDiT
import torch
import torchvision
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from meanflow import MeanFlow
from accelerate import Accelerator
import time
import os
from CustomDataset import CustomDataset

if __name__ == '__main__':
    # ========= 原始图像大小和目标patch大小 =========
    ORIGINAL_SIZE = 1024  # 原始数据集大小
    PATCH_SIZE_DATA = 256  # 切分成的patch大小
    TARGET_SIZE = 256  # 送入模型的大小（与PATCH_SIZE_DATA相同）

    # ========= 批大小 =========
    BATCH_SIZE = 32  # 256的patch可以用较大的batch_size

    # ========= DiT patch 设置 =========
    DIT_PATCH_SIZE = 16  # 模型内部的patch_size（256/16=16 tokens）

    n_steps = 200_000
    os.makedirs('images', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    accelerator = Accelerator(mixed_precision='fp16')

    # === 使用自己的数据集，切分成256*256 ===
    dataset = CustomDataset(
        root_dir='./data',
        patch_size=PATCH_SIZE_DATA,  # 切分成256*256的块
        transform=T.Compose([
            # 不需要Resize，因为已经切分成256*256
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),  # 可选：添加更多增强
            T.ToTensor()
        ])
    )


    def cycle(iterable):
        while True:
            for i in iterable:
                yield i


    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=8, pin_memory=True
    )
    train_dataloader = cycle(train_dataloader)

    # === 无条件模型 ===
    model = MFDiT(
        input_size=TARGET_SIZE,
        patch_size=DIT_PATCH_SIZE,
        in_channels=3,
        dim=384,
        depth=12,
        num_heads=6,
        num_classes=None,
    ).to(accelerator.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)

    # === MeanFlow配置 ===
    meanflow = MeanFlow(
        channels=3,
        image_size=TARGET_SIZE,
        num_classes=None,
        flow_ratio=0.50,
        time_dist=['lognorm', -0.4, 1.0],
        cfg_ratio=0.10,
        cfg_scale=2.0,
        cfg_uncond='u'
    )

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    global_step = 0
    losses = 0.0
    mse_losses = 0.0

    log_step = 500
    sample_step = 1000

    with tqdm(range(n_steps), dynamic_ncols=True) as pbar:
        pbar.set_description("Training (unconditional, 256x256 patches)")
        model.train()
        for step in pbar:
            x = next(train_dataloader).to(accelerator.device)
            loss, mse_val = meanflow.loss(model, x, c=None)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            losses += loss.item()
            mse_losses += mse_val.item()

            if accelerator.is_main_process and global_step % log_step == 0:
                current_time = time.asctime(time.localtime(time.time()))
                log_message = (
                    f'{current_time}\n'
                    f'Global Step: {global_step}    '
                    f'Loss: {losses / log_step:.6f}    MSE_Loss: {mse_losses / log_step:.6f}    '
                    f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}\n'
                )
                with open('log.txt', mode='a') as n:
                    n.write(log_message)
                losses = 0.0
                mse_losses = 0.0

            if global_step % sample_step == 0 and accelerator.is_main_process:
                model_module = model.module if hasattr(model, 'module') else model
                z = meanflow.sample_uncond(model_module, n_images=16, sample_steps=5, device=accelerator.device)
                log_img = make_grid(z, nrow=4)
                save_image(log_img, f"images/step_{global_step}.png")
                accelerator.wait_for_everyone()
                model.train()

    if accelerator.is_main_process:
        model_module = model.module if hasattr(model, 'module') else model
        accelerator.save(model_module.state_dict(), f"checkpoints/step_{global_step}.pt")
