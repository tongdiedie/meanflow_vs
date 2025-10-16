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
    # ========= 选择训练分辨率 =========
    TARGET_SIZE = 1024         # 可改为 256
    # ========= 选择批大小（按显存调）=========
    BATCH_SIZE_256 = 16       # 示例值，按显卡调大/调小
    BATCH_SIZE_1024 = 2       # 1024 会非常吃显存，建议先小批量
    batch_size = BATCH_SIZE_256 if TARGET_SIZE == 256 else BATCH_SIZE_1024

    # ========= DiT patch 设置（需整除 TARGET_SIZE）=========
    # 建议：256 用 16；1024 用 32（token 数为 (H/patch)*(W/patch)）
    PATCH_SIZE = 16 if TARGET_SIZE == 256 else 32

    n_steps = 200_000
    os.makedirs('images', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    accelerator = Accelerator(mixed_precision='fp16')

    # === 使用你自己的数据集（无标签）===
    # 建议的 transforms：Resize 到目标分辨率 + 随机翻转 + ToTensor
    dataset = CustomDataset(
        root_dir='./data',
        transform=T.Compose([
            T.Resize((TARGET_SIZE, TARGET_SIZE)),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ])
    )

    def cycle(iterable):
        while True:
            for i in iterable:
                yield i

    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=True
    )
    train_dataloader = cycle(train_dataloader)

    # === 无条件模型：num_classes=None ===
    model = MFDiT(
        input_size=TARGET_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=3,        # 若是灰度图改成 1
        dim=384,              # 可按算力调
        depth=12,
        num_heads=6,
        num_classes=None,     # ★ 无分类
    ).to(accelerator.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)

    # === 无条件 MeanFlow：关闭 CFG，num_classes=None ===
    meanflow = MeanFlow(
        channels=3,                 # 若灰度图改成 1
        image_size=TARGET_SIZE,
        num_classes=None,           # ★ 无分类
        flow_ratio=0.50,
        time_dist=['lognorm', -0.4, 1.0],
        cfg_ratio=0.10,             # TODO 关闭 CFG 掩码（不再丢弃条件）设置None
        cfg_scale=2.0,             # TODO 不做引导混合 设置None
        cfg_uncond='u'
    )

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    global_step = 0
    losses = 0.0
    mse_losses = 0.0

    log_step = 500
    sample_step = 1000

    with tqdm(range(n_steps), dynamic_ncols=True) as pbar:
        pbar.set_description("Training (unconditional)")
        model.train()
        for step in pbar:
            # === dataloader 只返回图像 ===
            x = next(train_dataloader).to(accelerator.device)

            # === 无条件训练，c=None ===
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
                # === 无条件采样（见第 3 节）===
                z = meanflow.sample_uncond(model_module, n_images=16, sample_steps=5, device=accelerator.device)
                log_img = make_grid(z, nrow=4)
                save_image(log_img, f"images/step_{global_step}.png")
                accelerator.wait_for_everyone()
                model.train()

    if accelerator.is_main_process:
        model_module = model.module if hasattr(model, 'module') else model
        accelerator.save(model_module.state_dict(), f"checkpoints/step_{global_step}.pt")