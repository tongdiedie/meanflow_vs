"""
虚拟染色训练脚本（最小测试版本）

数据结构要求：
./data_paired/
    source/  (源染色，如HE)
        img0001.png
        img0002.png
    target/  (目标染色，如IHC)
        img0001.png
        img0002.png

注意：这是一个最小测试版本，只跑几个iteration来验证流程
"""

from models.dit import MFDiT
import torch
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from meanflow import MeanFlowTranslation
import os
from PairedDataset import PairedDataset
from metrics_eval import evaluate as eval_dir   # ★ 新增
import time


if __name__ == '__main__':
    print(f"\n{'='*80}")
    print(f"虚拟染色训练脚本 - 最小测试版本")
    print(f"{'='*80}\n")
    
    # ========= 配置参数 =========
    PATCH_SIZE = 256          # patch大小
    BATCH_SIZE = 8            # 小批量用于测试
    N_TEST_STEPS = 1000         # iteration测试
    SAMPLE_STEPS = 5          # 翻译时的采样步数
    VAL_SAVE_PRED_DIR = "outputs/val/pred"    # 预测结果
    VAL_SAVE_GT_DIR   = "outputs/val/gt"      # Ground Truth
    EVAL_EVERY_STEPS  = 50  # 例如每50步快速评估一次（你也可换成每个epoch）
    
    DIT_PATCH_SIZE = 16       # DiT内部的patch size
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}\n")
    
    os.makedirs('images', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs(VAL_SAVE_PRED_DIR, exist_ok=True)
    os.makedirs(VAL_SAVE_GT_DIR,   exist_ok=True)
    os.makedirs("metrics", exist_ok=True)
    
    # ========= 检查或创建测试数据 =========
    if not os.path.exists('/root/autodl-tmp/Ki67/data_paired/source') or not os.path.exists('/root/autodl-tmp/Ki67/data_paired/target'):
        print("⚠️  未找到数据目录，创建虚拟测试数据...")
    
    # ========= 加载配对数据集 =========
    transform = T.Compose([
        T.ToTensor(),  # 转换为[0,1]的tensor
    ])
    
    try:
        dataset = PairedDataset(
            root_dir='/root/autodl-tmp/Ki67/data_paired',
            patch_size=PATCH_SIZE,
            transform=transform,
            recursive=False
        )
    except FileNotFoundError as e:
        print(f"\n❌ 错误: {e}")
        print("\n请确保数据目录结构正确：")
        print("  ./data_paired/")
        print("      source/")
        print("          img0001.png")
        print("      target/")
        print("          img0001.png")
        exit(1)
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        drop_last=True,
        num_workers=0  # 测试时用0避免multiprocessing问题
    )
    
    # ========= 初始化模型 =========
    print(f"\n{'='*60}")
    print(f"[模型初始化]")
    print(f"{'='*60}")
    
    model = MFDiT(
        input_size=PATCH_SIZE,      # 256
        patch_size=DIT_PATCH_SIZE,  # 16 -> 256/16 = 16x16 tokens
        in_channels=3,              # RGB
        dim=384,                    # 嵌入维度
        depth=4,                    # 只用4层测试（原来是12层）
        num_heads=6,
        num_classes=None,           # 无条件翻译
    ).to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数:")
    print(f"  - 总参数量: {total_params:,}")
    print(f"  - 输入尺寸: {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"  - Patch数量: {(PATCH_SIZE//DIT_PATCH_SIZE)**2}")
    print(f"  - 嵌入维度: 384")
    print(f"  - Transformer深度: 4层")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    
    # ========= 初始化MeanFlow翻译器 =========
    meanflow = MeanFlowTranslation(
        channels=3,
        image_size=PATCH_SIZE,
        normalizer=['minmax', None, None],  # [0,1] -> [-1,1]
        flow_ratio=0.50,
        time_dist=['lognorm', -0.4, 1.0],
        cfg_ratio=0.0,      # 禁用CFG
        cfg_scale=None,     # 禁用CFG
        cfg_uncond='v',
        jvp_api='autograd'
    )
    
    # ========= 测试训练循环 =========
    print(f"\n{'='*60}")
    print(f"[开始测试训练]")
    print(f"  只跑 {N_TEST_STEPS} 个iteration验证流程")
    print(f"{'='*60}\n")
    
    model.train()
    
    dataloader_iter = iter(dataloader)
    
    for step in range(N_TEST_STEPS):
        print(f"\n{'#'*60}")
        print(f"Iteration {step+1}/{N_TEST_STEPS}")
        print(f"{'#'*60}")
        
        try:
            x_source, x_target = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            x_source, x_target = next(dataloader_iter)
        
        x_source = x_source.to(device)
        x_target = x_target.to(device)
        
        print(f"\n[数据批次]")
        print(f"  - Batch size: {x_source.shape[0]}")
        print(f"  - x_source: shape={x_source.shape}, range=[{x_source.min():.3f}, {x_source.max():.3f}]")
        print(f"  - x_target: shape={x_target.shape}, range=[{x_target.min():.3f}, {x_target.max():.3f}]")
        
        # === 前向传播 ===
        loss, mse_val = meanflow.loss(model, x_source, x_target, c=None)
        
        # === 反向传播 ===
        print(f"\n[反向传播]")
        optimizer.zero_grad()
        loss.backward()
        
        # 打印梯度信息
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        print(f"  - 梯度范数: {total_norm:.6f}")
        
        optimizer.step()
        print(f"  - 参数已更新")
        
        print(f"\n[损失总结]")
        print(f"  - Loss: {loss.item():.6f}")
        print(f"  - MSE: {mse_val.item():.6f}")
        
        # === 每5步做一次翻译测试 ===
        if (step + 1) % EVAL_EVERY_STEPS == 0:
            print(f"\n{'='*60}\n[EVAL] 运行快速验证评估（保存若干patch）\n{'='*60}")
            model.eval()
            print(f"[翻译测试] Step {step+1}")
            print(f"{'='*60}")
            
            model.eval()
            with torch.no_grad():
                # 取一个小批次做验证推理并落盘（同名）
                b = min(8, x_source.size(0))  # 取前8张
                test_src = x_source[:b]
                test_tgt = x_target[:b]
                
                print(f"\n测试输入:")
                print(f"  - test_source: shape={test_src.shape}")

                # 进行翻译
                translated = meanflow.translate(
                    model, 
                    test_src, 
                    sample_steps=SAMPLE_STEPS,
                    device=device
                )
                
                print(f"测试输出:")
                print(f"  - translated: shape={translated.shape}, range=[{translated.min():.3f}, {translated.max():.3f}]")

                # 保存：pred 到 pred_dir，gt 到 gt_dir，文件名统一
                for i in range(b):
                    name = f"step{step+1:05d}_idx{i:02d}.png"
                    save_image(test_tgt[i].cpu(), os.path.join(VAL_SAVE_GT_DIR, name))
                    save_image(translated[i].cpu(), os.path.join(VAL_SAVE_PRED_DIR, name))
                
                # 保存对比图：[source | translated | ground_truth]
                comparison = torch.cat([
                    test_src.cpu(),
                    translated.cpu(),
                    test_tgt.cpu()
                ], dim=0)
                
                grid = make_grid(comparison, nrow=3)
                save_path = f"images/test_step_{step+1}.png"
                save_image(grid, save_path)
                print(f"  - 已保存对比图到: {save_path}")
                print(f"    格式: [源染色 | 翻译结果 | 目标染色]")

                # 调用评估,并保存每张图的指标到csv
                try:
                    stamp = time.strftime("%Y%m%d-%H%M%S")
                    csv_path = f"metrics/per_image_{stamp}_step{step+1:05d}.csv"
                    summary = eval_dir(VAL_SAVE_PRED_DIR, VAL_SAVE_GT_DIR, csv_out=csv_path)
                    print(f"[EVAL@{step+1}] "
                        f"FID={summary['FID']:.3f}  "
                        f"KID={summary['KID_mean']:.5f}±{summary['KID_std']:.5f}  "
                        f"PSNR={summary['PSNR_mean']:.3f}  "
                        f"SSIM={summary['SSIM_mean']:.3f}  "
                        f"LPIPS={summary['LPIPS_mean']:.4f}")
                except Exception as e:
                    print(f"[EVAL] 评估失败：{e}")

            model.train()
    
    # ========= 保存模型 =========
    print(f"\n{'='*60}")
    print(f"[保存模型]")
    ckpt_path = f"checkpoints/test_model.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"  - 已保存模型到: {ckpt_path}")
    print(f"{'='*60}\n")
    
    print(f"\n{'='*80}")
    print(f"✓ 测试完成！")
    print(f"{'='*80}")
    print(f"\n检查结果:")
    print(f"  1. 查看 images/ 文件夹中的翻译对比图")
    print(f"  2. 格式: [源染色 | 翻译结果 | 目标染色]")
    print(f"  3. 如果三张图有明显差异，说明模型正在学习翻译")
    print(f"\n下一步:")
    print(f"  - 如果流程正常，可以增加训练步数（修改N_TEST_STEPS）")
    print(f"  - 使用真实的配对染色数据替换虚拟数据")
    print(f"  - 调整模型深度（depth）和维度（dim）以提升性能")
    print(f"{'='*80}\n")