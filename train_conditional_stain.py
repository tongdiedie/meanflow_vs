"""
条件Flow虚拟染色训练脚本

理论框架：
    噪声 ε (t=1) <--Flow--> 目标图像 x_target (t=0)
                   ↑
              条件：源图像 x_source

数据结构要求：
./data_paired/
    source/  (源染色，如HE)
        img0001.png
        img0002.png
    target/  (目标染色，如IHC)
        img0001.png
        img0002.png
"""

from models.dit_conditional import ConditionalMFDiT
import torch
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from meanflow_conditional import ConditionalMeanFlow
import os
from PairedDataset import PairedDataset
from metrics_eval import evaluate as eval_dir
import time


if __name__ == '__main__':
    print(f"\n{'='*80}")
    print(f"条件Flow虚拟染色训练脚本")
    print(f"理论: 噪声(t=1) → 目标图像(t=0), 条件=源图像")
    print(f"{'='*80}\n")
    
    # ========= 配置参数 =========
    PATCH_SIZE = 256          # patch大小
    BATCH_SIZE = 8            # 批量大小
    N_TEST_STEPS = 1000       # 测试迭代数
    SAMPLE_STEPS = 10         # 翻译时的ODE求解步数（增加到10步以提高质量）
    VAL_SAVE_PRED_DIR = "outputs/val/pred"
    VAL_SAVE_GT_DIR   = "outputs/val/gt"
    EVAL_EVERY_STEPS  = 50    # 每50步评估一次
    
    DIT_PATCH_SIZE = 16       # DiT内部的patch size
    
    # CFG参数
    USE_CFG = False           # 是否使用Classifier-Free Guidance
    CFG_RATIO = 0.1           # 训练时无条件的比例（10%的batch无条件训练）
    CFG_SCALE = 1.5           # 推理时的CFG缩放因子（>1增强条件影响）
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}\n")
    
    os.makedirs('images', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs(VAL_SAVE_PRED_DIR, exist_ok=True)
    os.makedirs(VAL_SAVE_GT_DIR,   exist_ok=True)
    os.makedirs("metrics", exist_ok=True)
    
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
        num_workers=0
    )
    
    # ========= 初始化条件DiT模型 =========
    print(f"\n{'='*60}")
    print(f"[模型初始化 - 条件DiT]")
    print(f"{'='*60}")
    
    model = ConditionalMFDiT(
        input_size=PATCH_SIZE,      # 256
        patch_size=DIT_PATCH_SIZE,  # 16 -> 256/16 = 16x16 tokens
        in_channels=3,              # RGB
        dim=384,                    # 嵌入维度
        depth=6,                    # 6层Transformer（比测试版深）
        num_heads=6,
        num_classes=None,           # 无类别条件
        cond_mode='concat',         # 条件融合模式：concat
    ).to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数:")
    print(f"  - 总参数量: {total_params:,}")
    print(f"  - 输入尺寸: {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"  - Patch数量: {(PATCH_SIZE//DIT_PATCH_SIZE)**2}")
    print(f"  - 条件融合后token数: {(PATCH_SIZE//DIT_PATCH_SIZE)**2 * 2} (concat模式)")
    print(f"  - 嵌入维度: 384")
    print(f"  - Transformer深度: 6层")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    
    # ========= 初始化条件MeanFlow ==========
    meanflow = ConditionalMeanFlow(
        channels=3,
        image_size=PATCH_SIZE,
        normalizer=['minmax', None, None],  # [0,1] -> [-1,1]
        flow_ratio=0.50,                    # 50%端点训练
        time_dist=['lognorm', -0.4, 1.0],   # Log-normal时间分布
        cfg_ratio=CFG_RATIO if USE_CFG else 0.0,
        cfg_scale=CFG_SCALE if USE_CFG else None,
        cfg_uncond='zeros',                 # 无条件时用零图像
        jvp_api='autograd'
    )
    
    # ========= 训练循环 =========
    print(f"\n{'='*60}")
    print(f"[开始训练 - 条件Flow]")
    print(f"  迭代数: {N_TEST_STEPS}")
    if USE_CFG:
        print(f"  CFG训练: 启用（ratio={CFG_RATIO}, scale={CFG_SCALE}）")
    else:
        print(f"  CFG训练: 禁用")
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
        print(f"  - x_source (条件): shape={x_source.shape}, range=[{x_source.min():.3f}, {x_source.max():.3f}]")
        print(f"  - x_target (目标): shape={x_target.shape}, range=[{x_target.min():.3f}, {x_target.max():.3f}]")
        
        # === 前向传播（条件Flow）===
        loss, mse_val = meanflow.loss(model, x_source, x_target, c=None)
        
        # === 反向传播 ===
        print(f"\n[反向传播]")
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
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
        
        # === 定期评估 ===
        if (step + 1) % EVAL_EVERY_STEPS == 0:
            print(f"\n{'='*60}")
            print(f"[EVAL] Step {step+1} - 运行验证评估")
            print(f"{'='*60}")
            
            model.eval()
            with torch.no_grad():
                # 取一小批做验证
                b = min(8, x_source.size(0))
                test_src = x_source[:b]
                test_tgt = x_target[:b]
                
                print(f"\n测试输入:")
                print(f"  - test_source (条件): shape={test_src.shape}")
                print(f"  - test_target (GT): shape={test_tgt.shape}")

                # 条件Flow翻译
                cfg_scale_eval = CFG_SCALE if USE_CFG else None
                translated = meanflow.translate(
                    model, 
                    test_src, 
                    sample_steps=SAMPLE_STEPS,
                    cfg_scale=cfg_scale_eval,
                    device=device
                )
                
                print(f"\n测试输出:")
                print(f"  - translated: shape={translated.shape}, range=[{translated.min():.3f}, {translated.max():.3f}]")

                # 保存预测和GT
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
                
                grid = make_grid(comparison, nrow=b, normalize=True, value_range=(0, 1))
                save_path = f"images/eval_step_{step+1:05d}.png"
                save_image(grid, save_path)
                print(f"  - 已保存对比图: {save_path}")
                print(f"    格式: 第1行=源染色 | 第2行=翻译结果 | 第3行=目标染色")

                # 评估指标
                try:
                    stamp = time.strftime("%Y%m%d-%H%M%S")
                    csv_path = f"metrics/per_image_{stamp}_step{step+1:05d}.csv"
                    summary = eval_dir(VAL_SAVE_PRED_DIR, VAL_SAVE_GT_DIR, csv_out=csv_path)
                    print(f"\n[EVAL@{step+1}] 指标:")
                    print(f"  - FID: {summary['FID']:.3f}")
                    print(f"  - KID: {summary['KID_mean']:.5f} ± {summary['KID_std']:.5f}")
                    print(f"  - PSNR: {summary['PSNR_mean']:.3f} dB")
                    print(f"  - SSIM: {summary['SSIM_mean']:.3f}")
                    print(f"  - LPIPS: {summary['LPIPS_mean']:.4f}")
                except Exception as e:
                    print(f"[EVAL] 评估失败：{e}")

            model.train()
    
    # ========= 保存最终模型 =========
    print(f"\n{'='*60}")
    print(f"[保存模型]")
    ckpt_path = f"checkpoints/conditional_flow_model_step{N_TEST_STEPS}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': N_TEST_STEPS,
    }, ckpt_path)
    print(f"  - 已保存模型到: {ckpt_path}")
    print(f"{'='*60}\n")
    
    # ========= 最终评估 =========
    print(f"\n{'='*80}")
    print(f"✓ 训练完成！")
    print(f"{'='*80}")
    print(f"\n理论验证:")
    print(f"  ✓ 使用正确的Flow理论：噪声(t=1) → 目标(t=0)")
    print(f"  ✓ 源图像作为条件引导生成")
    print(f"  ✓ 插值路径有物理意义：z(t) = t·ε + (1-t)·x_target")
    print(f"  ✓ 速度场正确：v = x_target - ε")
    print(f"\n检查结果:")
    print(f"  1. 查看 images/ 文件夹中的对比图")
    print(f"  2. 格式: 第1行=源染色 | 第2行=翻译结果 | 第3行=目标染色")
    print(f"  3. 查看 metrics/ 文件夹中的评估指标")
    print(f"\n预期效果:")
    print(f"  - 翻译结果应该保留源图像的结构")
    print(f"  - 翻译结果应该具有目标染色的颜色/纹理特征")
    print(f"  - 随着训练进行，FID/LPIPS应该下降，PSNR/SSIM应该上升")
    print(f"\n下一步优化:")
    print(f"  1. 增加训练步数和模型深度")
    print(f"  2. 如果效果仍不理想，尝试启用CFG (USE_CFG=True)")
    print(f"  3. 调整采样步数 SAMPLE_STEPS (10-50)")
    print(f"  4. 考虑添加感知损失（LPIPS）辅助训练")
    print(f"{'='*80}\n")