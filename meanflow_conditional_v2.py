import torch
import torch.nn.functional as F
from einops import rearrange
from functools import partial
import numpy as np


class Normalizer:
    """
    归一化器：用于将图像归一化到模型训练空间
    - minmax: [0,1] -> [-1,1] (原始像素空间)
    - mean_std: 使用均值和标准差归一化（VAE latent空间）
    """
    def __init__(self, mode='minmax', mean=None, std=None):
        assert mode in ['minmax', 'mean_std'], "mode must be 'minmax' or 'mean_std'"
        self.mode = mode
        print(f"\n[Normalizer] 模式: {mode}")

        if mode == 'mean_std':
            if mean is None or std is None:
                raise ValueError("mean and std must be provided for 'mean_std' mode")
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)
            print(f"  - Mean: {mean}, Std: {std}")

    @classmethod
    def from_list(cls, config):
        """从配置列表创建: [mode, mean, std]"""
        mode, mean, std = config
        return cls(mode, mean, std)

    def norm(self, x):
        """归一化：原始空间 -> 训练空间"""
        if self.mode == 'minmax':
            return x * 2 - 1  # [0,1] -> [-1,1]
        elif self.mode == 'mean_std':
            return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def unnorm(self, x):
        """反归一化：训练空间 -> 原始空间"""
        if self.mode == 'minmax':
            x = x.clip(-1, 1)
            return (x + 1) * 0.5  # [-1,1] -> [0,1]
        elif self.mode == 'mean_std':
            return x * self.std.to(x.device) + self.mean.to(x.device)


def stopgrad(x):
    """停止梯度传播"""
    return x.detach()


def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    """
    自适应L2损失: sg(w) * ||Δ||_2^2, where w = 1 / (||Δ||^2 + c)^p, p = 1 - γ
    
    这个损失函数会自动调整每个样本的权重：
    - 误差大的样本权重小（避免outlier主导训练）
    - 误差小的样本权重大（精细化调整）
    
    Args:
        error: 误差张量, shape=(B, C, H, W)
        gamma: 幂次参数
        c: 稳定性常数
    Returns:
        标量损失
    """
    delta_sq = torch.mean(error ** 2, dim=(1, 2, 3), keepdim=False)  # (B,)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)  # 自适应权重
    loss = delta_sq  # ||Δ||^2
    return (stopgrad(w) * loss).mean()


class ConditionalMeanFlow:
    """
    条件MeanFlow用于虚拟染色（正确的理论框架）
    
    ===== 核心理论改变 =====
    原来的错误思路：源图像 (t=1) <--流--> 目标图像 (t=0)  ❌
    
    正确的条件生成思路（模仿Diffusion）：
    噪声 (t=1) <--流--> 目标图像 (t=0)
          ↑
      条件：源图像
    
    ===== Flow ODE =====
    dz/dt = v(z, t, c)
    
    其中：
    - z(t): 插值路径，z(1)=噪声ε, z(0)=x_target
    - c: 条件（源图像x_source）
    - v: 条件速度场
    
    ===== 插值路径 =====
    z(t) = t·ε + (1-t)·x_target
    
    物理意义：
    - t=1: 纯噪声（起点）
    - t=0: 目标图像（终点）
    - t∈(0,1): 噪声逐渐被目标图像替换
    
    这是有意义的，因为我们可以往任何图像上添加噪声！
    
    ===== 速度场 =====
    v_true = x_target - ε
    （从噪声到目标图像的方向）
    
    ===== 模型预测 =====
    u(z, t, r, c) ≈ E[v_true | z(t), x_source]
    
    模型学习：给定当前状态z(t)和源图像x_source，预测平均速度场
    """
    def __init__(
        self,
        channels=3,
        image_size=256,
        normalizer=['minmax', None, None],
        # mean flow settings
        flow_ratio=0.50,
        # time distribution
        time_dist=['lognorm', -0.4, 1.0],
        # cfg settings（虚拟染色建议禁用）
        cfg_ratio=0.0,
        cfg_scale=None,
        cfg_uncond='zeros',  # 'zeros' or 'v'
        jvp_api='autograd',
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        
        print(f"\n{'='*60}")
        print(f"[ConditionalMeanFlow] 条件Flow虚拟染色模型")
        print(f"  理论框架: 噪声(t=1) → 目标图像(t=0), 条件=源图像")
        print(f"  - 图像尺寸: {image_size}x{image_size}")
        print(f"  - 通道数: {channels}")
        print(f"  - Flow ratio: {flow_ratio} (端点训练比例)")
        print(f"  - Time distribution: {time_dist}")
        print(f"  - CFG: {'禁用' if cfg_scale is None else f'启用(scale={cfg_scale})'}")
        print(f"{'='*60}\n")

        self.normer = Normalizer.from_list(normalizer)

        self.flow_ratio = flow_ratio
        self.time_dist = time_dist
        self.cfg_ratio = cfg_ratio
        self.w = cfg_scale

        self.cfg_uncond = cfg_uncond
        self.jvp_api = jvp_api

        assert jvp_api in ['funtorch', 'autograd'], "jvp_api must be 'funtorch' or 'autograd'"
        if jvp_api == 'funtorch':
            self.jvp_fn = torch.func.jvp
            self.create_graph = False
        elif jvp_api == 'autograd':
            self.jvp_fn = torch.autograd.functional.jvp
            self.create_graph = True

    def sample_t_r(self, batch_size, device):
        """
        采样时间对(t, r)，其中 t >= r
        
        物理意义：
        - t=1: 完全是噪声
        - t=0: 完全是目标染色
        - t in (0,1): 噪声逐渐被目标图像替换
        
        - r: 参考时间点，用于计算时间间隔内的平均速度
        - 当r=t时，退化为端点的瞬时速度
        
        Returns:
            t: shape=(batch_size,), 较大的时间值
            r: shape=(batch_size,), 较小的时间值
        """
        if self.time_dist[0] == 'uniform':
            samples = np.random.rand(batch_size, 2).astype(np.float32)
        elif self.time_dist[0] == 'lognorm':
            mu, sigma = self.time_dist[-2], self.time_dist[-1]
            normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
            samples = 1 / (1 + np.exp(-normal_samples))  # Sigmoid变换

        # 确保 t >= r
        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])
        
        # 按flow_ratio概率让r=t（端点训练）
        num_selected = int(self.flow_ratio * batch_size)
        indices = np.random.permutation(batch_size)[:num_selected]
        r_np[indices] = t_np[indices]

        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)
        
        return t, r

    def loss(self, model, x_source, x_target, c=None):
        """
        条件Flow的训练损失
        
        ===== 理论框架 =====
        1. 采样噪声: ε ~ N(0, I)
        2. 采样时间: t, r ~ p(t, r)
        3. 插值状态: z(t) = t·ε + (1-t)·x_target
        4. 真实速度场: v = x_target - ε
        5. 条件: c = x_source
        6. 模型预测: u(z, t, r, c) ≈ E[v | z(t), c]
        7. 目标值: u_tgt = v - (t-r)·∂u/∂t
        8. 损失: ||u - u_tgt||^2
        
        Args:
            model: DiT模型，签名为 model(z, t, r, c)
            x_source: 源染色图像 (HE), shape=(B, C, H, W)
            x_target: 目标染色图像 (Ki67), shape=(B, C, H, W)
            c: 额外条件（如类别标签，可选）
        
        Returns:
            loss: 标量损失
            mse_val: MSE值（用于监控）
        """
        batch_size = x_source.shape[0]
        device = x_source.device
        
        print(f"\n[Loss计算 - 条件Flow]")
        print(f"  输入shape:")
        print(f"    - x_source: {x_source.shape}  (源染色，作为条件)")
        print(f"    - x_target: {x_target.shape}  (目标染色)")
        
        # ===== 步骤1: 采样时间对(t, r) =====
        t, r = self.sample_t_r(batch_size, device)
        print(f"  时间采样:")
        print(f"    - t: shape={t.shape}, range=[{t.min():.3f}, {t.max():.3f}]")
        print(f"    - r: shape={r.shape}, range=[{r.min():.3f}, {r.max():.3f}]")
        print(f"    - (t-r): mean={((t-r).mean()):.3f}")
        
        t_ = rearrange(t, "b -> b 1 1 1").detach().clone()  # shape=(B, 1, 1, 1)
        r_ = rearrange(r, "b -> b 1 1 1").detach().clone()

        # ===== 步骤2: 归一化 =====
        x_source = self.normer.norm(x_source)  # [0,1] -> [-1,1]
        x_target = self.normer.norm(x_target)
        print(f"  归一化后:")
        print(f"    - x_source: range=[{x_source.min():.3f}, {x_source.max():.3f}]")
        print(f"    - x_target: range=[{x_target.min():.3f}, {x_target.max():.3f}]")

        # ===== 步骤3: 采样噪声 ε ~ N(0, I) =====
        epsilon = torch.randn_like(x_target)
        print(f"  采样噪声:")
        print(f"    - ε: shape={epsilon.shape}, range=[{epsilon.min():.3f}, {epsilon.max():.3f}]")

        # ===== 步骤4: 插值状态 z(t) = t·ε + (1-t)·x_target =====
        # 关键改变：现在是噪声和目标图像的插值
        z = t_ * epsilon + (1 - t_) * x_target
        
        print(f"  插值状态:")
        print(f"    - z: shape={z.shape}, range=[{z.min():.3f}, {z.max():.3f}]")
        print(f"    - 公式: z = t·ε + (1-t)·x_target")
        print(f"    - 物理意义: t=1时是纯噪声，t=0时是目标图像")
        
        # ===== 步骤5: 真实速度场 v = x_target - ε =====
        # v = x_target - epsilon  修改
        v = epsilon - x_target  # ✅ 正确方向
        u_tgt = v
        print(f"  真实速度场:")
        print(f"    - v: shape={v.shape}, range=[{v.min():.3f}, {v.max():.3f}]")
        print(f"    - 公式: v = x_target - ε")
        print(f"    - 物理意义: 从噪声到目标图像的方向")

        # ===== 步骤6: CFG处理（可选，虚拟染色建议禁用）=====
        # CFG: Classifier-Free Guidance
        # 以cfg_ratio的概率将条件x_source置零（无条件训练）
        if self.cfg_ratio > 0 and np.random.rand() < self.cfg_ratio:
            if self.cfg_uncond == 'zeros':
                x_source_cond = torch.zeros_like(x_source)
            elif self.cfg_uncond == 'v':
                x_source_cond = v  # 用速度场作为无条件输入
            print(f"  [CFG] 本批次使用无条件训练（cfg_ratio={self.cfg_ratio}）")
        else:
            x_source_cond = x_source
        
        # v_hat = v  # 默认使用真实速度场   修改
        # 正确目标：直接拟合真实速度场

        
        # ===== 步骤7: 模型前向传播 =====
        # 输入：插值状态z，时间t和r，条件x_source_cond
        # 输出：预测的平均速度场u
        print(f"\n  [模型前向传播]")
        print(f"    - 输入z: {z.shape}")
        print(f"    - 条件c: {x_source_cond.shape}")
        print(f"    - 时间t: {t.shape}")
        print(f"    - 时间r: {r.shape}")
        
        # 使用partial创建条件模型
        # 注意：这里的y参数用于传入条件图像x_source
        #model_partial = partial(model, y=c, cond_image=x_source_cond)
        def model_with_cond(z, t, r):
            return model(z, t, r, y=c, cond_image=x_source_cond)        
        # TODO 改 JVP计算：同时得到u和∂u/∂t
        zeros_z = torch.zeros_like(z)
        jvp_args = (
            model_with_cond,
            (z, t, r),
            (zeros_z, torch.ones_like(t), torch.zeros_like(r)),  # ✅ 只对 t 求偏导
        )
        print(f"[DEBUG] 条件图像传入前:")
        print(f"  - x_source_cond: {x_source_cond.shape}, range=[{x_source_cond.min():.3f}, {x_source_cond.max():.3f}]")
        print(f"  - x_source_cond是否全零: {(x_source_cond == 0).all().item()}")
                
        if self.create_graph:
            u, dudt = self.jvp_fn(*jvp_args, create_graph=self.create_graph)
        else:
            u, dudt = self.jvp_fn(*jvp_args)
        
        # 检查输出
        print(f"[DEBUG] 模型输出:")
        print(f"  - u的标准差: {u.std().item():.6f}")
        print(f"  - 如果标准差接近1.0，说明模型只输出噪声")
        # ===== 步骤8: 目标值计算 =====
        # u_tgt = v_hat - (t - r) * dudt
        # 这是MeanFlow的核心公式，确保模型预测的是平均速度场
        # u_tgt = v_hat - (t_ - r_) * dudt 修改
        v = epsilon - x_target
        # u_tgt = v  # 直接拟合真实速度场（避免梯度抵消）
        u_tgt = v - (t_ - r_) * dudt  # TODO ← 关键修正：不要直接用 u_tgt = v
        print(f"    - 修正后u_tgt: range=[{u_tgt.min():.3f}, {u_tgt.max():.3f}]")
        print(f"    - 目标值u_tgt: shape={u_tgt.shape}, range=[{u_tgt.min():.3f}, {u_tgt.max():.3f}]")  
        print(f"    - 公式: u_tgt = v - (t-r)·∂u/∂t")

        # ===== 步骤9: 计算误差和损失 =====
        error = u - stopgrad(u_tgt)
        print(f"\n  [损失计算]")
        print(f"    - 误差error: shape={error.shape}, mean={error.abs().mean():.6f}")
        
        loss = adaptive_l2_loss(error)
        # mse_val = (stopgrad(error) ** 2).mean() 原来
        mse_val = torch.mean((u - u_tgt) ** 2)
        print(f"    - Adaptive L2 loss: {loss.item():.6f}")
        print(f"    - MSE: {mse_val.item():.6f}")
        
        return loss, mse_val

    @torch.no_grad()
    def translate(self, model, x_source, sample_steps=30, cfg_scale=None, device='cuda'):   # sample_steps改大
        """
        图像翻译：将源染色翻译为目标染色
        
        ===== 采样过程（条件Flow ODE求解）=====
        初始状态: z = ε ~ N(0, I)  (t=1, 纯噪声)
        条件: c = x_source (源染色图像)
        
        迭代更新 (从t=1到t=0):
            v = model(z, t, r, c)  # 预测速度场
            z = z - (t-r)·v        # 沿速度场移动
        
        最终状态: z ≈ x_target (t=0, 目标染色)
        
        ===== Classifier-Free Guidance (可选) =====
        如果启用CFG (cfg_scale > 1):
            v_cond = model(z, t, r, c)         # 条件预测
            v_uncond = model(z, t, r, zeros)   # 无条件预测
            v = v_uncond + w·(v_cond - v_uncond)  # 加权组合
        
        Args:
            model: DiT模型
            x_source: 源染色图像 (HE), shape=(B, C, H, W)
            sample_steps: 采样步数（越多越精细）
            cfg_scale: CFG缩放因子，None=禁用，>1=增强条件影响
            device: 设备
        
        Returns:
            x_translated: 翻译后的目标染色图像 (Ki67), shape=(B, C, H, W)
        """
        model.eval()
        
        print(f"\n{'='*60}")
        print(f"[图像翻译 - 条件Flow采样]")
        print(f"  理论: 噪声(t=1) → 目标图像(t=0), 条件=源图像")
        print(f"  - 输入shape: {x_source.shape}")
        print(f"  - 采样步数: {sample_steps}")
        if cfg_scale is not None:
            print(f"  - CFG scale: {cfg_scale}")
        print(f"{'='*60}\n")
        
        # ===== 步骤1: 归一化源图像（条件）=====
        x_source = self.normer.norm(x_source)
        
        # ===== 步骤2: 初始化为噪声 z ~ N(0, I) =====
        z = torch.randn_like(x_source)
        print(f"初始状态 (t=1.0):")
        print(f"  - z: shape={z.shape}, range=[{z.min():.3f}, {z.max():.3f}]")
        print(f"  - 意义: 纯高斯噪声")
        print(f"  - 条件c (x_source): range=[{x_source.min():.3f}, {x_source.max():.3f}]\n")

        # ===== 步骤3: 时间序列：从1.0到0.0均匀划分 =====
        t_vals = torch.linspace(1.0, 0.0, sample_steps + 1, device=device)
        print(f"时间序列: {t_vals.cpu().numpy()}\n")

        # ===== 步骤4: ODE求解（从t=1到t=0）=====
        for i in range(sample_steps):
            t = torch.full((z.size(0),), t_vals[i], device=device)
            r = torch.full((z.size(0),), t_vals[i + 1], device=device)

            print(f"步骤 {i+1}/{sample_steps}:")
            print(f"  - 当前时间 t={t[0].item():.4f}, 目标时间 r={r[0].item():.4f}")
            print(f"  - 时间间隔 Δt={(t-r)[0].item():.4f}")

            t_ = rearrange(t, "b -> b 1 1 1").detach().clone()
            r_ = rearrange(r, "b -> b 1 1 1").detach().clone()

            # ===== 步骤4.1: 模型预测速度场 =====
            if cfg_scale is not None and cfg_scale > 1.0:
                # CFG: 混合条件和无条件预测
                v_cond = model(z, t, r, y=None, cond_image=x_source)
                
                if self.cfg_uncond == 'zeros':
                    v_uncond = model(z, t, r, y=None, cond_image=torch.zeros_like(x_source))
                else:
                    v_uncond = model(z, t, r, y=None, cond_image=v_cond)
                
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
                print(f"  - [CFG] v_cond: [{v_cond.min():.3f}, {v_cond.max():.3f}]")
                print(f"  - [CFG] v_uncond: [{v_uncond.min():.3f}, {v_uncond.max():.3f}]")
                print(f"  - [CFG] v_final: [{v.min():.3f}, {v.max():.3f}]")
            else:
                # 标准条件预测
                # 确保明确传入：
                if cfg_scale is not None and cfg_scale > 1.0:
                    v_cond = model(z, t, r, y=None, cond_image=x_source)
                    v_uncond = model(z, t, r, y=None, cond_image=torch.zeros_like(x_source))
                    v = v_uncond + cfg_scale * (v_cond - v_uncond)
                else:
                    v = model(z, t, r, y=None, cond_image=x_source)
                print(f"  - 预测速度场v: range=[{v.min():.3f}, {v.max():.3f}]")
            
            # ===== 步骤4.2: 更新规则: z <- z - (t-r)*v =====
            # 物理意义：沿着速度场方向移动(t-r)的距离
            # 从噪声逐渐变成目标图像
            z = z - (t_ - r_) * v
            print(f"  - 更新后z: range=[{z.min():.3f}, {z.max():.3f}]")
            print()

        print(f"最终状态 (t=0.0):")
        print(f"  - z: shape={z.shape}, range=[{z.min():.3f}, {z.max():.3f}]")
        print(f"  - 意义: 目标染色图像（Ki67）\n")

        # ===== 步骤5: 反归一化到原始像素空间 =====
        z = self.normer.unnorm(z)
        print(f"反归一化后:")
        print(f"  - z: range=[{z.min():.3f}, {z.max():.3f}] (应该在[0,1])")
        print(f"{'='*60}\n")
        
        return z