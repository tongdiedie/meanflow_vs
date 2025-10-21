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


class MeanFlowTranslation:
    """
    MeanFlow用于图像翻译（虚拟染色）
    
    核心思想改变：
    原版：噪声e (t=1) <--流--> 干净图像x (t=0)
    翻译版：源图像x_source (t=1) <--流--> 目标图像x_target (t=0)
    
    关键公式：
    - 插值状态: z(t) = (1-t)*x_target + t*x_source
    - 速度场: v(t) = x_target - x_source（从源域到目标域的方向）
    - 模型预测: u(z, t, r) ≈ 平均速度场
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
        # 虚拟染色任务不需要CFG，但保留代码结构以便扩展
        cfg_ratio=0.0,  # 设为0禁用CFG
        cfg_scale=None,
        cfg_uncond='v',
        jvp_api='autograd',
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        
        print(f"\n{'='*60}")
        print(f"[MeanFlowTranslation] 初始化图像翻译模型")
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
        
        物理意义（翻译版本）：
        - t=1: 完全是源染色（如HE）
        - t=0: 完全是目标染色（如IHC）
        - t in (0,1): 源和目标的插值状态
        
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
        虚拟染色的训练损失
        
        核心改变：
        1. 输入变成配对数据：(x_source, x_target)
        2. 插值变成：z(t) = (1-t)*x_target + t*x_source
        3. 速度场变成：v = x_target - x_source
        
        Args:
            model: DiT模型
            x_source: 源染色图像, shape=(B, C, H, W)
            x_target: 目标染色图像, shape=(B, C, H, W)
            c: 条件标签（虚拟染色任务通常不需要，保留用于多任务扩展）
        
        Returns:
            loss: 标量损失
            mse_val: MSE值（用于监控）
        """
        batch_size = x_source.shape[0]
        device = x_source.device
        
        print(f"\n[Loss计算]")
        print(f"  输入shape:")
        print(f"    - x_source: {x_source.shape}  (源染色)")
        print(f"    - x_target: {x_target.shape}  (目标染色)")
        
        # 采样时间对(t, r)
        t, r = self.sample_t_r(batch_size, device)
        print(f"  时间采样:")
        print(f"    - t: shape={t.shape}, range=[{t.min():.3f}, {t.max():.3f}]")
        print(f"    - r: shape={r.shape}, range=[{r.min():.3f}, {r.max():.3f}]")
        print(f"    - (t-r): mean={((t-r).mean()):.3f}")
        
        t_ = rearrange(t, "b -> b 1 1 1").detach().clone()  # shape=(B, 1, 1, 1)
        r_ = rearrange(r, "b -> b 1 1 1").detach().clone()

        # === 关键改动1: 归一化配对图像 ===
        x_source = self.normer.norm(x_source)  # [0,1] -> [-1,1]
        x_target = self.normer.norm(x_target)
        print(f"  归一化后:")
        print(f"    - x_source: range=[{x_source.min():.3f}, {x_source.max():.3f}]")
        print(f"    - x_target: range=[{x_target.min():.3f}, {x_target.max():.3f}]")

        # === 关键改动2: 插值状态 z(t) = (1-t)*x_target + t*x_source ===
        # 物理意义：t=1时是源染色，t=0时是目标染色
        z = (1 - t_) * x_target + t_ * x_source
        print(f"  插值状态:")
        print(f"    - z: shape={z.shape}, range=[{z.min():.3f}, {z.max():.3f}]")
        print(f"    - 公式: z = (1-t)*x_target + t*x_source")
        
        # === 关键改动3: 速度场 v = x_target - x_source ===
        # 物理意义：从源染色到目标染色的方向
        v = x_target - x_source
        print(f"  真实速度场:")
        print(f"    - v: shape={v.shape}, range=[{v.min():.3f}, {v.max():.3f}]")
        print(f"    - 公式: v = x_target - x_source")

        # CFG部分（虚拟染色通常不需要，这里保留结构）
        v_hat = v  # 默认使用真实速度场
        
        # === 关键改动4: 模型前向传播 ===
        # 输入：插值状态z，时间t和r
        # 输出：预测的平均速度场u
        print(f"\n  [模型前向传播]")
        model_partial = partial(model, y=c)
        
        # JVP计算：同时得到u和∂u/∂t
        jvp_args = (
            lambda z, t, r: model_partial(z, t, r),
            (z, t, r),
            (v_hat, torch.ones_like(t), torch.zeros_like(r)),
        )
        
        if self.create_graph:
            u, dudt = self.jvp_fn(*jvp_args, create_graph=True)
        else:
            u, dudt = self.jvp_fn(*jvp_args)
        
        print(f"    - 模型输出u: shape={u.shape}, range=[{u.min():.3f}, {u.max():.3f}]")
        print(f"    - 时间导数dudt: shape={dudt.shape}, range=[{dudt.min():.3f}, {dudt.max():.3f}]")

        # === 关键改动5: 目标值计算 ===
        # u_tgt = v_hat - (t - r) * dudt
        # 这是MeanFlow的核心公式，确保模型预测的是平均速度场
        u_tgt = v_hat - (t_ - r_) * dudt
        print(f"    - 目标值u_tgt: shape={u_tgt.shape}, range=[{u_tgt.min():.3f}, {u_tgt.max():.3f}]")
        print(f"    - 公式: u_tgt = v_hat - (t-r)*dudt")

        # 计算误差和损失
        error = u - stopgrad(u_tgt)
        print(f"\n  [损失计算]")
        print(f"    - 误差error: shape={error.shape}, mean={error.abs().mean():.6f}")
        
        loss = adaptive_l2_loss(error)
        mse_val = (stopgrad(error) ** 2).mean()
        
        print(f"    - Adaptive L2 loss: {loss.item():.6f}")
        print(f"    - MSE: {mse_val.item():.6f}")
        
        return loss, mse_val

    @torch.no_grad()
    def translate(self, model, x_source, sample_steps=5, device='cuda'):
        """
        图像翻译：将源染色翻译为目标染色
        
        翻译过程：
        初始状态: z = x_source (t=1，完全是源染色)
        迭代更新: z <- z - (t-r)*v  (逐步向目标染色移动)
        最终状态: z ≈ x_target (t=0，完全是目标染色)
        
        Args:
            model: DiT模型
            x_source: 源染色图像, shape=(B, C, H, W)
            sample_steps: 采样步数（越多越精细，但也越慢）
            device: 设备
        
        Returns:
            x_translated: 翻译后的目标染色图像, shape=(B, C, H, W)
        """
        model.eval()
        
        print(f"\n{'='*60}")
        print(f"[图像翻译] 开始翻译过程")
        print(f"  - 输入shape: {x_source.shape}")
        print(f"  - 采样步数: {sample_steps}")
        print(f"{'='*60}\n")
        
        # === 初始化：z = x_source（归一化后）===
        z = self.normer.norm(x_source)
        print(f"初始状态 (t=1.0):")
        print(f"  - z: shape={z.shape}, range=[{z.min():.3f}, {z.max():.3f}]")
        print(f"  - 意义: 完全是源染色\n")

        # 时间序列：从1.0到0.0均匀划分
        t_vals = torch.linspace(1.0, 0.0, sample_steps + 1, device=device)
        print(f"时间序列: {t_vals.cpu().numpy()}\n")

        for i in range(sample_steps):
            t = torch.full((z.size(0),), t_vals[i], device=device)
            r = torch.full((z.size(0),), t_vals[i + 1], device=device)

            print(f"步骤 {i+1}/{sample_steps}:")
            print(f"  - 当前时间 t={t[0].item():.4f}, 目标时间 r={r[0].item():.4f}")
            print(f"  - 时间间隔 Δt={(t-r)[0].item():.4f}")

            t_ = rearrange(t, "b -> b 1 1 1").detach().clone()
            r_ = rearrange(r, "b -> b 1 1 1").detach().clone()

            # === 模型预测平均速度场 ===
            v = model(z, t, r, y=None)
            print(f"  - 预测速度场v: range=[{v.min():.3f}, {v.max():.3f}]")
            
            # === 更新规则: z <- z - (t-r)*v ===
            # 物理意义：沿着速度场方向移动(t-r)的距离
            z = z - (t_ - r_) * v
            print(f"  - 更新后z: range=[{z.min():.3f}, {z.max():.3f}]")
            print()

        print(f"最终状态 (t=0.0):")
        print(f"  - z: shape={z.shape}, range=[{z.min():.3f}, {z.max():.3f}]")
        print(f"  - 意义: 完全是目标染色\n")

        # === 反归一化到原始像素空间 ===
        z = self.normer.unnorm(z)
        print(f"反归一化后:")
        print(f"  - z: range=[{z.min():.3f}, {z.max():.3f}] (应该在[0,1])")
        print(f"{'='*60}\n")
        
        return z