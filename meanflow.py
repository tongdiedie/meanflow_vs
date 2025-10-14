import torch
import torch.nn.functional as F
from einops import rearrange
from functools import partial
import numpy as np


class Normalizer:
    # minmax for raw image, mean_std for vae latent
    def __init__(self, mode='minmax', mean=None, std=None):
        assert mode in ['minmax', 'mean_std'], "mode must be 'minmax' or 'mean_std'"
        self.mode = mode

        if mode == 'mean_std':
            if mean is None or std is None:
                raise ValueError("mean and std must be provided for 'mean_std' mode")
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)

    @classmethod
    def from_list(cls, config):
        """
        config: [mode, mean, std]
        """
        mode, mean, std = config
        return cls(mode, mean, std)

    def norm(self, x):
        if self.mode == 'minmax':
            return x * 2 - 1
        elif self.mode == 'mean_std':
            return (x - self.mean.to(x.device)) / self.std.to(x.device)
    # 将图像从归一化空间还原到原始像素空间
    def unnorm(self, x):
        if self.mode == 'minmax':
            x = x.clip(-1, 1)
            return (x + 1) * 0.5
        elif self.mode == 'mean_std':
            return x * self.std.to(x.device) + self.mean.to(x.device)


def stopgrad(x):
    return x.detach()


def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    """
    Adaptive L2 loss: sg(w) * ||Δ||_2^2, where w = 1 / (||Δ||^2 + c)^p, p = 1 - γ
    Args:
        error: Tensor of shape (B, C, W, H)
        gamma: Power used in original ||Δ||^{2γ} loss
        c: Small constant for stability
    Returns:
        Scalar loss
    """
    delta_sq = torch.mean(error ** 2, dim=(1, 2, 3), keepdim=False)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq  # ||Δ||^2
    return (stopgrad(w) * loss).mean()  # stopgrad() 是为了防止权重 w 在反向传播中影响梯度计算。
    # 换句话说，w 是根据当前误差计算出来的，但它本身不参与梯度更新，只是作为加权系数使用。这样可以保证训练稳定性。


class MeanFlow:
    def __init__(
        self,
        channels=1,
        image_size=32,
        num_classes=10,
        normalizer=['minmax', None, None],
        # mean flow settings
        flow_ratio=0.50,
        # time distribution, mu, sigma
        time_dist=['lognorm', -0.4, 1.0],
        cfg_ratio=0.10,
        # set scale as none to disable CFG distill
        cfg_scale=2.0,
        # experimental
        cfg_uncond='v',
        jvp_api='autograd',
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.num_classes = num_classes
        self.use_cond = num_classes is not None

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

    # fix: r should be always not larger than t
    def sample_t_r(self, batch_size, device):   # 用于生成满足特定分布的时间对 (t, r)，其中 t >= r
        if self.time_dist[0] == 'uniform':   # 根据分布类型(uniform)生成两列随机数
            samples = np.random.rand(batch_size, 2).astype(np.float32)

        elif self.time_dist[0] == 'lognorm':   # 根据分布类型(lognorm)生成两列随机数
            mu, sigma = self.time_dist[-2], self.time_dist[-1]
            normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu    # [b,2]
            samples = 1 / (1 + np.exp(-normal_samples))  # Apply sigmoid

        # Assign t = max, r = min, for each pair
        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])
        # 按照 flow_ratio 概率随机选取部分样本，使这些样本的r=t;
        num_selected = int(self.flow_ratio * batch_size)
        indices = np.random.permutation(batch_size)[:num_selected]
        r_np[indices] = t_np[indices]

        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)
        return t, r

    def loss(self, model, x, c=None):
        batch_size = x.shape[0]
        device = x.device
        # 该函数用于生成时间对(t,r),其中t表示较大的时间值，r表示较小的时间值，并以一定比例让r等于t。
        t, r = self.sample_t_r(batch_size, device)

        t_ = rearrange(t, "b -> b 1 1 1").detach().clone()
        r_ = rearrange(r, "b -> b 1 1 1").detach().clone()

        e = torch.randn_like(x) # 生成噪声e  [b,c,h,w]=[48,1,32,32]
        x = self.normer.norm(x)

        z = (1 - t_) * x + t_ * e  # 插值变量z
        v = e - x # 条件速度v_t

        if c is not None:
            assert self.cfg_ratio is not None
            uncond = torch.ones_like(c) * self.num_classes # [b],内部全是1
            cfg_mask = torch.rand_like(c.float()) < self.cfg_ratio # 以一定概率self.cfg_ratio将输入c中的元素替换为无条件输入uncond
            c = torch.where(cfg_mask, uncond, c)
            if self.w is not None:  # 使用无条件分支进行CFG(Classifier-Free Guidance)增强
                with torch.no_grad():
                    u_t = model(z, t, t, uncond)
                v_hat = self.w * v + (1 - self.w) * u_t
                if self.cfg_uncond == 'v':
                    # offical JAX repo uses original v for unconditional items
                    cfg_mask = rearrange(cfg_mask, "b -> b 1 1 1").bool()  # as v = wv -(1-w)v wv - (1-w)u in the unconditional case,should we directly use v instead?
                    v_hat = torch.where(cfg_mask, v, v_hat)  # 构造v_hat作为目标方向
            else:
                v_hat = v

        # forward pass
        # u = model(z, t, r, y=c)
        model_partial = partial(model, y=c)  # model_partial 是固定了部分参数(y=c)后的模型函数
        jvp_args = (   # 雅可比矩阵向量积(JVP)的参数元组jvp_args
            lambda z, t, r: model_partial(z, t, r),
            (z, t, r),
            (v_hat, torch.ones_like(t), torch.zeros_like(r)),
        )

        if self.create_graph:
            u, dudt = self.jvp_fn(*jvp_args, create_graph=True)  # 调用jvp_fn得到输u和其时间导数dudt
        else:
            u, dudt = self.jvp_fn(*jvp_args)

        u_tgt = v_hat - (t_ - r_) * dudt  # 目标值

        error = u - stopgrad(u_tgt)   # 计算当前输出u与目标输出utgt的误差
        loss = adaptive_l2_loss(error)  #  对误差使用自适应L2损失函数计算最终损失
        # loss = F.mse_loss(u, stopgrad(u_tgt))

        mse_val = (stopgrad(error) ** 2).mean()
        return loss, mse_val

    @torch.no_grad()
    def sample_each_class(self, model, n_per_class, classes=None,   # 为每个类别生成指定数量的图像样本
                          sample_steps=5, device='cuda'):
        model.eval()

        if classes is None:
            c = torch.arange(self.num_classes, device=device).repeat(n_per_class)  # [c]创建类别标签张量，每个类重复n_per_class次
        else:
            c = torch.tensor(classes, device=device).repeat(n_per_class)

        z = torch.randn(c.shape[0], self.channels,    # 初始化随机噪声图像 [c,channel,h,w]=[10,1,32,32]
                        self.image_size, self.image_size,
                        device=device)

        t_vals = torch.linspace(1.0, 0.0, sample_steps + 1, device=device)

        # print(t_vals)

        for i in range(sample_steps):
            t = torch.full((z.size(0),), t_vals[i], device=device)
            r = torch.full((z.size(0),), t_vals[i + 1], device=device)

            # print(f"t: {t[0].item():.4f};  r: {r[0].item():.4f}")

            t_ = rearrange(t, "b -> b 1 1 1").detach().clone()
            r_ = rearrange(r, "b -> b 1 1 1").detach().clone()

            v = model(z, t, r, c)  # 使用模型对噪声进行一次去噪操作
            z = z - (t_-r_) * v

        z = self.normer.unnorm(z)  # 将图像从归一化空间还原到原始像素空间
        return z