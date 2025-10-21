import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Mlp
from timm.models.vision_transformer import Attention
import torch.nn.functional as F
from einops import repeat, pack, unpack
from torch.cuda.amp import autocast


def modulate(x, scale, shift):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, dim, nfreq=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(nfreq, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.nfreq = nfreq

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half_dim, dtype=torch.float32)
            / half_dim
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t = t * 1000
        t_freq = self.timestep_embedding(t, self.nfreq)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, dim):
        super().__init__()
        self.embedding = nn.Embedding(num_classes + 1, dim)
        self.num_classes = num_classes

    def forward(self, labels):
        embeddings = self.embedding(labels)
        return embeddings


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.g


class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=True, qk_norm=True, norm_layer=RMSNorm)
        # flash attn can not be used with jvp
        self.attn.fused_attn = False
        self.norm2 = RMSNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_dim, act_layer=approx_gelu, drop=0
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), scale_msa, shift_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), scale_mlp, shift_mlp)
        )
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim, patch_size, out_dim):
        super().__init__()
        self.norm_final = RMSNorm(dim)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class ConditionalMFDiT(nn.Module):
    """
    条件DiT模型，用于虚拟染色
    
    关键改动：
    1. 添加条件图像编码器（与主图像共享patch embedding）
    2. 通过cross-attention或concatenation融合条件信息
    3. 支持CFG（Classifier-Free Guidance）
    
    模型签名：
        forward(z, t, r, y=None, cond_image=None)
        
    其中：
        - z: 噪声状态, shape=(B, C, H, W)
        - t, r: 时间参数
        - y: 类别标签（可选）
        - cond_image: 条件图像（源染色），shape=(B, C, H, W)
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        dim=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        num_classes=None,
        cond_mode='concat',  # 'concat', 'cross_attn', 'add'
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.cond_mode = cond_mode

        # 主图像的patch embedding
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, dim)
        
        # 条件图像的patch embedding（共享参数）
        self.cond_embedder = PatchEmbed(input_size, patch_size, in_channels, dim)
        
        # 时间编码器
        self.t_embedder = TimestepEmbedder(dim)
        self.r_embedder = TimestepEmbedder(dim)

        # 类别编码器（可选）
        self.use_cond = num_classes is not None
        self.y_embedder = LabelEmbedder(num_classes, dim) if self.use_cond else None

        num_patches = self.x_embedder.num_patches
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim), requires_grad=True)
        self.cond_pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim), requires_grad=True)

        # Transformer blocks
        if cond_mode == 'concat':
            # 拼接模式：条件和主图像拼接后一起处理
            self.blocks = nn.ModuleList([
                DiTBlock(dim, num_heads, mlp_ratio) for _ in range(depth)
            ])
            self.num_patches_total = num_patches * 2  # 拼接后的token数量
        elif cond_mode == 'cross_attn':
            # Cross-attention模式（更复杂，这里简化为concat）
            # TODO: 实现真正的cross-attention
            print("[WARNING] cross_attn模式暂未实现，fallback到concat模式")
            self.cond_mode = 'concat'
            self.blocks = nn.ModuleList([
                DiTBlock(dim, num_heads, mlp_ratio) for _ in range(depth)
            ])
            self.num_patches_total = num_patches * 2
        elif cond_mode == 'add':
            # 加法模式：条件和主图像逐元素相加
            self.blocks = nn.ModuleList([
                DiTBlock(dim, num_heads, mlp_ratio) for _ in range(depth)
            ])
            self.num_patches_total = num_patches
        else:
            raise ValueError(f"Unknown cond_mode: {cond_mode}")

        self.final_layer = FinalLayer(dim, patch_size, self.out_channels)

        self.initialize_weights()
        
        print(f"\n[ConditionalMFDiT] 初始化")
        print(f"  - 条件融合模式: {self.cond_mode}")
        print(f"  - Patch数量: {num_patches}")
        if cond_mode == 'concat':
            print(f"  - 总token数量: {self.num_patches_total} (包含条件)")

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.cond_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear:
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        
        w_cond = self.cond_embedder.proj.weight.data
        nn.init.xavier_uniform_(w_cond.view([w_cond.shape[0], -1]))
        nn.init.constant_(self.cond_embedder.proj.bias, 0)

        # Initialize label embedding table:
        if self.y_embedder is not None:
            nn.init.normal_(self.y_embedder.embedding.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.r_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.r_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1], f"Cannot unpatchify: {x.shape[1]} tokens, expected {h*w}"

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, r, y=None, cond_image=None):
        """
        条件DiT的前向传播
        
        Args:
            x: 噪声状态, shape=(B, C, H, W)
            t: 时间参数, shape=(B,)
            r: 参考时间, shape=(B,)
            y: 类别标签, shape=(B,) (可选)
            cond_image: 条件图像（源染色），shape=(B, C, H, W)
        
        Returns:
            output: 预测的速度场, shape=(B, C, H, W)
        """
        H, W = x.shape[-2:]
        
        # ===== 1. 编码主图像（噪声状态）=====
        x_tokens = self.x_embedder(x) + self.pos_embed  # (B, T, D)
        
        # ===== 2. 编码条件图像 =====
        if cond_image is not None:
            cond_tokens = self.cond_embedder(cond_image) + self.cond_pos_embed  # (B, T, D)
            
            if self.cond_mode == 'concat':
                # 拼接模式：[x_tokens | cond_tokens]
                x_tokens = torch.cat([x_tokens, cond_tokens], dim=1)  # (B, 2T, D)
            elif self.cond_mode == 'add':
                # 加法模式：x_tokens + cond_tokens
                x_tokens = x_tokens + cond_tokens  # (B, T, D)
        
        # ===== 3. 时间编码 =====
        t_emb = self.t_embedder(t)  # (B, D)
        r_emb = self.r_embedder(r)  # (B, D)
        c = t_emb + r_emb  # (B, D)
        
        # ===== 4. 类别条件（可选）=====
        if self.use_cond and y is not None:
            y_emb = self.y_embedder(y)
            c = c + y_emb
        
        # ===== 5. Transformer blocks =====
        for block in self.blocks:
            x_tokens = block(x_tokens, c)  # (B, T', D) where T'=T or 2T
        
        # ===== 6. 如果是concat模式，只取前半部分（对应主图像）=====
        if self.cond_mode == 'concat':
            num_patches = self.x_embedder.num_patches
            x_tokens = x_tokens[:, :num_patches, :]  # (B, T, D)
        
        # ===== 7. 最终层 =====
        x_tokens = self.final_layer(x_tokens, c)  # (B, T, patch_size^2 * C)
        output = self.unpatchify(x_tokens)  # (B, C, H, W)
        
        return output


# ===== 位置编码辅助函数 =====

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb