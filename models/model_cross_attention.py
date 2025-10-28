import torch, torch.nn as nn, torch.nn.functional as F, math
from timm.models.vision_transformer import PatchEmbed
from torch.backends.cuda import sdp_kernel


def modulate(x, scale, shift):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.scale = d**0.5
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.g


class TimestepEmbedder(nn.Module):
    def __init__(self, d, nfreq=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(nfreq, d), nn.SiLU(), nn.Linear(d, d))
        self.nfreq = nfreq

    @staticmethod
    def timestep_embedding(t, d, max_period=10000):
        half = d // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half).float() / half
        ).to(t.device)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1) if d % 2 else emb

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t * 1000, self.mlp[0].in_features))


class AdaLN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(d, 3 * d))
        self.norm = RMSNorm(d)

    def forward(self, x, c):
        shift, scale, gate = self.mod(c).chunk(3, dim=-1)
        return x + gate.unsqueeze(1) * self.norm(x), shift, scale


class CrossAttentionBlock(nn.Module):
    def __init__(self, d, h=6, mlp_ratio=4.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.proj_out = nn.Linear(d, d)
        self.ada_self = AdaLN(d)
        self.ada_cross = AdaLN(d)
        self.ada_mlp = AdaLN(d)
        m = int(d * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d, m), nn.GELU(), nn.Linear(m, d))
    '''
    def forward(self, x, cond, temb):
        xres, sh, sc = self.ada_self(x, temb)
        q = xres * (1 + sc.unsqueeze(1)) + sh.unsqueeze(1)
        out, _ = self.self_attn(q, q, q, need_weights=False)
        x = x + out
        xres, sh, sc = self.ada_cross(x, temb)
        q = xres * (1 + sc.unsqueeze(1)) + sh.unsqueeze(1)
        k = self.k_proj(cond)
        v = self.v_proj(cond)
        out, _ = self.cross_attn(q, k, v, need_weights=False)
        x = x + self.proj_out(out)
        xres, sh, sc = self.ada_mlp(x, temb)
        x = x + self.mlp(xres * (1 + sc.unsqueeze(1)) + sh.unsqueeze(1))
        return x
    '''
    def forward(self, x, cond, temb):
        # ---- Self-Attention ----
        xres, sh, sc = self.ada_self(x, temb)
        q = xres * (1 + sc.unsqueeze(1)) + sh.unsqueeze(1)
        # 禁用 Flash/MemEfficient 注意力，启用 math 后端
        with sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            out, _ = self.self_attn(q, q, q, need_weights=False)
        x = x + out

        # ---- Cross-Attention ----
        xres, sh, sc = self.ada_cross(x, temb)
        q = xres * (1 + sc.unsqueeze(1)) + sh.unsqueeze(1)
        k = self.k_proj(cond)
        v = self.v_proj(cond)
        with sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            out, _ = self.cross_attn(q, k, v, need_weights=False)
        x = x + self.proj_out(out)

        # ---- MLP ----
        xres, sh, sc = self.ada_mlp(x, temb)
        x = x + self.mlp(xres * (1 + sc.unsqueeze(1)) + sh.unsqueeze(1))
        return x



class FinalLayer(nn.Module):
    def __init__(self, d, p, out):
        super().__init__()
        self.norm = RMSNorm(d)
        self.proj = nn.Linear(d, p * p * out)
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))

    def forward(self, x, temb):
        sh, sc = self.mod(temb).chunk(2, dim=-1)
        x = self.norm(x)
        x = x * (1 + sc.unsqueeze(1)) + sh.unsqueeze(1)
        return self.proj(x)


class ConditionalCrossAttnDiT(nn.Module):
    def __init__(
        self,
        input_size=256,
        patch_size=16,
        in_channels=3,
        dim=384,
        depth=6,
        num_heads=6,
    ):
        super().__init__()
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, dim)
        self.cond_embedder = PatchEmbed(input_size, patch_size, in_channels, dim)
        T = self.x_embedder.num_patches
        self.pos_x = nn.Parameter(torch.zeros(1, T, dim))
        self.pos_c = nn.Parameter(torch.zeros(1, T, dim))
        self.t_emb = TimestepEmbedder(dim)
        self.r_emb = TimestepEmbedder(dim)
        self.blocks = nn.ModuleList(
            [CrossAttentionBlock(dim, num_heads) for _ in range(depth)]
        )
        self.final = FinalLayer(dim, patch_size, self.out_channels)
        nn.init.normal_(self.pos_x, std=0.02)
        nn.init.normal_(self.pos_c, std=0.02)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, p, p, c).permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.reshape(x.shape[0], c, h * p, w * p)

    def forward(self, x, t, r, y=None, cond_image=None):
        assert cond_image is not None
        x_tok = self.x_embedder(x) + self.pos_x
        c_tok = self.cond_embedder(cond_image) + self.pos_c
        temb = self.t_emb(t) + self.r_emb(r)
        for blk in self.blocks:
            x_tok = blk(x_tok, c_tok, temb)
        x_tok = self.final(x_tok, temb)
        return self.unpatchify(x_tok)
