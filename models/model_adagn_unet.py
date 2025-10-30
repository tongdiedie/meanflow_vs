import torch, torch.nn as nn


def conv3x3(ic, oc, s=1):
    return nn.Conv2d(ic, oc, 3, s, 1)


class AdaGN(nn.Module):
    def __init__(self, C, G=8, E=256):
        super().__init__()
        self.gn = nn.GroupNorm(G, C, affine=False)
        self.s = nn.Linear(E, C)
        self.b = nn.Linear(E, C)

    def forward(self, x, e):
        x = self.gn(x)
        s = self.s(e).unsqueeze(-1).unsqueeze(-1)
        b = self.b(e).unsqueeze(-1).unsqueeze(-1)
        return x * (1 + s) + b


class ResBlock(nn.Module):
    def __init__(self, ic, oc, E=256):
        super().__init__()
        self.c1 = conv3x3(ic, oc)
        self.n1 = AdaGN(oc, E=E)
        self.act = nn.SiLU()
        self.c2 = conv3x3(oc, oc)
        self.n2 = AdaGN(oc, E=E)
        self.skip = ic != oc
        self.p = nn.Conv2d(ic, oc, 1) if self.skip else None

    def forward(self, x, e):
        h = self.c1(x)
        h = self.n1(h, e)
        h = self.act(h)
        h = self.c2(h)
        h = self.n2(h, e)
        h = self.act(h)
        return (self.p(x) if self.skip else x) + h


class CondEnc(nn.Module):
    def __init__(self, ic=3, E=256):
        super().__init__()
        self.net = nn.Sequential(
            conv3x3(ic, 32),
            nn.SiLU(),
            nn.AvgPool2d(2),
            conv3x3(32, 64),
            nn.SiLU(),
            nn.AvgPool2d(2),
            conv3x3(64, 128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, E)

    def forward(self, x):
        h = self.net(x).flatten(1)
        return self.proj(h)


class TEmb(nn.Module):
    def __init__(self, E=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, E), nn.SiLU(), nn.Linear(E, E))

    def forward(self, t):
        return self.net(t.unsqueeze(-1))


class ConditionalAdaGNUNet(nn.Module):
    def __init__(self, in_ch=3, base=64, E=256):
        super().__init__()
        self.enc = CondEnc(in_ch, E)
        self.temb = TEmb(E)
        self.inp = conv3x3(in_ch * 2, base)
        self.b1 = ResBlock(base, base, E)
        self.d1 = nn.AvgPool2d(2)
        self.b2 = ResBlock(base, base * 2, E)
        self.d2 = nn.AvgPool2d(2)
        self.b3 = ResBlock(base * 2, base * 2, E)
        self.u1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.b4 = ResBlock(base * 2, base, E)
        self.u2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.b5 = ResBlock(base, base, E)
        self.out = conv3x3(base, in_ch)

    def forward(self, z, t, r, y=None, cond_image=None):
        assert cond_image is not None
        e = self.enc(cond_image) + self.temb(t) + self.temb(r)
        # 将条件图像与当前状态按通道维拼接，提供对齐的空间引导
        # 说明：MeanFlow 的调用里 z、cond_image 已做一致的归一化，这里无需再额外归一化
        h = self.inp(torch.cat([z, cond_image], dim=1))
        h = self.b1(h, e)
        h = self.d1(h)
        h = self.b2(h, e)
        h = self.d2(h)
        h = self.b3(h, e)
        h = self.u1(h)
        h = self.b4(h, e)
        h = self.u2(h)
        h = self.b5(h, e)
        return self.out(h)
