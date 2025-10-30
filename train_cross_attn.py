
import os, time
import torch
from torchvision import transforms as T
from torchvision.utils import save_image, make_grid

from PairedDataset import PairedDataset
from meanflow_conditional_v2 import ConditionalMeanFlow
from metrics_eval import evaluate as eval_dir
from models.model_cross_attention import ConditionalCrossAttnDiT
'''
方案二:Cross-attention(多层跨注意力,采样每步都用)

模型:ConditionalCrossAttnDiT

文件:model_cross_attention.py + train_cross_attn.py

做法：

主分支(当前状态 x)与条件分支(条件图像 c)分别做 patch embedding。

每个 Transformer 层：先自注意力(x→x),再跨注意力(Q=x,K/V=c),最后 MLP。

时间编码(t、r)通过 AdaLN 风格在每层调制(scale/shift/gate),保证与 MeanFlow 时间一致。

说明：因为采样每一步都会前向调用模型,而我们又在每一层做 cross-attn,所以天然满足“sample 的每一步不止加一次 attention”的要求。
'''
def main():
    PATCH_SIZE = 256
    BATCH_SIZE = 8
    N_TEST_STEPS = 1000
    SAMPLE_STEPS = 50
    EVAL_EVERY_STEPS = 50

    VAL_SAVE_PRED_DIR = "./outputs/val/pred"
    VAL_SAVE_GT_DIR   = "./outputs/val/gt"
    os.makedirs('./images', exist_ok=True)
    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs(VAL_SAVE_PRED_DIR, exist_ok=True)
    os.makedirs(VAL_SAVE_GT_DIR,   exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = PairedDataset(
        root_dir='/root/autodl-tmp/Ki67/data_paired',
        patch_size=PATCH_SIZE,
        transform=T.ToTensor(),
        recursive=False
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)

    model = ConditionalCrossAttnDiT(input_size=PATCH_SIZE, patch_size=16, in_channels=3, dim=384, depth=6, num_heads=6).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)

    meanflow = ConditionalMeanFlow(
        channels=3, image_size=PATCH_SIZE,
        normalizer=['minmax', None, None],
        flow_ratio=0.5, time_dist=['lognorm', -0.4, 1.0],
        cfg_ratio=0.0, cfg_scale=None, cfg_uncond='zeros', jvp_api='autograd'
    )

    model.train()
    it = iter(loader)
    for step in range(N_TEST_STEPS):
        try:
            x_src, x_tgt = next(it)
        except StopIteration:
            it = iter(loader); x_src, x_tgt = next(it)
        x_src = x_src.to(device); x_tgt = x_tgt.to(device)

        loss, mse_val = meanflow.loss(model, x_src, x_tgt, c=None)

        optim.zero_grad();
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optim.step()

        if (step + 1) % EVAL_EVERY_STEPS == 0:
            model.eval()
            with torch.no_grad():
                b = min(8, x_src.size(0))
                pred = meanflow.translate(model, x_src[:b], sample_steps=SAMPLE_STEPS, cfg_scale=None, device=device)
                for i in range(b):
                    name = "step%05d_idx%02d.png" % (step+1, i)
                    save_image(x_tgt[i].cpu(), os.path.join(VAL_SAVE_GT_DIR, name))
                    save_image(pred[i].cpu(),   os.path.join(VAL_SAVE_PRED_DIR, name))
                grid = make_grid(torch.cat([x_src[:b].cpu(), pred.cpu(), x_tgt[:b].cpu()], dim=0), nrow=b, normalize=True, value_range=(0,1))
                img_path = "./images/eval_cros_step_%05d.png" % (step+1)
                save_image(grid, img_path)
                try:
                    stamp = time.strftime("%Y%m%d-%H%M%S")
                    csv_path = "./metrics/per_image_%s_step%05d.csv" % (stamp, step+1)
                    eval_dir(VAL_SAVE_PRED_DIR, VAL_SAVE_GT_DIR, csv_out=csv_path)
                except Exception as e:
                    print("[Eval] failed:", e)
            model.train()

    ckpt = "./checkpoints/%s_final.pt" % ("ConditionalCrossAttnDiT".lower())
    torch.save({'model': model.state_dict(), 'step': N_TEST_STEPS}, ckpt)
    print("Saved:", ckpt)

if __name__ == "__main__":
    main()
