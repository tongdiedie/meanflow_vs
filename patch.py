from PIL import Image
import os

# 输入与输出路径
input_dir = "/root/autodl-tmp/Ki67/data_paired/target"   # 存放原图的文件夹
output_dir = "/root/autodl-tmp/Ki67/data_paired/target_patch"       # 输出patch的文件夹
os.makedirs(output_dir, exist_ok=True)

# patch 大小
patch_size = 256

# 获取所有图片文件
images = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]

for img_name in sorted(images):
    img_path = os.path.join(input_dir, img_name)
    img = Image.open(img_path)
    w, h = img.size

    # 计算 patch 行列数
    n_cols = w // patch_size
    n_rows = h // patch_size

    base_name = os.path.splitext(img_name)[0]  # e.g. image0001

    count = 1
    for i in range(n_rows):
        for j in range(n_cols):
            # 裁剪坐标
            left = j * patch_size
            upper = i * patch_size
            right = left + patch_size
            lower = upper + patch_size

            patch = img.crop((left, upper, right, lower))

            # 命名格式：image0001_01.png, image0001_02.png ...
            patch_name = f"{base_name}_{count:02d}.png"
            patch.save(os.path.join(output_dir, patch_name))
            count += 1

print("✅ 所有图像已成功切分并保存！")
