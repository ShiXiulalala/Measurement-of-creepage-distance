import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def calculate_mean_std(folder_path):
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    image_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, file))

    for img_path in tqdm(image_paths, desc="Processing Images"):
        try:
            img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)
            if img.size == 0:
                print(f"警告: 图像为空 {img_path}")
                continue
            pixel_sum += img.sum(axis=(0, 1))
            pixel_sq_sum += (img ** 2).sum(axis=(0, 1))
            pixel_count += img.shape[0] * img.shape[1]
        except Exception as e:
            print(f"错误: 加载 {img_path} 失败 - {e}")
            continue

    if pixel_count == 0:
        raise ValueError("没有有效的图像数据！请检查文件夹路径或文件格式。")

    mean = pixel_sum / pixel_count
    std = np.sqrt((pixel_sq_sum / pixel_count) - (mean ** 2))
    
    # 处理可能的数值不稳定（如方差接近0）
    std = np.nan_to_num(std, nan=0.0, posinf=1.0, neginf=0.0)

    return mean.tolist(), std.tolist()

# 使用示例
folder_path = "./Data/227"
mean, std = calculate_mean_std(folder_path)
print("均值 (mean_values):", [mean])
print("标准差 (std_values):", [std])