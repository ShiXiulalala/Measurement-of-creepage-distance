import os
import random
import shutil
from PIL import Image

def generate_quantization_dataset(
    input_folder,
    output_folder,
    output_txt="datasets.txt",
    sample_num=200,
    shuffle=True,
    check_valid=True,
    target_size=None,
    copy_files=True
):
    """
    生成RKNN量化数据集（强制使用Unix路径分隔符'/'的版本）
    
    参数:
        input_folder: 源文件夹路径（相对/绝对）
        output_folder: 目标文件夹路径（相对/绝对）
        output_txt: 输出的txt文件名（默认datasets.txt）
        sample_num: 抽取数量（默认200）
        shuffle: 是否随机打乱（默认True）
        check_valid: 是否检查图片有效性（默认True）
        target_size: 统一调整尺寸（可选，如(640,640)）
        copy_files: 是否实际复制文件（默认True）
    """
    # 内部使用绝对路径操作
    abs_input = os.path.abspath(input_folder)
    abs_output = os.path.abspath(output_folder)
    os.makedirs(abs_output, exist_ok=True)
    
    # 获取output_txt的绝对路径和所在目录
    abs_output_txt = os.path.abspath(output_txt)
    output_txt_dir = os.path.dirname(abs_output_txt)

    # 收集所有有效图片的相对路径（相对于input_folder）
    valid_files = []
    for root, _, files in os.walk(abs_input):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                abs_src_path = os.path.join(root, file)
                rel_src_path = os.path.relpath(abs_src_path, start=abs_input)
                
                # 验证图片有效性
                if check_valid:
                    try:
                        img = Image.open(abs_src_path)
                        img.verify()
                        if target_size:
                            img = img.resize(target_size)
                        valid_files.append(rel_src_path)
                    except (IOError, OSError) as e:
                        print(f"跳过无效图片: {rel_src_path} - {str(e)}")
                        continue
                else:
                    valid_files.append(rel_src_path)

    # 检查数量
    if len(valid_files) < sample_num:
        print(f"警告: 只有 {len(valid_files)} 张有效图片，少于要求的 {sample_num} 张")
        sample_num = len(valid_files)
    
    # 随机抽样
    if shuffle:
        random.shuffle(valid_files)
    sampled_files = valid_files[:sample_num]
    
    # 处理文件复制和路径记录
    unix_paths = []
    for rel_src_path in sampled_files:
        # 源路径和目标路径
        abs_src_path = os.path.join(abs_input, rel_src_path)
        abs_dst_path = os.path.join(abs_output, rel_src_path)
        os.makedirs(os.path.dirname(abs_dst_path), exist_ok=True)
        
        # 处理文件名冲突
        base, ext = os.path.splitext(abs_dst_path)
        counter = 1
        while os.path.exists(abs_dst_path):
            abs_dst_path = f"{base}_{counter}{ext}"
            counter += 1
        
        # 复制文件
        if copy_files:
            shutil.copy2(abs_src_path, abs_dst_path)
        
        # 计算相对于output_txt所在目录的路径，并强制转换为Unix格式
        rel_to_txt = os.path.relpath(abs_dst_path, start=output_txt_dir)
        unix_path = rel_to_txt.replace(os.path.sep, '/')
        unix_paths.append(unix_path)
    
    # 写入datasets.txt（强制使用Unix路径分隔符）
    with open(abs_output_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(unix_paths))
    
    print(f"操作完成:\n"
          f"- 已处理 {len(unix_paths)} 张图片\n"
          f"- 目标文件夹: {abs_output}\n"
          f"- datasets.txt 路径: {abs_output_txt}\n"
          f"- 示例路径: {unix_paths[0] if unix_paths else '无'}")

if __name__ == "__main__":
    # 示例配置
    config = {
        "input_folder": "./Data/227/image",  # 源文件夹
        "output_folder": "./Data/227/2_27",  # 输出文件夹
        "output_txt": "./Data/227/227.txt",     # 输出文件
        "sample_num": 100,
        "target_size": None,  # 可选尺寸调整
        "copy_files": True
    }
    
    generate_quantization_dataset(**config)