import json
import numpy as np
from pycocotools.coco import COCO

tasks = ['test', 'train', 'valid']
num = 0
for i in tasks:

    # 1. 加载COCO数据集
    coco_annotation_file = f'./422/{i}/_annotations.coco.json'  # 替换为你的COCO标注文件路径
    coco = COCO(coco_annotation_file)

    # 2. 获取所有图像ID和对应的标注
    image_ids = coco.getImgIds()
    images = coco.loadImgs(image_ids)

    # 3. 转换为LabelMe格式
    labelme_format = []

    for img in images:
        image_id = img['id']
        file_name = img['file_name']
        width = img['width']
        height = img['height']

        # 获取当前图像的标注
        ann_ids = coco.getAnnIds(imgIds=image_id)
        anns = coco.loadAnns(ann_ids)

        shapes = []
        for ann in anns:
            category_id = ann['category_id']
            category_name = coco.loadCats(category_id)[0]['name']
            segmentation = ann['segmentation']

            # 将COCO的多边形点转换为LabelMe格式
            for seg in segmentation:
                points = np.array(seg).reshape((-1, 2)).tolist()  # 将点转换为[[x1, y1], [x2, y2], ...]格式
                shape = {
                    "label": category_name,
                    "points": points,
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {}
                }
                shapes.append(shape)

        # 构建LabelMe格式的JSON
        labelme_image = {
            "version": "4.5.6",
            "flags": {},
            "shapes": shapes,
            "imagePath": file_name,
            "imageData": None,  # 如果需要嵌入图像数据，可以在这里处理
            "imageHeight": height,
            "imageWidth": width
        }

        labelme_format.append(labelme_image)

    # 4. 保存LabelMe格式的JSON文件
    output_dir = f'./422/polygon'  # 替换为你的输出目录
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, labelme_image in enumerate(labelme_format):
        output_file = os.path.join(output_dir, f'{os.path.splitext(labelme_image["imagePath"])[0]}.json')
        with open(output_file, 'w') as f:
            json.dump(labelme_image, f, indent=2)
    num += len(labelme_format)

print(f"转换完成！共生成 {num} 个LabelMe格式的JSON文件。")