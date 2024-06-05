2# YOLO-V5 GRADCAM

I constantly desired to know to which part of an object the object-detection models pay more attention. So I searched for it, but I didn't find any for Yolov5.
Here is my implementation of Grad-cam for YOLO-v5. To load the model I used the yolov5's main codes, and for computing GradCam I used the codes from the gradcam_plus_plus-pytorch repository.
Please follow my GitHub account and star ⭐ the project if this functionality benefits your research or projects.

## Update:
Repo works fine with yolov5-v6.1


## Installation
`pip install -r requirements.txt`

## Infer
`python main.py --model-path yolov5s.pt --img-path images/cat-dog.jpg --output-dir outputs`

**NOTE**: If you don't have any weights and just want to test, don't change the model-path argument. The yolov5s model will be automatically downloaded thanks to the download function from yolov5. 

**NOTE**: For more input arguments, check out the main.py or run the following command:

```python main.py -h```

### Custom Name
To pass in your custom model you might want to pass in your custom names as well, which be done as below:
```
python main.py --model-path cutom-model-path.pt --img-path img-path.jpg --output-dir outputs --names obj1,obj2,obj3 
```
## Examples
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pooya-mohammadi/yolov5-gradcam/blob/master/main.ipynb)

<img src="https://raw.githubusercontent.com/pooya-mohammadi/yolov5-gradcam/master/outputs/eagle-res.jpg" alt="cat&dog" height="300" width="1200">
<img src="https://raw.githubusercontent.com/pooya-mohammadi/yolov5-gradcam/master/outputs/cat-dog-res.jpg" alt="cat&dog" height="300" width="1200">
<img src="https://raw.githubusercontent.com/pooya-mohammadi/yolov5-gradcam/master/outputs/dog-res.jpg" alt="cat&dog" height="300" width="1200">

## Note
I checked the code, but I couldn't find an explanation for why the truck's heatmap does not show anything. Please inform me or create a pull request if you find the reason.

This problem is solved in version 6.1

Solve the custom dataset gradient not match.

# References
1. https://github.com/1Konny/gradcam_plus_plus-pytorch
2. https://github.com/ultralytics/yolov5
3. https://github.com/pooya-mohammadi/deep_utils
4. https://github.com/pooya-mohammadi/yolov5-gradcam
```python
import os
import json

def merge_yolo_coco(yolo_dir, coco_annotation_file, output_file):
    # 读取原始COCO注释文件
    with open(coco_annotation_file, 'r') as f:
        coco_data = json.load(f)

    annotation_id = max([anno['id'] for anno in coco_data['annotations']]) + 1
    image_id_map = {img['file_name']: img['id'] for img in coco_data['images']}
    category_id_map = {cat['name']: cat['id'] for cat in coco_data['categories']}
    
    # 添加新的类别81
    if 81 not in [cat['id'] for cat in coco_data['categories']]:
        coco_data['categories'].append({
            "id": 81,
            "name": "class_81"
        })

    # 读取YOLO结果文件并转换
    for filename in os.listdir(yolo_dir):
        if filename.endswith(".txt"):
            image_filename = filename.replace(".txt", ".jpg")
            if image_filename not in image_id_map:
                continue
            
            image_id = image_id_map[image_filename]
            image_info = next(img for img in coco_data['images'] if img['id'] == image_id)
            width = image_info['width']
            height = image_info['height']
            
            yolo_file_path = os.path.join(yolo_dir, filename)
            with open(yolo_file_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    x_center, y_center, bbox_width, bbox_height = map(float, parts[1:])
                    
                    x_min = (x_center - bbox_width / 2) * width
                    y_min = (y_center - bbox_height / 2) * height
                    bbox_width *= width
                    bbox_height *= height
                    
                    # 修改第0类为第81类
                    if class_id == 0:
                        class_id = 81
                    
                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_id,
                        "bbox": [x_min, y_min, bbox_width, bbox_height],
                        "area": bbox_width * bbox_height,
                        "iscrowd": 0
                    })
                    annotation_id += 1

    # 保存合并后的注释文件
    with open(output_file, "w") as f:
        json.dump(coco_data, f, indent=4)

# 使用示例
yolo_dir = "path/to/yolo/labels"  # 替换为YOLO标签文件的路径
coco_annotation_file = "path/to/coco/annotations.json"  # 替换为COCO注释文件的路径
output_file = "path/to/output/merged_annotations.json"  # 替换为输出合并文件的路径

merge_yolo_coco(yolo_dir, coco_annotation_file, output_file)
pip install torch==1.13.1+cu118 torchvision==0.14.1+cu118 torchaudio==0.13.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html -i
import json
from collections import defaultdict

# 加载COCO数据集的标注文件
with open('path_to_your_coco_annotations_file.json', 'r') as f:
    coco_data = json.load(f)

# 提取类别信息
categories = coco_data['categories']
category_id_to_name = {category['id']: category['name'] for category in categories}

# 初始化类别计数，包括所有类别
category_count = {category_id: 0 for category_id in category_id_to_name.keys()}

# 统计每个类别的数量
for annotation in coco_data['annotations']:
    category_id = annotation['category_id']
    category_count[category_id] += 1

# 输出结果，包括数量为0的类别
for category_id, count in category_count.items():
    print(f"Category: {category_id_to_name[category_id]}, Count: {count}")
