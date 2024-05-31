# YOLO-V5 GRADCAM

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

def yolo_to_coco(yolo_dir, image_dir, output_file):
    # 初始化COCO格式的字典
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # 添加类别
    for i in range(1, 81):  # YOLO类从0开始，COCO类从1开始
        coco_data["categories"].append({
            "id": i,
            "name": f"class_{i}"
        })
    # 添加第81类
    coco_data["categories"].append({
        "id": 81,
        "name": "class_81"
    })

    annotation_id = 1
    for idx, filename in enumerate(os.listdir(yolo_dir)):
        if filename.endswith(".txt"):
            image_id = idx + 1
            image_filename = filename.replace(".txt", ".jpg")
            image_path = os.path.join(image_dir, image_filename)
            
            # 获取图像尺寸
            from PIL import Image
            image = Image.open(image_path)
            width, height = image.size
            
            # 添加图像信息到COCO数据中
            coco_data["images"].append({
                "id": image_id,
                "file_name": image_filename,
                "width": width,
                "height": height
            })
            
            # 读取YOLO标签文件
            yolo_file_path = os.path.join(yolo_dir, filename)
            with open(yolo_file_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    x_center, y_center, bbox_width, bbox_height = map(float, parts[1:])
                    
                    # 转换坐标
                    x_min = (x_center - bbox_width / 2) * width
                    y_min = (y_center - bbox_height / 2) * height
                    bbox_width *= width
                    bbox_height *= height
                    
                    # 修改第0类为第81类
                    if class_id == 0:
                        class_id = 81
                    
                    # 添加标注信息到COCO数据中
                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_id,
                        "bbox": [x_min, y_min, bbox_width, bbox_height],
                        "area": bbox_width * bbox_height,
                        "iscrowd": 0
                    })
                    annotation_id += 1
    
    # 保存COCO格式的JSON文件
    with open(output_file, "w") as f:
        json.dump(coco_data, f, indent=4)

# 使用示例
yolo_dir = "path/to/yolo/labels"  # 替换为YOLO标签文件的路径
image_dir = "path/to/images"  # 替换为图像文件的路径
output_file = "path/to/output/coco_annotations.json"  # 替换为输出COCO文件的路径

yolo_to_coco(yolo_dir, image_dir, output_file)