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
from pycocotools.coco import COCO
import random
import os
import shutil

# 配置路径
annotations_path = 'annotations/instances_train2017.json'
images_dir = 'train2017'
output_dir = 'selected_images'

# 初始化 COCO 数据
coco = COCO(annotations_path)

# 获取 "person" 类别的 ID
person_category_id = coco.getCatIds(catNms=['person'])[0]

# 获取包含 "person" 的所有图片 ID
image_ids = coco.getImgIds(catIds=[person_category_id])

# 随机选择 100 张图片
selected_image_ids = random.sample(image_ids, 100)

# 获取图片信息
selected_images = coco.loadImgs(selected_image_ids)

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 复制选定的图片到输出目录
for img in selected_images:
    src_path = os.path.join(images_dir, img['file_name'])
    dst_path = os.path.join(output_dir, img['file_name'])
    shutil.copy(src_path, dst_path)

print(f"已复制 {len(selected_images)} 张图片到 {output_dir}")

