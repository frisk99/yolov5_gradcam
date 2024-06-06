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
import json
import os

def is_intersecting(bbox1, bbox2):
    x1_min, y1_min, w1, h1 = bbox1
    x1_max, y1_max = x1_min + w1, y1_min + h1
    
    x2_min, y2_min, w2, h2 = bbox2
    x2_max, y2_max = x2_min + w2, y2_min + h2

    intersect = not (x1_min > x2_max or x1_max < x2_min or y1_min > y2_max or y1_max < y2_min)
    return intersect

# 定义路径
ann_file = 'coco91_val.json'
# 读取JSON文件
with open(ann_file, 'r') as f:
    coco_data = json.load(f)
# 获取类别id
person_category_id = next(cat['id'] for cat in coco_data['categories'] if cat['name'] == 'person')
specific_category_id = next(cat['id'] for cat in coco_data['categories'] if cat['name'] == 'head') 
print(specific_category_id)
# 初始化要移除的标注列表
annotations_to_remove = []
remove_cnt = 0
# 遍历所有图像
for img in coco_data['images']:
    img_id = img['id']
    print(img_id)
    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]
    person_annotations = [ann for ann in annotations if ann['category_id'] == person_category_id]
    specific_annotations = [ann for ann in annotations if ann['category_id'] == specific_category_id]
    for specific_ann in specific_annotations:
        specific_bbox = specific_ann['bbox']
        has_intersection = False
        for person_ann in person_annotations:
            person_bbox = person_ann['bbox']
            if is_intersecting(specific_bbox, person_bbox):
                has_intersection = True
                break
        if not has_intersection:
            annotations_to_remove.append(specific_ann['id'])
            print("remove!")
            remove_cnt = remove_cnt +1

# 移除不符合条件的标注
coco_data['annotations'] = [ann for ann in coco_data['annotations'] if ann['id'] not in annotations_to_remove]

# 保存修改后的注释文件
output_file = 'coco_91_val_checked_1.json'
with open(output_file, 'w') as f:
    json.dump(coco_data, f)

print(f"Filtered annotations saved to {output_file}")
print(f"remove {remove_cnt} !")

# 输出结果，包括数量为0的类别
for category_id, count in category_count.items():
    print(f"Category: {category_id_to_name[category_id]}, Count: {count}")
eyJ2Ijoid2lufDEuOC4xMCIsImkiOiIyRXRLeVJKeGlkIiwibCI6IlNISS1aSE9VMDEgfCBzaGkuemhvdSB8IFdpbmRvd3MifQ==
