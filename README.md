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
import json
def merge_coco_jsons(json1_path, json2_path, output_path):
    with open(json1_path, 'r') as f:
        coco1 = json.load(f)
    
    with open(json2_path, 'r') as f:
        coco2 = json.load(f)
    
    # 以第一个JSON中的图片为准
    image_ids_in_coco1 = set(img['id'] for img in coco1['images'])
    
    # 假设coco2中只有一个类别，并且需要设置为ID 81
    if len(coco2['categories']) == 1:
        head_category_id = coco2['categories'][0]['id']
    else:
        raise ValueError("coco2 should contain only one category")
    
    # 合并annotations
    merged_annotations = coco1['annotations']
    for annotation in coco2['annotations']:
        if annotation['image_id'] in image_ids_in_coco1:
            # 如果注释的类别ID是head的类别ID，则修改为81
            if annotation['category_id'] == head_category_id:
                annotation['category_id'] = 81
            merged_annotations.append(annotation)
    
    # 更新coco1的annotations
    coco1['annotations'] = merged_annotations
    
    # 添加head类别到categories
    head_category = {
        "id": 81,
        "name": "head",
        "supercategory": "none"
    }
    coco1['categories'].append(head_category)
    
    # 保存合并后的JSON
    with open(output_path, 'w') as f:
        json.dump(coco1, f, indent=4)

# 使用示例
merge_coco_jsons('coco1.json', 'coco2.json', 'merged_coco.json')
