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
﹉python
import json

# 读取COCO格式的JSON文件
def load_coco_annotations(file_path):
    with open(file_path, 'r') as f:
        coco_data = json.load(f)
    return coco_data

# 合并两个COCO数据集，并去重
def merge_coco_datasets(coco_data1, coco_data2):
    categories1 = coco_data1['categories']
    categories2 = coco_data2['categories']

    # 确保类别一致
    if categories1 != categories2:
        raise ValueError("The category lists of the two datasets are not identical.")
    
    merged_categories = categories1

    images1 = coco_data1['images']
    images2 = coco_data2['images']
    annotations1 = coco_data1['annotations']
    annotations2 = coco_data2['annotations']

    # 使用文件名、宽度和高度来去重图像
    image_key_map = {}
    merged_images = []
    merged_annotations = []

    current_image_id = 1
    for image in images1 + images2:
        image_key = (image['file_name'], image['width'], image['height'])
        if image_key not in image_key_map:
            image_key_map[image_key] = current_image_id
            new_image = image.copy()
            new_image['id'] = current_image_id
            merged_images.append(new_image)
            current_image_id += 1

    # 使用图像ID和边界框来去重注释
    annotation_key_map = {}
    current_annotation_id = 1
    for annotation in annotations1 + annotations2:
        annotation_key = (annotation['image_id'], tuple(annotation['bbox']))
        if annotation_key not in annotation_key_map:
            annotation_key_map[annotation_key] = current_annotation_id
            new_annotation = annotation.copy()
            new_annotation['id'] = current_annotation_id
            new_annotation['image_id'] = image_key_map[(annotation['image_id'],)]
            merged_annotations.append(new_annotation)
            current_annotation_id += 1

    # 创建合并后的COCO数据集
    merged_coco_data = {
        'images': merged_images,
        'annotations': merged_annotations,
        'categories': merged_categories
    }

    return merged_coco_data

# 将合并后的数据集保存为JSON文件
def save_merged_coco_annotations(output_file_path, merged_coco_data):
    with open(output_file_path, 'w') as f:
        json.dump(merged_coco_data, f)

# 示例处理函数
def process_and_merge_coco_annotations(input_file_path1, input_file_path2, output_file_path):
    coco_data1 = load_coco_annotations(input_file_path1)
    coco_data2 = load_coco_annotations(input_file_path2)
    merged_coco_data = merge_coco_datasets(coco_data1, coco_data2)
    save_merged_coco_annotations(output_file_path, merged_coco_data)

# 示例调用
process_and_merge_coco_annotations('path/to/your/first_coco_annotations.json', 'path/to/your/second_coco_annotations.json', 'path/to/your/merged_coco_annotations.json')
