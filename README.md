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
#!/bin/bash

# 查找当前文件夹及其子文件夹中的所有 bin/pip 和 bin/pip3 开头的文件
find . -type f -path "*/bin/pip*" | while read -r file; do
  # 使用 sed 命令替换文件中的内容
  sed -i 's|!/home|!/data1|g' "$file"
  echo "Processed $file"
done
import onnx
import onnxruntime as ort
from onnx import numpy_helper

# 读取ONNX模型
model_path = 'your_model_with_external_data.onnx'
onnx_model = onnx.load(model_path)

# 检查模型
onnx.checker.check_model(onnx_model)

# 创建ONNX Runtime会话
ort_session = ort.InferenceSession(model_path)

# 获取输入名称和形状
input_info = ort_session.get_inputs()
for input in input_info:
    print(f"Input name: {input.name}")
    print(f"Input shape: {input.shape}")
    print(f"Input type: {input.type}")

# 获取输出名称和形状
output_info = ort_session.get_outputs()
for output in output_info:
    print(f"Output name: {output.name}")
    print(f"Output shape: {output.shape}")
    print(f"Output type: {output.type}")

# 读取和打印模型中的初始化参数
for initializer in onnx_model.graph.initializer:
    if initializer.HasField('data_location') and initializer.data_location == onnx.TensorProto.EXTERNAL:
        print(f"Tensor name: {initializer.name} is stored externally.")
    else:
        tensor_array = numpy_helper.to_array(initializer)
        print(f"Tensor name: {initializer.name}")
        print(tensor_array)
import tensorflow as tf
import numpy as np

# 加载TFLite模型
interpreter = tf.lite.Interpreter(model_path="your_model.tflite")
interpreter.allocate_tensors()

# 获取输入和输出张量信息
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 获取输入张量的形状
input_shape = input_details[0]['shape']

# 根据输入形状生成随机数据
input_data = np.random.random_sample(input_shape).astype(np.float32)

# 将随机输入数据赋值给输入张量
interpreter.set_tensor(input_details[0]['index'], input_data)

# 运行模型
interpreter.invoke()

# 获取输出数据
output_data = interpreter.get_tensor(output_details[0]['index'])

# 打印输出数据
print("Output Data:", output_data)
from PIL import Image

def is_close_to_color(color, target_color, threshold):
    """判断颜色是否接近目标颜色"""
    return all(abs(c - t) < threshold for c, t in zip(color, target_color))

# 打开图片
img = Image.open("input.png")

# 将图片转换为RGB模式
img = img.convert("RGB")

# 获取图片的像素数据
pixels = img.load()

# 定义接近黑色和白色的阈值
black_threshold = 50  # 设定接近黑色的阈值
white_threshold = 50  # 设定接近白色的阈值

# 定义黑色和白色的RGB值
black_color = (0, 0, 0)
white_color = (255, 255, 255)

# 获取图片的尺寸
width, height = img.size

# 遍历图片的每个像素
for y in range(height):
    for x in range(width):
        r, g, b = pixels[x, y]
        current_color = (r, g, b)
        
        # 如果接近黑色，将其变为白色
        if is_close_to_color(current_color, black_color, black_threshold):
            pixels[x, y] = white_color
        # 如果接近白色，将其变为黑色
        elif is_close_to_color(current_color, white_color, white_threshold):
            pixels[x, y] = black_color

# 保存转换后的图片
img.save("output.png")
from PIL import Image
import numpy as np
def png_to_rgb(input_png_file, output_rgb_file):
    img = Image.open(input_png_file)
    img = img.convert('RGB')
    img_data = np.array(img)
    img_data = img_data.astype(np.uint8)
    with open(output_rgb_file, 'wb') as f:
        f.write(img_data.tobytes())
def read_rgb_file(filename, width, height):
    with open(filename, 'rb') as f:
        img_data = f.read()
    img = np.frombuffer(img_data, dtype=np.uint8)
    img = img.reshape((height, width, 3)) 
    return img

def rgb_to_png(input_rgb_file, output_png_file, width, height):
    img = read_rgb_file(input_rgb_file, width, height)
    img = Image.fromarray(img)
    img.save(output_png_file)

input_rgb_file = r'man-on-skateboard-cropped.rgb' 
output_png_file = r'output_file.png'  
width, height = 512, 512

#rgb_to_png(input_rgb_file, output_png_file, width, height)
png_to_rgb('000009.jpg','000009.rgb')
rgb_to_png('000009.rgb', '000009-1.jpg', 256, 256)
import os
import PIL
import requests
import torch
from io import BytesIO
from diffusers import AutoPipelineForInpainting,AutoPipelineForText2Image
from transformers import T5Model, T5Tokenizer
# Define folders
image_folder = r"G:\train_control\celeba_hq\train\female"
mask_folder = r"G:\train_control\celeba_hq\train\female_masks"
output_folder = r"G:\train_control\celeba_hq\train\female_close"

# Load images and masks
def load_image(image_path):
    return PIL.Image.open(image_path).convert("RGB")

# Initialize the pipeline
pipe_pre = AutoPipelineForText2Image.from_pretrained(
    "G:\huggingface\Juggernaut-XL-v8", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

prompt = "A person sleeping"
pipe = AutoPipelineForInpainting.from_pipe(pipe_pre).to("cuda")
# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Process each pair of image and mask
for image_name in os.listdir(image_folder):
    if image_name.endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(image_folder, image_name)
        mask_path = os.path.join(mask_folder, image_name) 

        if os.path.exists(mask_path):
            init_image = load_image(image_path).resize((512, 512))
            mask_image = load_image(mask_path).resize((512, 512))
            image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]
            output_path = os.path.join(output_folder, f"inpainted_{image_name}")
            image.save(output_path)

            print(f"Saved inpainted image to {output_path}")
        else:
            print(f"Mask not found for image: {image_name}")
import cv2
import numpy as np
import os

def enlarge_white_ellipse(img, scale_factor):
    # 查找白色区域的轮廓
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建一个空白图像，大小与原图一致
    result = np.zeros_like(img)

    for contour in contours:
        # 拟合轮廓为椭圆
        if contour.shape[0] >= 5:  # 拟合椭圆要求至少有5个点
            ellipse = cv2.fitEllipse(contour)
            
            # 放大椭圆的长短轴
            center, axes, angle = ellipse
            
            # 检查拟合出的轴是否有效
            if axes[0] > 0 and axes[1] > 0 and np.isfinite(axes[0]) and np.isfinite(axes[1]):
                axes = (int(axes[0] * scale_factor), int(axes[1] * scale_factor))
                # 在空白图像上绘制放大后的椭圆
                cv2.ellipse(result, (center, axes, angle), 255, thickness=cv2.FILLED)
            else:
                print(f"Skipping invalid contour with axes: {axes}")

    return result

def process_images_in_folder(input_folder, output_folder, scale_factor):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            # 读取图像
            img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            
            # 放大白色椭圆区域
            enlarged_img = enlarge_white_ellipse(img, scale_factor)
            
            # 保存结果
            cv2.imwrite(output_path, enlarged_img)
            print(f'Processed and saved: {output_path}')

# 使用示例
input_folder = 'path_to_input_folder'  # 输入文件夹路径
output_folder = 'path_to_output_folder'  # 输出文件夹路径
scale_factor = 1.5  # 放大系数

process_images_in_folder(input_folder, output_folder, scale_factor)

参加AI相关的入职培训让我对人工智能技术及其在实际应用中的广泛前景有了更深入的理解。这次培训涵盖了AI的基础知识，包括机器学习、深度学习、自然语言处理等核心概念，同时还深入探讨了AI模型的训练与优化流程。通过培训，我了解了如何有效利用开源工具和框架，如TensorFlow和PyTorch，来构建和部署AI模型。此外，培训中强调了数据的重要性，介绍了数据清洗、特征工程以及如何构建高质量的数据集以提高模型的准确性和泛化能力。

在实际操作部分，我们学习了如何从零开始构建AI应用，从数据预处理、模型训练到最终的模型部署。我还对AI模型在云端的部署、管理和优化有了初步的掌握，这为今后的实际开发工作奠定了基础。通过这次培训，我不仅巩固了AI的理论知识，还提升了实际操作技能，更重要的是，我开始认识到AI技术的潜力以及在解决现实世界问题中的重要性。未来，我将继续深入学习AI相关的技术，争取在工作中将其应用到更多的项目中，推动业务创新和效率提升。