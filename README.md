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
import torch

# 固定随机数种子
torch.manual_seed(42)

# 读取文件中的数据
input_file = 'input.txt'
output_file = 'output.txt'

with open(input_file, 'r') as file:
    # 读取整行数据并转换为浮点数列表
    line = file.readline()
    numbers = list(map(float, line.split()))

# 使用torch生成随机浮点数
random_numbers = torch.rand(len(numbers))

# 计算差值
differences = [num - rand_num for num, rand_num in zip(numbers, random_numbers)]

# 将差值转换为字符串，并拼接成一行文本
output_line = ' '.join(map(str, differences))

# 将结果写入输出文件
with open(output_file, 'w') as file:
    file.write(output_line)

print("处理完成，结果已保存到", output_file)
import tensorflow as tf
import keras_cv
from tensorflow import keras
import numpy as np

# Load the pipeline and get models
model = keras_cv.models.StableDiffusionV2(img_width=512, img_height=512)
text_encoder_model = model.text_encoder
decoder_model = model.decoder
diffusion_model = model.diffusion_model
image_encoder_model = model.image_encoder

def compare_models(keras_model, tflite_model_path, input_data):
    keras_output = keras_model(input_data).numpy()
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 打印输入张量的详细信息
    print('###################################')
    for i, input_detail in enumerate(input_details):
        print(f"Input {i}:")
        print(f"  Name: {input_detail['name']}")
        print(f"  Shape: {input_detail['shape']}")
        print(f"  Data Type: {input_detail['dtype']}")

    # 设置输入张量
    if isinstance(input_data, (list, tuple)):
        for i, data in enumerate(input_data):
            interpreter.set_tensor(input_details[i]['index'], data)
    else:
        interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_details[0]['index'])
    difference = np.mean(np.abs((keras_output - tflite_output) / keras_output)) * 100
    return difference

def generate_random_inputs():
    text_encoder_input = [np.random.random((1, 77)).astype(np.int32) for _ in range(2)]
    diffusion_input = [
        np.random.normal(loc=0.0, scale=1.0, size=(1, 64, 64, 4)).astype(np.float32),
        np.random.normal(loc=0.0, scale=1.0, size=(1, 320)).astype(np.float32),
        np.random.normal(loc=0.0, scale=1.0, size=(1, 77, 1024)).astype(np.float32)
    ]
    decoder_input = np.random.normal(loc=0.0, scale=1.0, size=(1, 64, 64, 4)).astype(np.float32)
    image_encoder_input = np.random.normal(loc=0.0, scale=1.0, size=(1, 512, 512, 3)).astype(np.float32)

    return text_encoder_input, diffusion_input, decoder_input, image_encoder_input

cnt = 12
text_encoder_differences = []
diffusion_differences = []
decoder_differences = []
image_encoder_differences = []

for _ in range(cnt):
    text_encoder_input, diffusion_input, decoder_input, image_encoder_input = generate_random_inputs()

    text_encoder_differences.append(compare_models(text_encoder_model, './tmp512/sd2_text_encoder_dynamic.tflite', text_encoder_input))
    diffusion_differences.append(compare_models(diffusion_model, './tmp512/sd2_diffusion_model_dynamic.tflite', diffusion_input))
    decoder_differences.append(compare_models(decoder_model, './tmp512/sd2_decoder_dynamic.tflite', decoder_input))
    image_encoder_differences.append(compare_models(image_encoder_model, './tmp512/sd2_image_encoder_dynamic.tflite', image_encoder_input))

# 计算并打印平均差异
print(text_encoder_differences)
print(f"Text Encoder 模型差异: {np.mean(text_encoder_differences):.2f}%")
print(diffusion_differences)
print(f"Diffusion 模型差异: {np.mean(diffusion_differences):.2f}%")
print(decoder_differences)
print(f"Decoder 模型差异: {np.mean(decoder_differences):.2f}%")
print(image_encoder_differences)
print(f"Image Encoder 模型差异: {np.mean(image_encoder_differences):.2f}%")

print(f"Decoder 模型差异: {decoder_difference / cnt:.2f}%")
print(f"Image Encoder 模型差异: {image_encoder_difference / cnt:.2f}%")
