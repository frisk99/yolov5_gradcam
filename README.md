1# YOLO-V5 GRADCAM

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