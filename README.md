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
import os
import keras_cv
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow import keras

# Initialize the Stable Diffusion model
stable_diffusion = keras_cv.models.StableDiffusion()

# Set the image, mask, and output directories
image_folder = 'path_to_image_folder'
mask_folder = 'path_to_mask_folder'
output_folder = 'path_to_output_folder'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Function to resize image and mask
def resize_image_and_mask(image, mask, size=(512, 512)):
    image = image.resize(size)
    mask = mask.resize(size)
    image = np.array(image)
    mask = np.array(mask)
    return image, mask

# Function to perform inpainting, plot results, and save the image
def inpaint_and_save(image, mask, prompt, output_path):
    mask = np.where(mask == 0, 1, 0)  # Inverting the mask
    image = np.expand_dims(image, axis=0)
    mask = np.expand_dims(mask, axis=0)

    generated = stable_diffusion.inpaint(
        prompt,
        image=image,
        mask=mask,
    )

    # Convert the generated image to PIL format and save it
    generated_image = Image.fromarray((generated[0] * 255).astype(np.uint8))
    generated_image.save(output_path)
    print(f"Saved inpainted image to {output_path}")

# Loop through the images and masks
for filename in os.listdir(image_folder):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        image_path = os.path.join(image_folder, filename)
        mask_path = os.path.join(mask_folder, filename)  # Assuming masks have the same name as images

        # Load the image and mask
        image = Image.open(image_path)
        mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale

        # Resize image and mask
        image, mask = resize_image_and_mask(image, mask)

        # Define the output file path
        output_path = os.path.join(output_folder, filename)

        # Perform inpainting and save the results
        inpaint_and_save(image, mask, prompt="glancing at something", output_path=output_path)

import gradio as gr
import imageio

def dummy(img):
  imageio.imwrite("output_image.png", img["mask"])
  return img["image"], img["mask"]

with gr.Blocks() as demo:
  with gr.Row():
    img = gr.Image(tool="sketch", label="base image", show_label=True)
    img1 = gr.Image()
    img2 = gr.Image(label="mask image", show_label=True)
  btn = gr.Button()
  btn.click(dummy, img, [img1, img2])

demo.launch(debug=True)
import tensorflow as tf

# Load your TFLite model
interpreter = tf.lite.Interpreter(model_path="your_model.tflite")
interpreter.allocate_tensors()

# Get model details
tensor_details = interpreter.get_tensor_details()

# Calculate the total number of parameters
total_params = 0
for layer in tensor_details:
    if 'weights' in layer['name'] or 'bias' in layer['name']:
        total_params += layer['shape'].num_elements()

print(f"Total parameters: {total_params}")
from safetensors.torch import load_file
import torch

# 加载 SafeTensor 文件
model_path = "path_to_your_model.safetensors"
state_dict = load_file(model_path)

# 计算参数量
total_params = 0
for param_name, param_tensor in state_dict.items():
    total_params += param_tensor.numel()  # 统计每个张量的元素数量

print(f"模型的总参数量: {total_params}")