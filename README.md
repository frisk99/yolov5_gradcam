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
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from tensorflow import keras

# Initialize the Stable Diffusion model
stable_diffusion = keras_cv.models.StableDiffusion()

# Set the image and mask directories
image_folder = 'path_to_image_folder'
mask_folder = 'path_to_mask_folder'

# Function to resize image and mask
def resize_image_and_mask(image, mask, size=(512, 512)):
    image = image.resize(size)
    mask = mask.resize(size)
    image = np.array(image)
    mask = np.array(mask)
    return image, mask

# Function to perform inpainting and plot results
def inpaint_and_plot(image, mask, prompt):
    mask = np.where(mask == 0, 1, 0)  # Inverting the mask
    image = np.expand_dims(image, axis=0)
    mask = np.expand_dims(mask, axis=0)

    generated = stable_diffusion.inpaint(
        prompt,
        image=image,
        mask=mask,
    )

    # Plot the result
    plt.imshow(generated[0])
    plt.axis('off')
    plt.show()

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

        # Perform inpainting and plot the results
        inpaint_and_plot(image, mask, prompt="pig on cart")


