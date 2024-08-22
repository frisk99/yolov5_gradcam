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
import numpy as np
import onnxruntime as ort
import torch
import argparse
from pathlib import Path
from diffusers import StableDiffusionPipeline

def compare_outputs(torch_output, onnx_output):
    # Convert Torch output to numpy array
    torch_output = torch_output.detach().cpu().numpy()
    
    # Ensure the outputs have the same shape
    assert torch_output.shape == onnx_output.shape, f"Shape mismatch: {torch_output.shape} vs {onnx_output.shape}"

    # Calculate the relative error
    relative_error = np.mean(np.abs(torch_output - onnx_output) / np.maximum(np.abs(torch_output), np.abs(onnx_output)))
    return relative_error

@torch.no_grad()
def test_model_precision(model_path: str, onnx_path: str, fp16: bool = False):
    dtype = torch.float16 if fp16 else torch.float32
    device = "cuda" if fp16 and torch.cuda.is_available() else "cpu"
    
    # Load the pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=dtype).to(device)

    # --- UNET ---
    sample_input = torch.randn(1, pipeline.unet.config.in_channels, pipeline.unet.config.sample_size, pipeline.unet.config.sample_size).to(device)
    torch_unet_output = pipeline.unet(sample_input, torch.tensor([1.0]).to(device), torch.randn(1, 77, 768).to(device))[0]
    
    ort_unet_session = ort.InferenceSession(str(Path(onnx_path) / "unet" / "model.onnx"))
    ort_unet_inputs = {
        "sample": sample_input.cpu().numpy(),
        "timestep": np.array([1.0]),
        "encoder_hidden_states": np.random.randn(1, 77, 768).astype(np.float32)
    }
    onnx_unet_output = ort_unet_session.run(None, ort_unet_inputs)[0]
    
    unet_error = compare_outputs(torch_unet_output, onnx_unet_output)
    print(f"Relative error between Torch and ONNX UNet outputs: {unet_error}")
    
    # --- TEXT ENCODER ---
    text_input = pipeline.tokenizer(
        "A sample prompt",
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    
    torch_text_output = pipeline.text_encoder(text_input.input_ids)[0]
    
    ort_text_session = ort.InferenceSession(str(Path(onnx_path) / "text_encoder" / "model.onnx"))
    ort_text_inputs = {"input_ids": text_input.input_ids.cpu().numpy()}
    onnx_text_output = ort_text_session.run(None, ort_text_inputs)[0]
    
    text_error = compare_outputs(torch_text_output, onnx_text_output)
    print(f"Relative error between Torch and ONNX Text Encoder outputs: {text_error}")

    # --- VAE ENCODER ---
    vae_input = torch.randn(1, pipeline.vae.config.in_channels, pipeline.vae.config.sample_size, pipeline.vae.config.sample_size).to(device)
    torch_vae_encoder_output = pipeline.vae.encode(vae_input)[0].sample()
    
    ort_vae_encoder_session = ort.InferenceSession(str(Path(onnx_path) / "vae_encoder" / "model.onnx"))
    ort_vae_encoder_inputs = {"sample": vae_input.cpu().numpy()}
    onnx_vae_encoder_output = ort_vae_encoder_session.run(None, ort_vae_encoder_inputs)[0]
    
    vae_encoder_error = compare_outputs(torch_vae_encoder_output, onnx_vae_encoder_output)
    print(f"Relative error between Torch and ONNX VAE Encoder outputs: {vae_encoder_error}")

    # --- VAE DECODER ---
    torch_vae_decoder_output = pipeline.vae.decode(torch_vae_encoder_output)
    
    ort_vae_decoder_session = ort.InferenceSession(str(Path(onnx_path) / "vae_decoder" / "model.onnx"))
    ort_vae_decoder_inputs = {"latent_sample": onnx_vae_encoder_output}
    onnx_vae_decoder_output = ort_vae_decoder_session.run(None, ort_vae_decoder_inputs)[0]
    
    vae_decoder_error = compare_outputs(torch_vae_decoder_output, onnx_vae_decoder_output)
    print(f"Relative error between Torch and ONNX VAE Decoder outputs: {vae_decoder_error}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True, help="Path to the Torch model.")
    parser.add_argument("--onnx_path", type=str, required=True, help="Path to the ONNX model.")
    parser.add_argument("--fp16", action="store_true", default=False, help="Use float16 precision for Torch model.")
    
    args = parser.parse_args()
    
    test_model_precision(args.model_path, args.onnx_path, args.fp16)

