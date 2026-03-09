# YOLO-V5 GRADCAM

I constantly desired to know to which part of an object the object-detection models pay more attention. So I searched for it, but I didn't find any for Yolov5.
Here is my implementation of Grad-cam for YOLO-v5. To load the model I used the yolov5's main codes, and for computing GradCam I used the codes from the gradcam_plus_plus-pytorch repository.
Please follow my GitHub account and star ⭐ the project if this functionality benefits your research or projects.
light-toned wood, likely a natural or lightly stained wood species, top-down view, overhead perspective, flat angle, clear wood grain texture, realistic lighting, high detail


wall with wallpaper only, front view, flat angle, light-toned wallpaper, photo-realistic, high resolution  
Negative prompt: floor, ceiling, furniture, window, door, people, clutter

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
```python
import torch
import psutil
import os
import glob
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

def print_memory_usage(stage="当前状态"):
    """打印当前进程的 RAM 和 GPU 显存占用"""
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / (1024 ** 2)
    
    if torch.cuda.is_available():
        vram_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        vram_max_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"[{stage}]")
        print(f" ├─ 系统内存 (RAM): {ram_mb:.2f} MB")
        print(f" ├─ GPU 当前显存 (VRAM): {vram_mb:.2f} MB")
        print(f" └─ GPU 显存峰值 (Max VRAM): {vram_max_mb:.2f} MB\n")
    else:
        print(f"[{stage}]\n └─ 系统内存 (RAM): {ram_mb:.2f} MB (未检测到 GPU)\n")

# ================= 1. 初始化模型 =================

print_memory_usage("1. 初始化/加载模型前")

model_id = "Qwen/Qwen2-VL-2B-Instruct" 
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

print_memory_usage("2. 模型加载完成 (静态显存)")

# ================= 2. 读取所有图片 =================

image_folder = "./my_images"  # 替换为你的文件夹路径
image_paths = glob.glob(os.path.join(image_folder, "*.[jp][pn]*g"))

if not image_paths:
    print(f"在 {image_folder} 下没有找到图片文件。")
    exit()

print(f"找到 {len(image_paths)} 张图片，正在准备一次性输入...")

# 加载所有图片对象
images = [Image.open(img_path).convert("RGB") for img_path in image_paths]

# 构建 content 列表：包含所有的图片，以及最后的一个文本提问
content_list = [{"type": "image", "image": img} for img in images]
content_list.append({"type": "text", "text": "请结合以上所有图片，总结它们共同的主题，并分别简述每张图片的内容。"})

messages = [
    {
        "role": "user", 
        "content": content_list
    }
]

# ================= 3. 处理输入与推理 =================

# 重置显存峰值统计，以便单独测量本次多图推理的峰值
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

# 预处理输入
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# 将所有图片传给 processor
inputs = processor(text=[text], images=images, padding=True, return_tensors="pt").to(model.device)

print_memory_usage("3. 准备好包含所有图片的输入张量")

# 推理生成
with torch.no_grad():
    print("正在进行多图推理，请耐心等待...")
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    
    # 截取新生成的内容
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print(f"🤖 模型回复:\n{response}\n")

print_memory_usage("4. 推理完成 (查看此时的显存峰值)")

# ================= 4. 清理 =================
del inputs
del generated_ids
del generated_ids_trimmed
torch.cuda.empty_cache()
print_memory_usage("5. 清除张量缓存后")