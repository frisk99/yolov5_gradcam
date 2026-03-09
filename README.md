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
        print(f" ├─ GPU 显存 (VRAM): {vram_mb:.2f} MB")
        print(f" └─ 显存峰值 (Max VRAM): {vram_max_mb:.2f} MB\n")
    else:
        print(f"[{stage}]\n └─ 系统内存 (RAM): {ram_mb:.2f} MB (未检测到 GPU)\n")

# ================= 测试开始 =================

print_memory_usage("1. 初始化/加载模型前")

# 注意：这里必须使用 Qwen 的 VL (视觉) 版本，例如 Qwen2-VL-2B-Instruct
model_id = "Qwen/Qwen2-VL-2B-Instruct" 

# 视觉模型需要 Processor 来同时处理文本和图片
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# 加载视觉语言模型
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

print_memory_usage("2. 模型加载完成 (静态显存)")

# ================= 遍历文件夹 =================

# 替换为你存放图片的实际文件夹路径
image_folder = "./my_images" 
# 匹配常见的图片格式
image_paths = glob.glob(os.path.join(image_folder, "*.[jp][pn]*g")) # 匹配 jpg, jpeg, png

if not image_paths:
    print(f"在 {image_folder} 下没有找到图片文件。")
else:
    print(f"找到 {len(image_paths)} 张图片，开始逐一处理...\n")

    for idx, img_path in enumerate(image_paths):
        print(f"--- 正在处理第 {idx+1} 张图片: {os.path.basename(img_path)} ---")
        
        # 读取图片
        image = Image.open(img_path).convert("RGB")
        
        # 构建符合 Qwen-VL 格式的对话输入
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "请详细描述这张图片的内容。"}
            ]}
        ]
        
        # 使用 Processor 将图片和文本转换为模型输入张量
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)
        
        print_memory_usage(f"3. 准备好输入张量 (图片 {idx+1})")
        
        # 推理生成
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            # 截断 input_ids 部分，只保留生成的回复
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            
        print(f"🤖 模型描述: {response}\n")
        print_memory_usage(f"4. 推理生成完成 (图片 {idx+1})")
        
        # [关键步骤] 清理单次循环产生的缓存，防止处理多张图片时显存不断叠加导致 OOM
        del inputs
        del generated_ids
        del generated_ids_trimmed
        torch.cuda.empty_cache()

print("所有图片处理完毕！")
