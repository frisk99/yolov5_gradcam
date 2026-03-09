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
from transformers import AutoModelForCausalLM, AutoTokenizer

def print_memory_usage(stage="当前状态"):
    """打印当前进程的 RAM 和 GPU 显存占用"""
    # 1. 获取系统内存 (RAM) 占用
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / (1024 ** 2)
    
    # 2. 获取显存 (VRAM) 占用
    if torch.cuda.is_available():
        # 当前分配的显存
        vram_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        # 历史峰值显存（非常重要，因为推理时会产生临时显存峰值）
        vram_max_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        print(f"[{stage}]")
        print(f" ├─ 系统内存 (RAM): {ram_mb:.2f} MB")
        print(f" ├─ GPU 显存 (VRAM): {vram_mb:.2f} MB")
        print(f" └─ 显存峰值 (Max VRAM): {vram_max_mb:.2f} MB\n")
    else:
        print(f"[{stage}]\n └─ 系统内存 (RAM): {ram_mb:.2f} MB (未检测到可用的 GPU)\n")

# ================= 测试开始 =================

print_memory_usage("1. 初始化/加载模型前")

# 替换为实际的 Qwen3 2B 模型路径或 Hugging Face ID
model_id = "Qwen/Qwen3-2B-Instruct"  # 示例名称，请根据实际情况修改

# 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# 加载模型 (建议使用 float16 或 bfloat16 以节省一半显存)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",       # 自动将模型加载到可用的 GPU 上
    torch_dtype=torch.float16, # 使用半精度
    trust_remote_code=True
)

print_memory_usage("2. 模型加载完成 (静态显存)")

# 准备测试数据并进行推理
text = "你好，请详细介绍一下量子计算的基本原理。"
inputs = tokenizer(text, return_tensors="pt").to(model.device)

# 生成回复
with torch.no_grad(): # 推理时务必关闭梯度计算，否则会爆显存
    outputs = model.generate(
        **inputs, 
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7
    )

print_memory_usage("3. 推理生成完成 (包含 KV Cache 显存)")

# （可选）清理显存缓存
# del model
# del inputs
# del outputs
# torch.cuda.empty_cache()
# print_memory_usage("4. 清理缓存后")
