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

def print_memory_usage(stage="当前状态"):
    """打印当前进程的 RAM 和 GPU 显存占用，包含各自的峰值"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    # 1. 获取当前系统内存 (RAM)
    ram_mb = mem_info.rss / (1024 ** 2)
    
    # 2. 获取系统内存峰值 (Max RAM)
    # 根据不同操作系统获取不同的底层属性
    if hasattr(mem_info, 'peak_wset'):
        # Windows 系统
        max_ram_mb = mem_info.peak_wset / (1024 ** 2)
    elif hasattr(mem_info, 'hwm'):
        # Linux 系统
        max_ram_mb = mem_info.hwm / (1024 ** 2)
    else:
        # macOS 或其他不支持直接获取峰值的系统 (Fallback fallback为当前值)
        max_ram_mb = ram_mb 
        
    print(f"[{stage}]")
    print(f" ├─ 系统内存 (RAM): {ram_mb:.2f} MB")
    print(f" ├─ 内存峰值 (Max RAM): {max_ram_mb:.2f} MB")
    
    # 3. 获取 GPU 显存 (VRAM)
    if torch.cuda.is_available():
        vram_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        vram_max_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        print(f" ├─ GPU 显存 (VRAM): {vram_mb:.2f} MB")
        print(f" └─ 显存峰值 (Max VRAM): {vram_max_mb:.2f} MB\n")
    else:
        print(f" └─ (未检测到可用的 GPU)\n")
