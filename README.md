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
1. https://github.com/1Konny/gradcam_plus_plus-pytorch
2. https://github.com/ultralytics/yolov5
3. https://github.com/pooya-mohammadi/deep_utils
4. https://github.com/pooya-mohammadi/yolov5-gradcam




```cpp
import subprocess
import time
import os

def run_with_file_monitor(output_file="output.txt", max_chars=3000):
    # 1. 准备启动命令 (确保写死线程数 -t 16)
    cmd = ["./your_executable", "-m", "model.gguf", "-t", "16"]
    
    # 2. 以写入模式打开文件，并将 stdout/stderr 全都指向它
    with open(output_file, "w", encoding="utf-8") as f_out:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,  # 只有输入留着管道，用于发命令
            stdout=f_out,           # 输出直接去文件，不经过 Python 管道
            stderr=f_out,           # 错误日志也去文件
            text=True,
            bufsize=1
        )

    print(f"--- 程序已启动，输出实时重定向至 {output_file} ---")

    # 3. 阶段一：等待加载完毕 (由于加载很快，这里每 2 秒看一次)
    command_sent = False
    while not command_sent:
        time.sleep(2)
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f_check:
                content = f_check.read()
                # 检查是否出现了提示符
                if "Input:" in content or ">" in content:
                    print("[系统] 检测到就绪，发送指令...")
                    process.stdin.write("请详细介绍量子力学\n")
                    process.stdin.flush()
                    command_sent = True
        
        if process.poll() is not None:
            print("错误：程序在加载阶段意外退出。")
            return

    # 4. 阶段二：每分钟监听一次结果
    print("--- 指令已发送，进入分钟级监控模式 ---")
    while True:
        time.sleep(60) # 每一分钟监听一次
        
        if not os.path.exists(output_file):
            continue
            
        with open(output_file, "r", encoding="utf-8") as f_check:
            # 移动到文件末尾检查最后一部分内容
            content = f_check.read()
            char_count = len(content)
            
            print(f"[{time.strftime('%H:%M:%S')}] 当前字符数: {char_count}")

            # 条件判断：是否输出完毕
            # 检查最后 100 个字符里是否有 User:
            last_segment = content[-100:] 
            if "User:" in last_segment:
                print("[系统] 检测到 'User:' 标识，任务完成。")
                break
            
            # 条件判断：是否超过 3000 字
            if char_count > max_chars:
                print(f"[系统] 字数已达 {char_count}，超过上限 {max_chars}，强制终止。")
                break

        # 检查进程是否还在跑
        if process.poll() is not None:
            print("[系统] 进程已自行结束。")
            break

    # 5. 任务结束，清理
    process.terminate()
    process.wait()
    print(f"--- 运行全过程已记录在 {output_file} ---")

if __name__ == "__main__":
    run_with_file_monitor()
