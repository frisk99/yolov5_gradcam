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
import sys

def run_llm_auto_stop(output_file="llm_result.txt"):
    # 1. 启动子进程 (根据实际情况修改命令)
    # 示例: ["adb", "shell", "-t", "./llama-cli -m ..."] 或 ["python3", "demo.py"]
    cmd = ["python3", "your_llm_demo.py"] 

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=0  # 无缓冲，确保实时性
    )

    with open(output_file, "w", encoding="utf-8") as f:
        print(f"--- 任务启动，实时记录至 {output_file} ---")
        
        output_buffer = "" # 用于匹配关键词的缓冲区
        command_sent = False
        
        while True:
            # 2. 逐字符读取输出
            char = process.stdout.read(1)
            if not char:
                break
            
            # 实时显示和写入文件
            sys.stdout.write(char)
            sys.stdout.flush()
            f.write(char)
            f.flush()

            # 将字符加入缓冲区
            output_buffer += char
            
            # 3. 逻辑判断：等待加载完成发送命令
            if not command_sent:
                # 这里假设加载完出现的提示符是 "Input:" 或 ">"
                if "Input:" in output_buffer or ">" in output_buffer:
                    print("\n[系统] 检测到加载完成，发送指令...")
                    my_prompt = "请用50字介绍量子力学。\n"
                    process.stdin.write(my_prompt)
                    process.stdin.flush()
                    
                    f.write(f"\n[Sent Command]: {my_prompt}\n")
                    command_sent = True
                    output_buffer = "" # 清空缓冲区，开始监听结果

            # 4. 逻辑判断：检测到 "User:" 则结束
            else:
                # 如果缓冲区末尾出现了 "User:"，说明回答结束
                if output_buffer.strip().endswith("User:"):
                    print("\n\n[系统] 检测到 'User:' 标识，任务完成，正在退出...")
                    break
        
        # 5. 清理工作
        process.terminate() # 强制关闭子进程
        process.wait()
        print(f"--- 所有输出已保存至 {output_file} ---")

if __name__ == "__main__":
    run_llm_auto_stop()
