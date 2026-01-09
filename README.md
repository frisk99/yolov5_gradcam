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

def run_adb_interactive():
    # 1. 启动 adb shell 并运行你的程序
    # xxx 是你的可执行程序路径，例如 /data/local/tmp/xxx
    cmd = ["adb", "shell", "/data/local/tmp/xxx"]
    
    # stdout=PIPE: 捕获输出
    # stdin=PIPE: 允许写入输入
    # text=True: 以字符串模式处理（Python 3.7+），旧版本用 universal_newlines=True
    process = subprocess.Popen(
        cmd, 
        stdin=subprocess.PIPE, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True,
        bufsize=1 # 行缓冲
    )

    print("正在等待程序加载...")

    # 2. 读取输出直到发现特定的“加载完成”标志
    while True:
        line = process.stdout.readline()
        if not line:
            break
        
        print(f"[Device]: {line.strip()}")
        
        # 假设程序加载完后会打印 "Ready" 或 "Enter command:"
        if "Ready" in line or ">" in line: 
            print("--- 检测到加载完成，正在发送命令 ---")
            break

    # 3. 输入你的命令 (cmd)
    # 注意要加上 \n 换行符模拟回车
    your_cmd = "my_command_here\n"
    process.stdin.write(your_cmd)
    process.stdin.flush() # 强制刷新缓冲区，确保命令发送出去

    # 4. 获取后续的输出
    # 如果命令执行完程序就退出，可以用 process.communicate()
    # 如果程序继续运行，可以继续用 readline()
    for line in process.stdout:
        print(f"[Result]: {line.strip()}")
        # 这里可以根据特定的结束标志 break 循环
        if "Done" in line:
            break

    # 5. 清理并关闭
    process.terminate()

if __name__ == "__main__":
    run_adb_interactive()
