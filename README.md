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
import cv2
import numpy as np
from PIL import Image

def video_to_9_grid(video_path, output_path="grid_output.jpg", grid_size=(3, 3)):
    # 1. 读取视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 2. 计算等间距的 9 帧索引 (均匀采样)
    # 取从第 0 帧到最后一帧之间的 9 个点
    frame_indices = np.linspace(0, total_frames - 1, grid_size[0] * grid_size[1], dtype=int)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # 转换为 RGB (OpenCV 默认是 BGR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            # 如果读取失败，补一张黑图
            h, w, _ = frames[0].shape if frames else (224, 224, 3)
            frames.append(np.zeros((h, w, 3), dtype=np.uint8))

    cap.release()

    # 3. 统一图片大小 (可选，建议统一以保证网格整齐)
    # 假设每张子图缩放到 512x512
    target_size = (512, 512)
    resized_frames = [cv2.resize(f, target_size) for f in frames]

    # 4. 拼接成九宫格
    # 将 9 张图分成 3 行，每行 3 张
    rows = []
    for i in range(0, 9, 3):
        row = np.hstack(resized_frames[i:i+3]) # 水平拼接
        rows.append(row)
    
    grid_image = np.vstack(rows) # 垂直拼接

    # 5. 保存结果
    final_img = Image.fromarray(grid_image)
    final_img.save(output_path)
    print(f"九宫格图片已保存至: {output_path}")

# 使用示例
video_to_9_grid("your_video.mp4")
