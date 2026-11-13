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
https://drive.google.com/file/d/1PpIjAHH3NfyZX88tQVgLnppSv3JaVGgb/view?usp=drivesdk

flowchart TD
    subgraph S1["第一阶段：从 CAD 得到导航地图"]
        A["CAD / BIM 建筑模型"] --> B["导入或转换为 USD"]
        B --> C["校验比例、坐标和碰撞几何"]
        U["你已有的大楼 USD"] --> C
        C --> D["按楼层设置高度范围和栅格分辨率"]
        D --> E["Occupancy Map Generator"]
        E --> F["导出 PNG + YAML 静态占据地图"]
    end

    subgraph S2["第二阶段：在仿真中跑通导航"]
        F --> G["Map Server 加载并发布地图"]
        R["机器人模型、模拟传感器和底盘控制"] --> H["AMCL 定位 + Nav2 导航"]
        G --> H
        H --> I["验证定位、路径规划、避障和运动"]
    end

    subgraph S3["第三阶段：接入真实机器人"]
        I --> J["接入真实雷达、里程计、TF 和底盘驱动"]
        J --> K["现场扫描与地图对齐，完成标定"]
        K --> L{"能否稳定定位？"}
        L -- "不能" --> M["检查地图差异、尺度、外参和里程计"]
        M --> K
        L -- "能" --> N["按实机尺寸和运动能力调整导航参数"]
        N --> O["实测窄门、转弯、动态障碍和定位恢复"]
        O --> P{"巡航验收通过？"}
        P -- "未通过" --> N
        P -- "通过" --> Q["实际巡航运行"]
    end

    subgraph S4["第四阶段：运行中的环境维护"]
        Q --> V["临时变化：实时更新障碍层"]
        Q --> W["长期变化：修改地图或进行 SLAM 续建"]
        W --> X["验证并发布新版地图"]
        X --> K
    end