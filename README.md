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
from langgraph.channels import LastValue
from langgraph.pregel import Pregel, NodeBuilder

# ===== 定义节点 =====

# Node1: 读取 input，写入 channel_a
node1 = (
    NodeBuilder()
    .subscribe_only("input")
    .do(lambda x: print(f"  [执行阶段] Node1 开始执行，读取 input: {x}") or x)
    .do(lambda x: print(f"  [执行阶段] Node1 计算中...") or x + " -> processed")
    .do(lambda x: print(f"  [执行阶段] Node1 完成，将写入 channel_a") or x)
    .write_to("channel_a")
)

# Node2: 读取 channel_a，写入 channel_b
node2 = (
    NodeBuilder()
    .subscribe_only("channel_a")
    .do(lambda x: print(f"  [执行阶段] Node2 开始执行，读取 channel_a: {x}") or x)
    .do(lambda x: print(f"  [执行阶段] Node2 计算中...") or x + " -> further processed")
    .do(lambda x: print(f"  [执行阶段] Node2 完成，将写入 channel_b") or x)
    .write_to("channel_b")
)

# Node3: 读取 channel_b，写入 output
node3 = (
    NodeBuilder()
    .subscribe_only("channel_b")
    .do(lambda x: print(f"  [执行阶段] Node3 开始执行，读取 channel_b: {x}") or x)
    .do(lambda x: print(f"  [执行阶段] Node3 计算中...") or x + " -> final")
    .do(lambda x: print(f"  [执行阶段] Node3 完成，将写入 output") or x)
    .write_to("output")
)

# ===== 创建 Pregel 应用 =====

app = Pregel(
    nodes={"node1": node1, "node2": node2, "node3": node3},
    channels={
        "input": LastValue(str),
        "channel_a": LastValue(str),
        "channel_b": LastValue(str),
        "output": LastValue(str),
    },
    input_channels=["input"],
    output_channels=["output"],
    debug=True,  # 启用调试模式
)

# ===== 运行应用并观察输出 =====

print("\n" + "="*60)
print("开始执行 Pregel 应用")
print("="*60)

result = app.invoke({"input": "Hello"})

print("\n" + "="*60)
print("执行完成")
print("="*60)
print(f"最终结果: {result}")
