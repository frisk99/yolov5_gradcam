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
import asyncio
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.models.openai import OpenAIModel
import os

# 配置本地 vLLM
os.environ["OPENAI_BASE_URL"] = "http://localhost:8000/v1"
os.environ["OPENAI_API_KEY"] = "EMPTY"

# --- 1. 定义环境状态 (模拟真实的物理设备) ---
class DeviceState(BaseModel):
    temperature: float = 95.0  # 初始温度很高
    fan_speed: int = 0
    is_shutdown: bool = False
    
    # 模拟环境变化：每次读取温度时，根据设备状态改变温度
    def update_physics(self):
        if self.is_shutdown:
            self.temperature = 25.0 # 关机后冷却
        elif self.fan_speed > 0:
            self.temperature -= 2.0 # 风扇开启，温度微降（模拟降温不够快的情况）
        else:
            self.temperature += 1.0 # 没风扇，温度持续升高

# --- 2. 定义 Agent 和 依赖 ---
model = OpenAIModel('my-local-model')

# 定义 Agent，设置最大循环次数为 5 次，防止死循环
agent = Agent(
    model,
    deps_type=DeviceState,
    result_type=str, # 最终返回一个文本报告
    system_prompt=(
        "你是一名高级设备操作员。你的目标是将设备温度控制在 80°C 以下。\n"
        "你可以使用工具来检查温度和操作设备。\n"
        "规则：\n"
        "1. 先检查温度 (Perception)。\n"
        "2. 如果过热，尝试开启风扇 (Action)。\n"
        "3. 操作后，**必须**再次检查温度以确认效果 (Observation & Reflection)。\n"
        "4. 如果风扇无法有效降温且温度仍危急 (>90°C)，必须执行紧急关机！"
    )
)

# --- 3. 定义工具 (Agent 的手脚) ---

@agent.tool
def read_temperature(ctx: RunContext[DeviceState]) -> str:
    """读取当前设备的核心温度传感器。"""
    ctx.deps.update_physics() # 模拟时间流逝导致的环境变化
    temp = ctx.deps.temperature
    print(f"👁️ [感知] 读取温度: {temp}°C")
    return f"{temp}°C"

@agent.tool
def set_fan_speed(ctx: RunContext[DeviceState], speed_percent: int) -> str:
    """设置风扇转速 (0-100)。"""
    print(f"✋ [行动] 设置风扇转速: {speed_percent}%")
    ctx.deps.fan_speed = speed_percent
    return "风扇已设定，正在运行。"

@agent.tool
def emergency_shutdown(ctx: RunContext[DeviceState]) -> str:
    """执行紧急断电关机。仅在其他手段无效时使用。"""
    print(f"🛑 [行动] !!! 执行紧急关机 !!!")
    ctx.deps.is_shutdown = True
    return "设备已切断电源，正在强制冷却。"

# --- 4. 运行 Agentic Loop ---

async def main():
    # 初始化设备状态
    device = DeviceState()
    
    print(f"--- 任务开始: 监控并处理设备 (初始温度: {device.temperature}) ---")
    
    # 这一句 run() 包含了整个 思考->行动->观察->再思考 的循环
    result = await agent.run(
        "警报：核心模块温度异常，请处理。",
        deps=device
    )
    
    print("\n--- 任务结束 ---")
    print(f"AI 最终报告: {result.data}")
    print(f"设备最终状态: 温度={device.temperature}, 关机={device.is_shutdown}")

if __name__ == '__main__':
    asyncio.run(main())
