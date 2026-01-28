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
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

# 1. 配置模型连接到本地 vLLM
# vLLM 默认地址通常是 http://localhost:8000/v1
# api_key 随便填一个字符串即可，本地服务通常不校验
model = OpenAIModel(
    'my-local-model',  # 必须与 vLLM 启动参数 --served-model-name 一致
    base_url='http://localhost:8000/v1',
    api_key='EMPTY'
)

# 2. 定义 Agent
# 这里我们定义一个简单的数学助手 Agent
agent = Agent(
    model,
    system_prompt='你是一个乐于助人的数学助手。请用简洁的方式回答问题。',
)

# 3. 运行 Agent (异步方式)
async def main():
    try:
        # 发送简单的文本 Prompt
        prompt = "如果我有3个苹果，吃掉了1个，还剩几个？请用JSON格式回答：{\"remaining\": int}"
        
        print(f"User: {prompt}")
        print("-" * 20)
        
        # 获取响应
        result = await agent.run(prompt)
        
        print(f"AI: {result.data}")
        
        # 打印使用 Token 统计 (vLLM 会返回这些信息)
        print("-" * 20)
        print(f"Usage: {result.usage()}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    asyncio.run(main())
