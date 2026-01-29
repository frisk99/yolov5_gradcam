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
import operator
from typing import Annotated, TypedDict, Union
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

# --- 定义工具 ---
@tool
def calculate_multiply(a: int, b: int) -> int:
    """计算乘法"""
    print(f"\n>>>>>> [DEBUG] 终于进来了！正在计算: {a} * {b} <<<<<<\n") # 你的 Print
    return a * b

tools = [calculate_multiply]

# --- 设置 LLM (请替换你的 vLLM 地址) ---
llm = ChatOpenAI(
    base_url="http://localhost:8000/v1", 
    api_key="EMPTY", 
    model="qwen-vllm", # 确保模型名正确
    temperature=0
).bind_tools(tools)

# --- 构建图 ---
class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_node("tools", ToolNode(tools)) # 关键：添加工具节点

builder.add_edge(START, "chatbot")
# 关键：条件路由
builder.add_conditional_edges("chatbot", tools_condition)
builder.add_edge("tools", "chatbot") # 关键：闭环

graph = builder.compile()

# --- 运行并观察 ---
print("开始测试...")
inputs = {"messages": [HumanMessage(content="计算 3 乘以 8")]}

for event in graph.stream(inputs):
    for node_name, value in event.items():
        print(f"--- 当前执行节点: {node_name} ---")
        if node_name == "chatbot":
            msg = value["messages"][-1]
            if not msg.tool_calls:
                print("❌ LLM 没有发起工具调用 (tool_calls 为空)")
            else:
                print(f"✅ LLM 发起了调用: {msg.tool_calls}")
        elif node_name == "tools":
            print("✅ 成功进入 Tools 节点 (你应该能在上方看到 DEBUG print)")
