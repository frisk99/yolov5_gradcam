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
import os
from typing import TypedDict, Annotated
from pydantic import Field
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# ==========================================
# 0. 环境准备
# ==========================================
# 请确保设置了你的 OpenAI API Key。
# 如果你使用的是其他兼容 OpenAI 接口的模型（如 DeepSeek, Qwen），请修改 base_url 和模型名称。
os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here" 

# ==========================================
# 1. 定义动作工具 (Tools)
# ==========================================
@tool
def call_phone(target_role: str = Field(description="The role or person to call, e.g., 'admin', 'police'"), 
               reason: str = Field(description="The reason for calling")):
    """Use this tool to make a phone call to a specific role."""
    print(f"\n📞 [Tool Executed] Calling {target_role}... Reason: {reason}")
    return f"Successfully called {target_role}"

@tool
def send_sms(target_role: str = Field(description="The role or person to text, e.g., 'car owner'"), 
             message: str = Field(description="The text message content")):
    """Use this tool to send an SMS text message."""
    print(f"\n✉️ [Tool Executed] Sending SMS to {target_role}... Message: {message}")
    return f"SMS sent to {target_role}"

@tool
def trigger_alarm(floor: int = Field(description="The floor number to trigger the alarm on"), 
                  alarm_type: str = Field(description="Type of alarm, e.g., 'fire', 'intruder'")):
    """Use this tool to sound the physical building alarm."""
    print(f"\n🚨 [Tool Executed] Triggering {alarm_type} alarm on floor {floor}!")
    return f"Alarm triggered on floor {floor}"

# 将所有可用工具打包
action_tools = [call_phone, send_sms, trigger_alarm]

# ==========================================
# 2. 定义图状态 (State)
# ==========================================
class InspectionState(TypedDict):
    floor: int
    condition: str             # 触发条件 (例如: "detect human presence")
    action_rule: str           # 触发后的动作规则 (例如: "call the floor administrator")
    camera_data: str           # 模拟的摄像头文字描述（真实场景中可以是图片的 Base64）
    should_alert: bool         # 内部状态：是否需要报警
    messages: Annotated[list, add_messages] # 用于存储 Agent 和 Tool 之间的对话历史

# ==========================================
# 3. 定义图的节点 (Nodes)
# ==========================================
def analyze_node(state: InspectionState):
    """
    节点 1：负责“看”。根据传入的 condition 判断当前画面是否异常。
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # 纯英文 Prompt，保证逻辑严密性
    sys_prompt = f"""You are an advanced security visual analysis agent.
Your task is to observe the provided surveillance data and determine if it triggers a specific condition.

[Trigger Condition]: {state['condition']}

Rules:
1. Carefully compare the visual data against the [Trigger Condition].
2. If the visual data STRICTLY MATCHES the condition, reply with exactly "YES".
3. If it does not match, or the scene is normal, reply with exactly "NO".
4. Do not output any additional explanation, formatting, or punctuation."""

    messages = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=f"Surveillance Data: {state['camera_data']}")
    ]
    
    response = llm.invoke(messages).content.strip().upper()
    should_alert = "YES" in response
    
    print(f"👀 [Analyze Node] Floor: {state['floor']} | Condition: '{state['condition']}'")
    print(f"   -> Model Judgment: {response} | Trigger Alert: {should_alert}")
    
    return {"should_alert": should_alert}


def action_agent_node(state: InspectionState):
    """
    节点 2：负责“决策”。当发现异常时，根据 action_rule 选择合适的工具并生成调用参数。
    """
    llm_with_tools = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(action_tools)
    
    sys_prompt = f"""You are a security action execution agent.
An anomaly has just been confirmed on floor {state['floor']}. 
The detected situation matches the condition: {state['condition']}.

[Required Action Rule]: {state['action_rule']}

Your task:
Based on the [Required Action Rule], select the most appropriate tool to execute the action.
Extract the necessary parameters (like target role, reason, or message) from the context.
Execute the tool immediately. Do not ask for user confirmation."""

    messages = [SystemMessage(content=sys_prompt)]
    
    print(f"🧠 [Action Agent] Deciding action based on rule: '{state['action_rule']}'...")
    response = llm_with_tools.invoke(messages)
    
    return {"messages": [response]}


def alert_router(state: InspectionState):
    """
    条件路由：根据 analyze_node 的结果决定去向。
    """
    if state.get("should_alert"):
        return "action_agent"
    return END

# ==========================================
# 4. 组装 LangGraph 工作流
# ==========================================
workflow = StateGraph(InspectionState)

# 添加节点
workflow.add_node("analyze", analyze_node)
workflow.add_node("action_agent", action_agent_node)
# ToolNode 是 LangGraph 内置的专门用于执行大模型输出的工具调用的节点
workflow.add_node("execute_tools", ToolNode(action_tools)) 

# 设置起点
workflow.set_entry_point("analyze")

# 设置边与路由
# 1. 分析完后，判断是否需要动作
workflow.add_conditional_edges(
    "analyze", 
    alert_router, 
    {"action_agent": "action_agent", END: END}
)

# 2. 动作决策完后，执行工具 (tools_condition 会检查 LLM 是否真的请求了工具)
workflow.add_conditional_edges(
    "action_agent", 
    tools_condition, 
    {"tools": "execute_tools", END: END}
)

# 3. 工具执行完毕后，流程结束
workflow.add_edge("execute_tools", END)

# 编译图
app = workflow.compile()

# ==========================================
# 5. 测试用例运行
# ==========================================
if __name__ == "__main__":
    print("==================================================")
    print("TEST CASE 1: 发现可疑人员，触发拨打电话 (Should Call)")
    print("==================================================")
    state_1 = {
        "floor": 6,
        "condition": "detect human presence",
        "action_rule": "call the floor administrator",
        "camera_data": "A person in a black hoodie is walking down the hallway.",
        "messages": []
    }
    app.invoke(state_1)
    
    print("\n==================================================")
    print("TEST CASE 2: 发现违停，触发发送短信 (Should SMS)")
    print("==================================================")
    state_2 = {
        "floor": -1,
        "condition": "detect a car parked outside of the designated parking lines",
        "action_rule": "send an SMS to the car owner asking them to move the car",
        "camera_data": "A red sedan is parked blocking the fire exit.",
        "messages": []
    }
    app.invoke(state_2)

    print("\n==================================================")
    print("TEST CASE 3: 画面正常，不触发任何动作 (Should Ignore)")
    print("==================================================")
    state_3 = {
        "floor": 3,
        "condition": "detect fire or smoke",
        "action_rule": "trigger the fire alarm",
        "camera_data": "The server room is dark and quiet, all indicator lights are normal.",
        "messages": []
    }
    app.invoke(state_3)
