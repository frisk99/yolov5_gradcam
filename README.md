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
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

def analyze_node(state: InspectionState):
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # 【纯英文 Prompt】：精准定义角色、任务、条件和输出格式
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
    
    print(f"[Analyze Node] Condition: '{state['condition']}' | Response: {response}")
    return {"should_alert": should_alert}
