# YOLO-V5 GRADCAM

I constantly desired to know to which part of an object the object-detection models pay more attention. So I searched for it, but I didn't find any for Yolov5.
Here is my implementation of Grad-cam for YOLO-v5. To load the model I used the yolov5's main codes, and for computing GradCam I used the codes from the gradcam_plus_plus-pytorch repository.
Please follow my GitHub account and star ⭐ the project if this functionality benefits your research or projects.

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
```python
import gradio as gr

def switch_page(page_index):
    if page_index == 0:
        return "上传图片 1", gr.Image(label="图片 1", source="upload"), False
    elif page_index == 1:
        return "上传图片 2", gr.Image(label="图片 2", source="upload"), False
    elif page_index == 2:
        return "输入文本", gr.Textbox(label="输入文本"), True

with gr.Blocks() as demo:
    page_index = gr.State(value=0)
    
    with gr.Row():
        with gr.Column(scale=1):
            prev_button = gr.Button("⬅️ 上一页")
        with gr.Column(scale=8):
            title = gr.Textbox(label="页面标题", interactive=False)
            content_placeholder = gr.Column()
            run_button = gr.Button("运行", visible=False)
        with gr.Column(scale=1):
            next_button = gr.Button("下一页 ➡️")
    
    def prev_click(idx):
        idx = (idx - 1) % 3
        title_text, content_component, run_visible = switch_page(idx)
        return idx, title_text, content_component, run_button.update(visible=run_visible)

    def next_click(idx):
        idx = (idx + 1) % 3
        title_text, content_component, run_visible = switch_page(idx)
        return idx, title_text, content_component, run_button.update(visible=run_visible)

    prev_button.click(fn=prev_click, inputs=page_index, outputs=[page_index, title, content_placeholder, run_button])
    next_button.click(fn=next_click, inputs=page_index, outputs=[page_index, title, content_placeholder, run_button])
You have unused kwarg parameters in Column, please remove them: {'label': '图片 1', 'source': 'upload'}
demo.launch(server_port=8890)
