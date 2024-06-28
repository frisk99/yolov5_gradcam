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

def next_page_func(page_index):
    page_index = (page_index + 1) % 3
    return page_index, *update_page(page_index)

def pre_page_func(page_index):
    page_index = (page_index - 1) % 3
    return page_index, *update_page(page_index)

def update_page(page_index):
    if page_index == 0:
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
    elif page_index == 1:
        return gr.update(visible(False), gr.update(visible=True), gr.update(visible(False)
    else:
        return gr.update(visible(False), gr.update(visible(False), gr.update(visible(True)

def greet(name, page_index):
    return f"你好 {name}，你在页面 {page_index}!"

with gr.Blocks(css=".button {height: 300px;} .gradio-image {height: 300px;}") as demo:
    page_index = gr.State(value=0)

    with gr.Row():
        with gr.Column(scale=1):
            prev_button = gr.Button("⬅️ 上一页", elem_classes="button")
            next_button = gr.Button("下一页 ➡️", elem_classes="button")
        with gr.Column(scale=8):
            name = gr.Textbox(label="姓名")
            output = gr.Textbox(label="输出框")
            greet_btn = gr.Button("问候")
            greet_btn.click(fn=greet, inputs=[name, page_index], outputs=output)

    # 页面内容容器
    page0 = gr.Column(visible=True)
    with page0:
        gr.Image(label="上传图片到页面 0", elem_classes="gradio-image")
        
    page1 = gr.Column(visible=False)
    with page1:
        gr.Image(label="上传图片到页面 1", elem_classes="gradio-image")
        
    page2 = gr.Column(visible=False)
    with page2:
        gr.Textbox(label="页面 2 内容")

    prev_button.click(fn=pre_page_func, inputs=page_index, outputs=[page_index, page0, page1, page2])
    next_button.click(fn=next_page_func, inputs=page_index, outputs=[page_index, page0, page1, page2])

demo.launch(server_port=8890)