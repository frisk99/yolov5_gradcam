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
page_index=0
def next_page_func():
    global page_index
    page_index = (page_index+1) %3
    # print('done!')
    # print(page_index)
def pre_page_func():
    global page_index
    page_index = (page_index-1) %3
    # print('done!')
    # print(page_index)
def greet(name):
    global page_index
    idx = str(page_index)
    return "Hello " + name+" "+idx + "!"

with gr.Blocks() as demo:
    if page_index == 0:
        print('catch 0')
    elif page_index ==1:
        print('catch 1')
    else:
        print('catch 2')
    with gr.Row():
        with gr.Column(scale=1):
            prev_button = gr.Button("⬅️ 上一页")
        with gr.Column(scale=8):
            name = gr.Textbox(label="Name")
            output = gr.Textbox(label="Output Box")
            greet_btn = gr.Button("Greet")
            greet_btn.click(fn=greet, inputs=name, outputs=output, api_name="greet")

        with gr.Column(scale=1):
            next_button = gr.Button("下一页 ➡️")

    prev_button.click(fn=pre_page_func)
    next_button.click(fn=next_page_func)

demo.launch(server_port=8890)
