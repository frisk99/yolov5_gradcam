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
from utils.ui_utils import get_points, undo_points
from utils.ui_utils import clear_all, store_img, train_lora_interface, run_drag
import os
LENGTH = 650

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
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

with gr.Blocks(css="""
.button {height: 480px !important; width: 50px !important; padding: 0; margin: 0; display: flex; align-items: center; justify-content: center;}
.button1 {height: 50px !important; width: 0px !important; padding: 0; margin: 0; display: flex; align-items: center; justify-content: center;}
.button-column {padding: 0 !important; margin: 0 !important; display: flex; align-items: center; justify-content: center;}
.gradio-image {height: 300px; width: 2000px;}
.gradio-image1 {height: 300px; width: 300px;}
""") as demo:
    page_index = gr.State(value=0)

    with gr.Row():
        with gr.Column(scale=1, elem_classes="button-column"):
            prev_button = gr.Button("⬅️", elem_classes="button")
        with gr.Column(scale=8):
            # 页面内容容器
            page0 = gr.Column(visible=True)
            page1 = gr.Column(visible=False)
            page2 = gr.Column(visible=False)
            with page0:
                mask = gr.State(value=None)  # store mask
                selected_points = gr.State([])  # store points
                original_image = gr.State(value=None)  # store original input image
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("""<p style="text-align: center; font-size: 20px">Draw Mask</p>""")
                        canvas = gr.Image(type="numpy", tool="sketch", label="Draw Mask",
                                          show_label=True, height=LENGTH, width=LENGTH)  # for mask painting
                        train_lora_button = gr.Button("Train LoRA")
                    with gr.Column():
                        gr.Markdown("""<p style="text-align: center; font-size: 20px">Click Points</p>""")
                        input_image = gr.Image(type="numpy", label="Click Points",
                                               show_label=True, height=LENGTH, width=LENGTH, interactive=False)  # for points clicking
                        undo_button = gr.Button("Undo point")

            with page1:
                with gr.Row():
                    output_image = gr.Image(type="numpy", label="Editing Results",
                                                show_label=True, height=LENGTH, width=LENGTH, interactive=True)
                        #run_button = gr.Button("Run", interactive=True)
                    # with gr.Row():
                    with gr.Tab("Drag Config",scale=1):
                        with gr.Row():
                            n_pix_step = gr.Number(
                                value=20,
                                label="number of pixel steps",
                                info="Number of gradient descent (motion supervision) steps on latent.",
                                precision=0)
                            lam = gr.Number(value=0.1, label="lam", info="regularization strength on unmasked areas")
                            # n_actual_inference_step = gr.Number(value=40, label="optimize latent step", precision=0)
                            inversion_strength = gr.Slider(0, 1.0,
                                value=0.7,
                                label="inversion strength",
                                info="The latent at [inversion-strength * total-sampling-steps] is optimized for dragging.")
                            latent_lr = gr.Number(value=0.01, label="latent lr")
                            start_step = gr.Number(value=0, label="start_step", precision=0, visible=False)
                            start_layer = gr.Number(value=10, label="start_layer", precision=0, visible=False)
                    with gr.Tab("Base Model Config"):
                                    #             "gsdf/Counterfeit-V2.5",
                                    # "stablediffusionapi/anything-v5",
                                    # "SG161222/Realistic_Vision_V2.0",
                        with gr.Row():
                            local_models_dir = r'G:\huggingface'
                            local_models_choice = \
                                [os.path.join(local_models_dir,d) for d in os.listdir(local_models_dir) if os.path.isdir(os.path.join(local_models_dir,d))]
                            model_path = gr.Dropdown(value="runwayml/stable-diffusion-v1-5",
                                label="Diffusion Model Path",
                                choices=[
                                    "runwayml/stable-diffusion-v1-5",

                                ] + local_models_choice
                            )
                            vae_path = gr.Dropdown(value="default",
                                label="VAE choice",
                                choices=["default",
                                "stabilityai/sd-vae-ft-mse"] + local_models_choice
                            )

                    with gr.Tab("LoRA Parameters"):
                        with gr.Row():
                            lora_step = gr.Number(value=80, label="LoRA training steps", precision=0)
                            lora_lr = gr.Number(value=0.0005, label="LoRA learning rate")
                            lora_batch_size = gr.Number(value=4, label="LoRA batch size", precision=0)
                            lora_rank = gr.Number(value=16, label="LoRA rank", precision=0)
                            #clear_all_button = gr.Button("Clear All", interactive=True)
                    with gr.Column():
                        prompt = gr.Textbox(label="Prompt", interactive=True)
                        lora_path = gr.Textbox(value="./lora_tmp", label="LoRA path", interactive=True)
                        lora_status_bar = gr.Textbox(label="display LoRA training status", interactive=True)
                        run_button = gr.Button("Run", interactive=True)
                        clear_all_button = gr.Button("Clear All", interactive=True)
                        # algorithm specific parameters
                    #gr.Markdown("""<p style="text-align: center; font-size: 20px">Editing Results</p>""")
                    #gr.Markdown("""<p style="text-align: center; font-size: 20px">Editing Results</p>""")
            with page2:
                gr.Textbox(label="页面 2 内容")
        with gr.Column(scale=1, elem_classes="button-column"):
            next_button = gr.Button("➡️", elem_classes="button")
        
        # Store image and points
        canvas.edit(store_img, [canvas], [original_image, selected_points, input_image, mask])
        input_image.select(
            get_points,
            [input_image, selected_points],
            [input_image],
        )
        undo_button.click(
            undo_points,
            [original_image, mask],
            [input_image, selected_points]
        )
        train_lora_button.click(
            train_lora_interface,
            [original_image,
            prompt,
            model_path,
            vae_path,
            lora_path,
            lora_step,
            lora_lr,
            lora_batch_size,
            lora_rank],
            [lora_status_bar]
        )
        run_button.click(
            run_drag,
            [original_image,
            input_image,
            mask,
            prompt,
            selected_points,
            inversion_strength,
            lam,
            latent_lr,
            n_pix_step,
            model_path,
            vae_path,
            lora_path,
            start_step,
            start_layer,
            ],
            [output_image]
        )
        clear_all_button.click(
            clear_all,
            [gr.Number(value=LENGTH, visible=False, precision=0)],
            [canvas,
            input_image,
            output_image,
            selected_points,
            original_image,
            mask]
        )
        input_image.select(get_points, [input_image, selected_points], [input_image])
        undo_button.click(undo_points, [original_image, mask], [input_image, selected_points])
        
        # Page navigation
        prev_button.click(fn=pre_page_func, inputs=page_index, outputs=[page_index, page0, page1, page2])
        next_button.click(fn=next_page_func, inputs=page_index, outputs=[page_index, page0, page1, page2])

demo.queue().launch(server_port=8890)
