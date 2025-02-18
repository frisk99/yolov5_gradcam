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

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>  // 使用 OpenCV 读取图像
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>

int main() {
    // 加载 TFLite 模型
    const std::string model_path = "your_model.tflite";
    std::ifstream model_file(model_path, std::ios::binary);
    std::string model_data((std::istreambuf_iterator<char>(model_file)),
                           std::istreambuf_iterator<char>());
    auto model = tflite::FlatBufferModel::BuildFromBuffer(model_data.c_str(), model_data.size());
    if (!model) {
        std::cerr << "加载模型失败！" << std::endl;
        return -1;
    }

    // 注册解释器
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder builder(*model, resolver);
    builder(&interpreter);
    if (!interpreter) {
        std::cerr << "创建解释器失败！" << std::endl;
        return -1;
    }

    // 读取并预处理图像（640x480 图像）
    const std::string image_path = "image.jpg";
    cv::Mat img = cv::imread(image_path);  // 使用 OpenCV 读取图像
    if (img.empty()) {
        std::cerr << "读取图像失败！" << std::endl;
        return -1;
    }

    // 调整图像大小为 640x480，并确保是 RGB
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(640, 480));  // 调整大小

    // 将图像数据转换为浮动类型，并归一化到 [0, 1] 范围
    resized_img.convertTo(resized_img, CV_32F, 1.0 / 255);

    // 将 OpenCV 图像数据填充到 TFLite 输入张量
    auto input_tensor = interpreter->tensor(interpreter->inputs()[0]);
    if (!input_tensor) {
        std::cerr << "获取输入张量失败！" << std::endl;
        return -1;
    }

    // 填充输入张量数据，假设张量形状是 [1, 640, 480, 3]
    int input_height = 640;
    int input_width = 480;
    int input_channels = 3;
    int input_size = input_height * input_width * input_channels;

    // 获取输入张量的数据指针
    float* input_data = input_tensor->data.f;

    // 遍历图像每个像素并填充张量
    for (int i = 0; i < input_height; ++i) {
        for (int j = 0; j < input_width; ++j) {
            for (int c = 0; c < input_channels; ++c) {
                // OpenCV 图片是 BGR 格式，TensorFlow Lite 通常需要 RGB 格式
                input_data[(i * input_width + j) * input_channels + c] =
                    resized_img.at<cv::Vec3f>(i, j)[c];  // RGB 值
            }
        }
    }

    // 运行推理
    if (interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "推理执行失败！" << std::endl;
        return -1;
    }

    // 获取输出张量
    auto output_tensor = interpreter->tensor(interpreter->outputs()[0]);
    if (!output_tensor) {
        std::cerr << "获取输出张量失败！" << std::endl;
        return -1;
    }

    // 打印输出结果
    std::cout << "推理结果：" << std::endl;
    for (int i = 0; i < output_tensor->dims->data[1]; ++i) {
        std::cout << output_tensor->data.f[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
auto* delegate = TfLiteGpuDelegateV2Create(/*default options=*/nullptr);
if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;

// Run inference