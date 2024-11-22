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




```cpp
#include <iostream>
#include <sstream>
#include <vector>
#include <string>

// 自定义类型
enum abnormal_type {
    TYPE_A,
    TYPE_B,
    TYPE_C
};

// 定义结构体
struct Rect {
    abnormal_type type;
    int x1, x2, y1, y2;

    // 打印函数，用于调试
    void print() const {
        std::cout << "Type: " << type 
                  << ", x1: " << x1 << ", x2: " << x2 
                  << ", y1: " << y1 << ", y2: " << y2 << std::endl;
    }
};

// 将结构体数组序列化为字符串
std::string structArrayToString(const std::vector<Rect>& rects) {
    std::ostringstream oss;
    for (const auto& rect : rects) {
        oss << rect.type << "," << rect.x1 << "," << rect.x2 << "," 
            << rect.y1 << "," << rect.y2 << ";";
    }
    return oss.str();
}

// 将字符串反序列化为结构体数组
std::vector<Rect> stringToStructArray(const std::string& data) {
    std::vector<Rect> rects;
    std::istringstream iss(data);
    std::string token;

    while (std::getline(iss, token, ';')) { // 按结构体分隔符解析
        if (!token.empty()) {
            std::istringstream fieldStream(token);
            int type, x1, x2, y1, y2;

            char delim; // 用于跳过逗号
            if (fieldStream >> type >> delim >> x1 >> delim >> x2 
                           >> delim >> y1 >> delim >> y2) {
                rects.push_back({static_cast<abnormal_type>(type), x1, x2, y1, y2});
            }
        }
    }

    return rects;
}

int main() {
    // 初始化结构体数组
    std::vector<Rect> rects = {
        {TYPE_A, 10, 20, 30, 40},
        {TYPE_B, 15, 25, 35, 45},
        {TYPE_C, 5, 10, 15, 20}
    };

    // 结构体数组转字符串
    std::string serialized = structArrayToString(rects);
    std::cout << "Serialized: " << serialized << std::endl;

    // 字符串转回结构体数组
    std::vector<Rect> deserialized = stringToStructArray(serialized);

    // 打印解析结果
    std::cout << "Deserialized:" << std::endl;
    for (const auto& rect : deserialized) {
        rect.print();
    }

    return 0;
}