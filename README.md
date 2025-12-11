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
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept> // 用于抛出异常

/**
 * @brief 加载 Numpy .tofile() 生成的 Raw 二进制文件
 * * @param file_path  raw 文件的路径
 * @param n_vocab    词表大小 (Qwen通常是 152064)
 * @param n_dim      向量维度 (Qwen通常是 3584)
 * @return std::vector<float> 包含所有 Embedding 的一维数组
 */
std::vector<float> load_embeddings(const std::string& file_path, size_t n_vocab, size_t n_dim) {
    // 1. 计算预期的总浮点数个数和总字节数
    size_t total_floats = n_vocab * n_dim;
    size_t expected_bytes = total_floats * sizeof(float); // float = 4 bytes

    // 2. 打开文件 (二进制模式 + 定位到末尾以获取大小)
    std::ifstream file(file_path, std::ios::binary | std::ios::ate);

    // 检查文件是否存在
    if (!file.is_open()) {
        throw std::runtime_error("❌ 无法打开文件: " + file_path);
    }

    // 3. 校验文件大小 (安全检查)
    std::streamsize file_size = file.tellg();
    if (file_size != static_cast<std::streamsize>(expected_bytes)) {
        // 构造详细的错误信息
        std::string err_msg = "❌ 文件大小不匹配!\n";
        err_msg += "  预期: " + std::to_string(expected_bytes) + " 字节 (FP32)\n";
        err_msg += "  实际: " + std::to_string(file_size) + " 字节\n";
        err_msg += "  提示: 请检查 Python 端是否做了 .float() 转换，或维度是否正确。";
        throw std::runtime_error(err_msg);
    }

    // 4. 回到文件开头准备读取
    file.seekg(0, std::ios::beg);

    // 5. 分配内存 (一次性申请约 2GB，避免 resize 开销)
    std::vector<float> embeddings(total_floats);

    // 6. 执行读取 (DMA 拷贝)
    // 将 vector 内部指针强转为 char*，直接填充二进制数据
    if (!file.read(reinterpret_cast<char*>(embeddings.data()), expected_bytes)) {
        throw std::runtime_error("❌ 读取文件流中断");
    }

    std::cout << "✅ 成功加载 Embedding: " << file_path << std::endl;
    std::cout << "   内存占用: " << expected_bytes / 1024 / 1024 << " MB" << std::endl;
    std::cout << "   矩阵形状: [" << n_vocab << " x " << n_dim << "]" << std::endl;

    return embeddings;
}
