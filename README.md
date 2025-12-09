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
#include "include/llama.h"  // 确保路径正确
#include <vector>
#include <string>
#include <iostream>
#include <cstring>

// 辅助函数：将 token id 转回字符串 (Detokenize)
// 注意：参数类型已修改为 const llama_vocab*
std::string token_to_piece(const llama_vocab* vocab, llama_token token) {
    std::vector<char> result(8, 0);
    // 新版 API：传入 vocab
    int n_tokens = llama_token_to_piece(vocab, token, result.data(), result.size(), 0, true);
    
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        // 新版 API：传入 vocab
        int check = llama_token_to_piece(vocab, token, result.data(), result.size(), 0, true);
        if (check == -n_tokens) return std::string(result.data(), result.size());
    } else {
        return std::string(result.data(), n_tokens);
    }
    return "";
}

int main() {
    // 1. 初始化
    llama_backend_init();
    // 关掉烦人的加载日志，让输出干净点
    llama_log_set(NULL, NULL);

    // 2. 加载模型 (Vocab Only 模式)
    auto mparams = llama_model_default_params();
    mparams.vocab_only = true; // 关键：只加载词表，不吃显存

    // 指向你的纯词表 GGUF 文件
    struct llama_model * model = llama_load_model_from_file("qwen_vocab.gguf", mparams);

    if (!model) {
        std::cerr << "加载失败: 找不到 qwen_vocab.gguf 文件" << std::endl;
        return 1;
    }

    // ==========================================
    // 【关键修改】获取 Vocab 指针
    // ==========================================
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // 3. 执行分词
    std::string text = "帮我点击<box>(100,200)</box>";
    
    // 预分配缓冲区
    std::vector<llama_token> tokens(text.length() + 16);

    // 【关键修改】这里传入 vocab 而不是 model
    int n_tokens = llama_tokenize(
        vocab,           // <--- 修改了这里
        text.c_str(), 
        text.length(), 
        tokens.data(), 
        tokens.size(), 
        true, // add_bos
        true  // parse_special (识别 <box> 等)
    );

    if (n_tokens < 0) {
        // 如果返回负数，说明 buffer 不够大，这里为了演示直接报错
        // 实际工程中可以根据 -n_tokens 重新 resize 并再次调用
        n_tokens = -n_tokens; 
        std::cerr << "Buffer too small" << std::endl;
    }
    tokens.resize(n_tokens);

    // 4. 打印 Token IDs
    std::cout << "Original Text: " << text << std::endl;
    std::cout << "Token IDs: [ ";
    for (auto id : tokens) {
        std::cout << id << " ";
    }
    std::cout << "]" << std::endl;

    // 5. 验证还原 (Detokenize)
    std::cout << "Decode Check: ";
    for (auto id : tokens) {
        // 【关键修改】辅助函数里也传入 vocab
        std::string piece = token_to_piece(vocab, id);
        std::cout << piece;
    }
    std::cout << std::endl;

    // 6. 清理
    llama_free_model(model);
    llama_backend_free();

    return 0;
}