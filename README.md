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
#include <opencv2/opencv.hpp>
#include <iostream>

std::vector<float> qwen2_vl_process_image(
    const cv::Mat& img, 
    int temporal_patch_size,
    int patch_size,
    int merge_size
) {
    if (img.empty() || img.channels() != 3) {
        return {};
    }

    int height = img.rows;
    int width = img.cols;
    int channel = 3;

    int grid_h = height / patch_size;
    int grid_w = width / patch_size;
    int grid_t = 1;

    int grid_h_sub = grid_h / merge_size;
    int grid_w_sub = grid_w / merge_size;

    // 内存分配：7 个数字相乘
    size_t total_elements = (size_t)grid_t * grid_h * grid_w * channel * temporal_patch_size * patch_size * patch_size;
    //两次merge size 可以忽略
    std::vector<float> result(total_elements, 0.0f);
    float* out_ptr = result.data();

    // 循环：8 层 (i3, i6, i4, i7, i2, i1, i5, i8) + 1 层隐式 (i0)
    for (int i3 = 0; i3 < grid_h_sub; ++i3) {
        for (int i6 = 0; i6 < grid_w_sub; ++i6) {
            for (int i4 = 0; i4 < merge_size; ++i4) {
                for (int i7 = 0; i7 < merge_size; ++i7) {
                    for (int i2 = 0; i2 < channel; ++i2) {
                        int channel_idx = i2; 
                        for (int i1 = 0; i1 < temporal_patch_size; ++i1) {
                            for (int i5 = 0; i5 < patch_size; ++i5) {
                                for (int i8 = 0; i8 < patch_size; ++i8) {
                                    
                                    int h_idx = i3 * (merge_size * patch_size) + i4 * patch_size + i5;
                                    int w_idx = i6 * (merge_size * patch_size) + i7 * patch_size + i8;

                                    if (h_idx < height && w_idx < width) {
                                        cv::Vec3b pixel = img.at<cv::Vec3b>(h_idx, w_idx);
                                        *out_ptr++ = static_cast<float>(pixel[channel_idx]);
                                    } else {
                                        *out_ptr++ = 0.0f;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    //repeat 
    result.reserve(result.size() * 2);
    result.insert(result.end(), result.begin(), result.end());
    return result;
}
std::string extractContent(const std::string& source) {
    const std::string startDelim = "BEGIN]";
    const std::string endDelim = "[END]";
    size_t startPos = source.find(startDelim);
    if (startPos == std::string::npos) return "";
    startPos += startDelim.length();
    size_t endPos = source.find(endDelim, startPos);
    if (endPos == std::string::npos) return "";
    return source.substr(startPos, endPos - startPos);
}
template <typename T>
bool saveVectorToRaw(const std::vector<T>& vec, const std::string& filename) {
    static_assert(std::is_trivially_copyable<T>::value, "Error: Type must be POD (Plain Old Data)");

    if (vec.empty()) {
        std::cerr << "Warning: Vector is empty, creating empty file." << std::endl;
    }

    std::ofstream outFile(filename, std::ios::binary | std::ios::trunc);

    if (!outFile.is_open()) {
        std::cerr << "Error: Failed to open file " << filename << " for writing." << std::endl;
        return false;
    }
    size_t totalBytes = vec.size() * sizeof(T);
    outFile.write(reinterpret_cast<const char*>(vec.data()), totalBytes);

    outFile.close();
    if (!outFile) {
        std::cerr << "Error: Write operation failed!" << std::endl;
        return false;
    }

    std::cout << "Saved " << totalBytes << " bytes to " << filename << std::endl;
    return true;
}