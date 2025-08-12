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
#include <Eigen/Dense>

// 将 pitch, yaw, roll 转换为旋转矩阵
Eigen::Matrix3f euler_to_rotation_matrix(float pitch, float yaw, float roll) {
    Eigen::AngleAxisf rollAngle(roll, Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf pitchAngle(pitch, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf yawAngle(yaw, Eigen::Vector3f::UnitZ());
    Eigen::Quaternion<float> q = yawAngle * pitchAngle * rollAngle;
    Eigen::Matrix3f R = q.matrix();
    return R;
}

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

// 你的 ToF 相机内参
const cv::Mat K_tof = (cv::Mat_<double>(3, 3) <<
    320, 0, 240,
    0, 320, 180,
    0, 0, 1);

// 你的 RGB 相机内参
const cv::Mat K_rgb = (cv::Mat_<double>(3, 3) <<
    800, 0, 640,
    0, 800, 360,
    0, 0, 1);

// ... (上面定义的 euler_to_rotation_matrix 函数) ...

cv::Mat alignDepthToRgbWithPyw(
    const cv::Mat& depth_map, 
    const Eigen::Vector3f& xyz_tof_to_rgb, 
    const Eigen::Vector3f& pyw_tof_to_rgb, 
    int rgb_width, 
    int rgb_height) {

    // 从 pyw 计算旋转矩阵 R
    Eigen::Matrix3f R = euler_to_rotation_matrix(pyw_tof_to_rgb[0], pyw_tof_to_rgb[1], pyw_tof_to_rgb[2]);
    Eigen::Vector3f T = xyz_tof_to_rgb;

    // 创建空的对齐后的深度图，与 RGB 图像尺寸相同
    cv::Mat aligned_depth = cv::Mat::zeros(rgb_height, rgb_width, CV_32FC1);

    // 获取 ToF 内参
    double fx_tof = K_tof.at<double>(0, 0);
    double fy_tof = K_tof.at<double>(1, 1);
    double cx_tof = K_tof.at<double>(0, 2);
    double cy_tof = K_tof.at<double>(1, 2);

    // 获取 RGB 内参
    double fx_rgb = K_rgb.at<double>(0, 0);
    double fy_rgb = K_rgb.at<double>(1, 1);
    double cx_rgb = K_rgb.at<double>(0, 2);
    double cy_rgb = K_rgb.at<double>(1, 2);
    
    // 遍历 ToF 深度图中的每个像素
    for (int v_tof = 0; v_tof < depth_map.rows; ++v_tof) {
        for (int u_tof = 0; u_tof < depth_map.cols; ++u_tof) {
            float Z_tof = depth_map.at<float>(v_tof, u_tof);

            if (Z_tof <= 0) continue;

            // 1. 反向投影到 ToF 坐标系下的 3D 点
            float X_tof = (u_tof - cx_tof) * Z_tof / fx_tof;
            float Y_tof = (v_tof - cy_tof) * Z_tof / fy_tof;
            Eigen::Vector3f p_tof(X_tof, Y_tof, Z_tof);

            // 2. 将 3D 点转换到 RGB 坐标系
            Eigen::Vector3f p_rgb = R * p_tof + T;

            // 3. 投影回 RGB 图像平面
            float X_rgb = p_rgb(0);
            float Y_rgb = p_rgb(1);
            float Z_rgb = p_rgb(2);

            if (Z_rgb <= 0) continue;

            int u_rgb = static_cast<int>((X_rgb * fx_rgb / Z_rgb) + cx_rgb);
            int v_rgb = static_cast<int>((Y_rgb * fy_rgb / Z_rgb) + cy_rgb);

            // 4. 检查边界并更新深度图
            if (u_rgb >= 0 && u_rgb < rgb_width && v_rgb >= 0 && v_rgb < rgb_height) {
                // ... 深度冲突处理逻辑 ...
                float current_depth = aligned_depth.at<float>(v_rgb, u_rgb);
                if (current_depth == 0 || Z_rgb < current_depth) {
                    aligned_depth.at<float>(v_rgb, u_rgb) = Z_rgb;
                }
            }
        }
    }
    return aligned_depth;
}
