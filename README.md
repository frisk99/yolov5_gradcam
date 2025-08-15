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
#include <iostream>
#include <vector>
#include <complex>
#include <opencv2/opencv.hpp>

/**
 * @brief 计算轮廓的傅里叶描述子
 * * @param contour 输入的轮廓点 (std::vector<cv::Point>)
 * @param num_descriptors 希望得到的描述子数量
 * @return std::vector<double> 归一化后的傅里叶描述子 (幅值)
 */
std::vector<double> calculateFourierDescriptors(const std::vector<cv::Point>& contour, int num_descriptors) {
    if (contour.empty()) {
        return {};
    }

    // 1. 将轮廓点转换为复数序列
    std::vector<std::complex<double>> complex_points;
    for (const auto& p : contour) {
        complex_points.emplace_back(static_cast<double>(p.x), static_cast<double>(p.y));
    }

    // 2. 使用OpenCV进行离散傅里叶变换 (DFT)
    cv::Mat dft_input(complex_points.size(), 1, CV_64FC2, complex_points.data());
    cv::Mat dft_output;
    cv::dft(dft_input, dft_output);

    // 3. 归一化傅里叶描述子
    std::vector<double> descriptors;
    
    // 获取第一个非直流分量 u_1 的幅值，用于缩放归一化
    // dft_output.at<cv::Vec2d>(i, 0)[0] 是实部, [1] 是虚部
    double mag_u1 = cv::sqrt(
        dft_output.at<cv::Vec2d>(1, 0)[0] * dft_output.at<cv::Vec2d>(1, 0)[0] +
        dft_output.at<cv::Vec2d>(1, 0)[1] * dft_output.at<cv::Vec2d>(1, 0)[1]
    );

    // 提取描述子，从 u_1 开始，实现平移不变性
    // 使用幅值，实现旋转不变性
    // 除以 |u_1|，实现缩放不变性
    for (int i = 1; i < std::min((int)dft_output.rows, num_descriptors + 1); ++i) {
        double real = dft_output.at<cv::Vec2d>(i, 0)[0];
        double imag = dft_output.at<cv::Vec2d>(i, 0)[1];
        double magnitude = cv::sqrt(real * real + imag * imag);
        
        if (mag_u1 > 1e-9) { // 避免除以零
            descriptors.push_back(magnitude / mag_u1);
        } else {
            descriptors.push_back(0.0);
        }
    }

    return descriptors;
}

/**
 * @brief (可选) 从傅里叶系数重建轮廓，用于验证
 * * @param dft_coeffs DFT变换后的完整系数
 * @param num_to_keep 保留的系数数量 (用于低通滤波重建)
 * @param original_size 原始轮廓的点数
 * @return std::vector<cv::Point> 重建的轮廓点
 */
std::vector<cv::Point> reconstructFromFourierDescriptors(const cv::Mat& dft_coeffs, int num_to_keep, int original_size) {
    cv::Mat truncated_coeffs = cv::Mat::zeros(dft_coeffs.size(), dft_coeffs.type());

    // 保留低频分量和对应的高频共轭分量
    for (int i = 0; i < num_to_keep; ++i) {
        truncated_coeffs.at<cv::Vec2d>(i, 0) = dft_coeffs.at<cv::Vec2d>(i, 0);
        // 保留共轭对称部分
        if (i > 0) {
           truncated_coeffs.at<cv::Vec2d>(original_size - i, 0) = dft_coeffs.at<cv::Vec2d>(original_size - i, 0);
        }
    }
    
    cv::Mat reconstructed_mat;
    // 进行逆傅里叶变换 (IDFT)
    cv::idft(truncated_coeffs, reconstructed_mat, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);

    std::vector<cv::Point> reconstructed_points;
    for (int i = 0; i < reconstructed_mat.rows; ++i) {
        // IDFT输出的是CV_64FC2, 但因为我们指定了DFT_REAL_OUTPUT，虚部应为0
        // 但为了安全，我们还是从CV_64FC2读取
        double x = reconstructed_mat.at<cv::Vec2d>(i, 0)[0];
        double y = reconstructed_mat.at<cv::Vec2d>(i, 0)[1];
        reconstructed_points.emplace_back(cv::saturate_cast<int>(x), cv::saturate_cast<int>(y));
    }
    
    // 注意：IDFT的结果需要手动转换为cv::Point。
    // 在OpenCV 4.5.2+，使用DFT_REAL_OUTPUT时，输出是一个单通道实数矩阵。
    // 为兼容旧版，这里仍按双通道读取。如果使用新版，可以简化为：
    // reconstructed_mat.convertTo(reconstructed_mat, CV_32S);
    // std::vector<cv::Point> reconstructed_points(reconstructed_mat.begin<cv::Point>(), reconstructed_mat.end<cv::Point>());


    return reconstructed_points;
}


int main() {
    // 1. 创建一个示例二值图像 (这里用一个手形轮廓)
    cv::Mat binary_image = cv::Mat::zeros(500, 500, CV_8UC1);
    std::vector<cv::Point> hand_contour = {
        {100, 300}, {120, 200}, {150, 180}, {180, 220}, {200, 210}, {220, 250},
        {250, 240}, {280, 280}, {300, 270}, {320, 320}, {300, 350}, {250, 360},
        {200, 380}, {150, 350}, {120, 320}
    };
    std::vector<std::vector<cv::Point>> contours_to_draw = {hand_contour};
    cv::drawContours(binary_image, contours_to_draw, 0, cv::Scalar(255), cv::FILLED);

    // 2. 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary_image, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    if (contours.empty()) {
        std::cerr << "Error: No contours found!" << std::endl;
        return -1;
    }

    // 假设我们处理最大的轮廓
    const auto& main_contour = contours[0];

    // 3. 计算傅里叶描述子
    int num_descriptors = 10; // 我们需要10个描述子
    std::vector<double> fds = calculateFourierDescriptors(main_contour, num_descriptors);

    std::cout << "Calculated " << num_descriptors << " Fourier Descriptors:" << std::endl;
    for (size_t i = 0; i < fds.size(); ++i) {
        std::cout << "FD " << i + 1 << ": " << fds[i] << std::endl;
    }
    
    // 4. (可选) 从傅里叶系数重建并可视化
    // 首先，我们需要未经归一化的完整DFT系数
    std::vector<std::complex<double>> complex_points;
    for (const auto& p : main_contour) {
        complex_points.emplace_back(static_cast<double>(p.x), static_cast<double>(p.y));
    }
    cv::Mat dft_input(complex_points.size(), 1, CV_64FC2, complex_points.data());
    cv::Mat full_dft_coeffs;
    cv::dft(dft_input, full_dft_coeffs);
    
    // 使用前8个描述子（系数）来重建
    int coeffs_to_reconstruct = 8;
    std::vector<cv::Point> reconstructed_contour = reconstructFromFourierDescriptors(full_dft_coeffs, coeffs_to_reconstruct, main_contour.size());
    
    // 5. 可视化结果
    cv::Mat result_image = cv::Mat::zeros(500, 500, CV_8UC3);
    
    // 绘制原始轮廓 (绿色)
    std::vector<std::vector<cv::Point>> original_contours_vec = {main_contour};
    cv::drawContours(result_image, original_contours_vec, 0, cv::Scalar(0, 255, 0), 2); // Green

    // 绘制重建轮廓 (红色)
    if (!reconstructed_contour.empty()) {
        std::vector<std::vector<cv::Point>> reconstructed_contours_vec = {reconstructed_contour};
        cv::drawContours(result_image, reconstructed_contours_vec, 0, cv::Scalar(0, 0, 255), 2); // Red
    }

    cv::imshow("Original (Green) vs Reconstructed (Red)", result_image);
    cv::waitKey(0);

    return 0;
}

    canvas_size = 400
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    
    # 将点集平移到画布中心，方便观察
    # min_x, min_y = np.min(leg_points, axis=0)
    # max_x, max_y = np.max(leg_points, axis=0)
    # center_x = (min_x + max_x) / 2
    # center_y = (min_y + max_y) / 2
    # leg_points_centered = leg_points - [center_x, center_y] + [canvas_size / 2, canvas_size / 2]
    
    # 绘制原始点
    for point in leg_points.astype(np.int32):
        cv2.circle(canvas, tuple(point + 150), 2, (255, 255, 255), -1)
    
    # 绘制凸包
    cv2.drawContours(canvas, [hull.astype(np.int32) + [150, 150]], -1, (0, 255, 0), 2)
    
    cv2.imshow("Convex Hull of Scattered Points", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- 1. 创建一个示例图像 ---
# 创建一个 500x500 的黑色背景图像
image = np.zeros((500, 500), dtype=np.uint8)

# 在图像上绘制一个白色的 "L" 形作为我们的目标轮廓
# 这是一个相对复杂的形状，有尖角，能更好地展示频谱
pts = np.array([[100, 100], [300, 100], [300, 150], [150, 150], [150, 300], [100, 300]], np.int32)
cv2.fillPoly(image, [pts], 255)

# --- 2. 查找轮廓 ---
# cv2.RETR_EXTERNAL: 只检测最外层的轮廓
# cv2.CHAIN_APPROX_NONE: 获取轮廓上所有的点，这对于精确的傅里葉分析很重要
contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# 检查是否找到了轮廓
if not contours:
    print("没有找到轮廓！")
    exit()

# 我们处理找到的第一个（也是最大的）轮廓
contour = contours[0]

# 创建一个彩色图像用于可视化轮廓
contour_visual = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(contour_visual, [contour], -1, (0, 255, 0), 3)

# --- 3. 将轮廓点转换为复数序列 ---
# contour 的形状是 (N, 1, 2)，N是点的数量。我们需要将其转换为 (N, 2)
contour_2d = contour.squeeze()

# 将 (x, y) 坐标转换为复数 x + iy
# 这是傅里叶描述子的标准预处理步骤
complex_contour = contour_2d[:, 0] + 1j * contour_2d[:, 1]

# --- 4. 执行快速傅里叶变换 (FFT) ---
fourier_coeffs = np.fft.fft(complex_contour)

# --- 5. 计算频谱 (傅里叶系数的幅度) ---
# 幅度表示了每个频率分量对形状贡献的大小
spectrum = np.abs(fourier_coeffs)

# --- 6. 可视化频谱图 ---
# 通常，第一个系数 (f_coeffs[0]) 是直流分量，代表轮廓的质心。
# 它的值通常非常大，会压缩其他分量的显示范围，所以我们通常在绘图时忽略它，
# 或者从第二个系数开始绘制，以更好地观察定义形状的交流分量。
# 我们这里只显示前50个频率分量，因为高频分量通常很快衰减到0。
num_descriptors_to_show = 50
plot_spectrum = spectrum[1:num_descriptors_to_show + 1]

# 设置绘图
plt.figure(figsize=(12, 6))

# 子图1: 显示原始图像和轮廓
plt.subplot(1, 2, 1)
plt.imshow(contour_visual)
plt.title('Original Contour')
plt.axis('off')

# 子图2: 显示频谱图
plt.subplot(1, 2, 2)
# 使用条形图更直观
x_axis = range(1, len(plot_spectrum) + 1)
plt.bar(x_axis, plot_spectrum, color='skyblue')
plt.title('Fourier Descriptor Spectrum (First 50 Descriptors)')
plt.xlabel('Frequency Index (Descriptor)')
plt.ylabel('Magnitude')
plt.grid(True, linestyle='--', alpha=0.6)

# 调整布局并显示
plt.tight_layout()
plt.show()

# 打印前10个频谱值
print("Spectrum Magnitudes (starting from the 2nd coefficient):")
for i, mag in enumerate(spectrum[1:11]):
    print(f"  Descriptor {i+1}: {mag:.2f}")

