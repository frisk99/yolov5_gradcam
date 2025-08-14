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
import cv2
import numpy as np

def get_fourier_descriptors(contour, n_descriptors=20):
    """
    计算轮廓的傅里叶描述符。
    
    Args:
        contour (np.array): 轮廓点的数组。
        n_descriptors (int): 截取傅里叶描述符的数量。
        
    Returns:
        np.array: 归一化后的傅里叶描述符。
    """
    if contour is None or contour.shape[0] < 2:
        return None

    # 将轮廓坐标转换为复数序列
    x = contour[:, 0, 0] if contour.ndim == 3 else contour[:, 0]
    y = contour[:, 0, 1] if contour.ndim == 3 else contour[:, 1]
    complex_sequence = x + 1j * y
    
    # 进行傅里叶变换
    fourier_coeffs = np.fft.fft(complex_sequence)

    # 归一化以实现不变性
    if fourier_coeffs.size < n_descriptors + 2:
        return None
        
    fourier_coeffs_normalized = fourier_coeffs[1:n_descriptors+1]
    
    # 除以第二个系数的模，实现缩放和旋转不变性
    fd = np.abs(fourier_coeffs_normalized) / np.abs(fourier_coeffs[1])
    
    return fd

def get_descriptors_from_points(points):
    """
    从一组(x, y)坐标点计算傅里叶描述符。
    
    Args:
        points (np.array): Nx2的数组，表示(x, y)坐标。
        
    Returns:
        np.array: 归一化后的傅里叶描述符，如果失败则返回None。
    """
    if points is None or points.shape[0] < 3:
        print("点数量少于3个，无法计算凸包。")
        return None

    # 1. 计算最小凸包
    # 将 NumPy 数组转换为适合 OpenCV 的格式 (N, 1, 2)
    points_for_cv = points.astype(np.int32).reshape(-1, 1, 2)
    
    try:
        convex_hull = cv2.convexHull(points_for_cv)
        # 2. 计算傅里叶描述符
        fourier_descriptors = get_fourier_descriptors(convex_hull)
        return fourier_descriptors
    except cv2.error as e:
        print(f"计算凸包失败: {e}")
        return None

# --- 主程序示例 ---
if __name__ == '__main__':
    # 1. 模拟一组人腿形状的(x, y)坐标点
    # 这组点可以是TOF深度图的(x, y)投影，或任何其他方式获取的点集
    num_points = 500
    
    # 模拟“大腿”部分（不规则的椭圆点）
    t = np.linspace(0, 2 * np.pi, num_points // 2)
    x1 = 50 * np.cos(t) + np.random.normal(0, 2, num_points // 2)
    y1 = 100 * np.sin(t) + np.random.normal(0, 2, num_points // 2)
    
    # 模拟“小腿”部分（矩形点）
    x2 = np.linspace(-30, 30, num_points // 2) + 150
    y2 = np.random.uniform(50, -50, num_points // 2)
    
    leg_points = np.vstack((np.vstack((x1, y1)).T, np.vstack((x2, y2)).T))
    
    # 2. 从这些点集中直接计算傅里叶描述符
    fourier_descriptors = get_descriptors_from_points(leg_points)
    
    if fourier_descriptors is not None:
        print("计算出的人腿轮廓傅里叶描述符：")
        print(fourier_descriptors)

    # 3. 可视化
    hull = cv2.convexHull(leg_points.astype(np.int32).reshape(-1, 1, 2))
    
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
