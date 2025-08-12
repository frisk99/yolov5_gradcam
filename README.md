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




```python
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

def euler_to_rotation_matrix(pyw):
    """
    将 Pitch (y), Yaw (z), Roll (x) 欧拉角转换为旋转矩阵。
    """
    pitch, yaw, roll = pyw
    
    # 使用SciPy库进行转换，它更可靠且易于使用
    # 定义旋转顺序为 Z-Y-X (Yaw, Pitch, Roll)
    rotation = R_scipy.from_euler('zyx', [yaw, pitch, roll], degrees=False)
    
    return rotation.as_matrix()

def align_depth_to_rgb_with_pyw(depth_map, K_tof, K_rgb, xyz_tof_to_rgb, pyw_tof_to_rgb, rgb_image_shape):
    """
    使用ToF到RGB的外参（xyz, pyw）将ToF深度图对齐到RGB图像。

    Args:
        depth_map (np.ndarray): 原始ToF深度图 (H_tof x W_tof)。
        K_tof (np.ndarray): ToF相机的内参矩阵。
        K_rgb (np.ndarray): RGB相机的内参矩阵。
        xyz_tof_to_rgb (np.ndarray): ToF相机到RGB相机的平移向量 (3x1)。
        pyw_tof_to_rgb (np.ndarray): ToF相机到RGB相机的欧拉角 [pitch, yaw, roll] (1x3)。
        rgb_image_shape (tuple): RGB图像的尺寸 (高度, 宽度)。

    Returns:
        np.ndarray: 对齐后的深度图，与RGB图像尺寸相同。
    """
    h_tof, w_tof = depth_map.shape
    h_rgb, w_rgb = rgb_image_shape

    # 从 pyw 欧拉角计算旋转矩阵 R
    R = euler_to_rotation_matrix(pyw_tof_to_rgb)
    T = xyz_tof_to_rgb.reshape((3, 1))

    # 创建一个空的对齐后的深度图，填充为0
    aligned_depth = np.zeros((h_rgb, w_rgb), dtype=np.float32)

    # 获取ToF内参
    fx_tof, fy_tof = K_tof[0, 0], K_tof[1, 1]
    cx_tof, cy_tof = K_tof[0, 2], K_tof[1, 2]
    # 获取RGB内参
    fx_rgb, fy_rgb = K_rgb[0, 0], K_rgb[1, 1]
    cx_rgb, cy_rgb = K_rgb[0, 2], K_rgb[1, 2]
    
    # 遍历ToF深度图的每一个像素
    for v_tof in range(h_tof):
        for u_tof in range(w_tof):
            Z_tof = depth_map[v_tof, u_tof]

            # 忽略无效的深度值
            if Z_tof <= 0:
                continue

            # 1. 反向投影到ToF坐标系下的3D点
            X_tof = (u_tof - cx_tof) * Z_tof / fx_tof
            Y_tof = (v_tof - cy_tof) * Z_tof / fy_tof
            p_tof = np.array([X_tof, Y_tof, Z_tof]).reshape((3, 1))

            # 2. 将3D点转换到RGB坐标系
            p_rgb = np.dot(R, p_tof) + T

            # 3. 将3D点投影回RGB图像平面
            X_rgb, Y_rgb, Z_rgb = p_rgb.flatten()
            
            # 避免除以0
            if Z_rgb <= 0:
                continue

            u_rgb = int( (X_rgb * fx_rgb / Z_rgb) + cx_rgb )
            v_rgb = int( (Y_rgb * fy_rgb / Z_rgb) + cy_rgb )

            # 4. 检查像素是否在RGB图像范围内，并更新对齐后的深度图
            if 0 <= u_rgb < w_rgb and 0 <= v_rgb < h_rgb:
                # 只保留最近的深度值（解决多点映射到同一像素的问题）
                if aligned_depth[v_rgb, u_rgb] == 0 or Z_rgb < aligned_depth[v_rgb, u_rgb]:
                    aligned_depth[v_rgb, u_rgb] = Z_rgb

    return aligned_depth

# --------------------------------------------------------------------------
# --- 使用示例 ---
# 假设你已经有了ToF和RGB的内参
K_rgb = np.array([[800, 0, 640],
                  [0, 800, 360],
                  [0, 0, 1]], dtype=np.float32)
K_tof = np.array([[320, 0, 240],
                  [0, 320, 180],
                  [0, 0, 1]], dtype=np.float32)

# 你的ToF相机相对于RGB相机的xyz和平移向量
# 假设单位是米
xyz_tof_to_rgb = np.array([0.05, 0.0, 0.0]) # 例如，ToF相机在RGB相机右边5厘米
# 你的ToF相机相对于RGB相机的pyw欧拉角
# 假设单位是弧度 (rad)
pyw_tof_to_rgb = np.array([0.0, 0.0, 0.0]) # 假设没有旋转

# 模拟数据
depth_map_tof = np.zeros((360, 480), dtype=np.float32)
depth_map_tof[100:200, 100:300] = 1500 # 模拟一个深度为1.5米的物体

rgb_shape = (1080, 1920) # 假设RGB图像是1920x1080
aligned_depth_map = align_depth_to_rgb_with_pyw(depth_map_tof, K_tof, K_rgb, xyz_tof_to_rgb, pyw_tof_to_rgb, rgb_shape)

# 可视化结果
cv2.imwrite("aligned_depth_map_optimized.png", aligned_depth_map)
print("深度图对齐完成，结果已保存。")
