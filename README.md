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

import numpy as np
from scipy.spatial.transform import Rotation as R

# 假设已知两个相机的位姿
xyz_left = np.array([0, 0, 0])    # 左相机位置（世界坐标系原点）
ypr_left = np.array([0, 0, 0])    # 左相机欧拉角（yaw, pitch, roll，单位：度）

xyz_right = np.array([0.1, 0, 0]) # 右相机位置（X 方向偏移 10cm）
ypr_right = np.array([5, 0, 0])   # 右相机欧拉角（绕 Y 轴偏转 5°）

# (1) 将欧拉角转换为旋转矩阵
def ypr_to_rotation_matrix(yaw, pitch, roll):
    """ 欧拉角 (Z-Y-X 顺序) 转旋转矩阵 """
    yaw, pitch, roll = np.radians([yaw, pitch, roll])  # 转为弧度
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw),  np.cos(yaw), 0],
                   [0,            0,           1]])
    Ry = np.array([[np.cos(pitch),  0, np.sin(pitch)],
                   [0,             1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    Rx = np.array([[1, 0,            0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll),  np.cos(roll)]])
    return Rz @ Ry @ Rx  # 组合旋转

# 计算左右相机的旋转矩阵
R_left = ypr_to_rotation_matrix(*ypr_left)
R_right = ypr_to_rotation_matrix(*ypr_right)

# (2) 计算右相机相对于左相机的变换
T = xyz_right - xyz_left  # 平移向量
R = np.linalg.inv(R_left) @ R_right  # 相对旋转矩阵

print("相对旋转矩阵 R:\n", R)
print("相对平移向量 T:\n", T)

import cv2

# 假设已经标定了相机内参（K）和畸变系数（D）
K_left = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])  # 左相机内参
D_left = np.array([k1, k2, p1, p2, k3])                   # 左相机畸变
K_right = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])  # 右相机内参
D_right = np.array([k1, k2, p1, p2, k3])                  # 右相机畸变

# 立体校正
image_size = (640, 480)  # 图像尺寸
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    K_left, D_left, K_right, D_right, 
    image_size, R, T,  # 使用计算得到的 R 和 T
    flags=cv2.CALIB_ZERO_DISPARITY,
    alpha=0  # 裁剪无效区域
)

# 计算校正映射
left_map1, left_map2 = cv2.initUndistortRectifyMap(K_left, D_left, R1, P1, image_size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(K_right, D_right, R2, P2, image_size, cv2.CV_16SC2)

# 校正图像（假设已经读取左右图像）
left_rectified = cv2.remap(left_img, left_map1, left_map2, cv2.INTER_LINEAR)
right_rectified = cv2.remap(right_img, right_map1, right_map2, cv2.INTER_LINEAR)


# 转换为灰度图
left_gray = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)

# 计算视差图（SGBM 算法）
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=64,  # 视差范围
    blockSize=5,
    P1=8 * 3 * 5**2,
    P2=32 * 3 * 5**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)
disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

# 转换为深度图
baseline = np.linalg.norm(T)  # 基线距离（单位：米）
focal_length = K_left[0, 0]   # 焦距（像素单位）
depth_map = (baseline * focal_length) / (disparity + 1e-6)  # 避免除以零

# 显示深度图
depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imshow("Depth Map", depth_vis)
cv2.waitKey(0)
