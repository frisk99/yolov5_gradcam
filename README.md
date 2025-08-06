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
from scipy.spatial.transform import Rotation as R

# -------------------------- 1. 加载数据 ----------------------------
# 读取两张图像（上摄像头和下摄像头）
img_top = cv2.imread("top_image.jpg")
img_bottom = cv2.imread("bottom_image.jpg")

# 相机内参（需替换为实际值）
K = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]])  # 假设上下摄像头内参相同
D = np.zeros(5)  # 假设无畸变

# 上摄像头的外参（世界坐标系到相机坐标系）
xyz_top = np.array([0, 0, 0])       # 上摄像头位置（单位：米）
ypr_top = np.array([0, 0, 0])        # 上摄像头欧拉角（yaw, pitch, roll，单位：度）

# 下摄像头的外参
xyz_bottom = np.array([0, -0.1, 0])  # 下摄像头位置（Y方向低10cm）
ypr_bottom = np.array([0, 0, 0])     # 假设光轴平行

# -------------------------- 2. 计算相对位姿 ----------------------------
# 将欧拉角转换为旋转矩阵
def ypr_to_rotation(yaw, pitch, roll):
    """ 欧拉角（Z-Y-X顺序）转旋转矩阵 """
    return R.from_euler('zyx', [yaw, pitch, roll], degrees=True).as_matrix()

# 计算上下相机的旋转矩阵
R_top = ypr_to_rotation(*ypr_top)
R_bottom = ypr_to_rotation(*ypr_bottom)

# 计算下摄像头相对于上摄像头的变换
R_rel = R_top.T @ R_bottom                     # 相对旋转
T_rel = R_top.T @ (xyz_bottom - xyz_top)       # 相对平移（注意坐标系转换）
T_rel = T_rel.reshape(3, 1)                    # 转为列向量

print("相对旋转矩阵 R:\n", R_rel)
print("相对平移向量 T:\n", T_rel)

# -------------------------- 3. 立体校正 ----------------------------
image_size = img_top.shape[1], img_top.shape[0]  # (width, height)

# 立体校正
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    K, D, K, D,
    image_size, R_rel, T_rel,
    flags=cv2.CALIB_ZERO_DISPARITY,
    alpha=0.9
)

# 生成校正映射
map_top1, map_top2 = cv2.initUndistortRectifyMap(K, D, R1, P1, image_size, cv2.CV_16SC2)
map_bottom1, map_bottom2 = cv2.initUndistortRectifyMap(K, D, R2, P2, image_size, cv2.CV_16SC2)

# 校正图像
img_top_rect = cv2.remap(img_top, map_top1, map_top2, cv2.INTER_LINEAR)
img_bottom_rect = cv2.remap(img_bottom, map_bottom1, map_bottom2, cv2.INTER_LINEAR)

# -------------------------- 4. 立体匹配（垂直视差） ----------------------------
# 转换为灰度图
gray_top = cv2.cvtColor(img_top_rect, cv2.COLOR_BGR2GRAY)
gray_bottom = cv2.cvtColor(img_bottom_rect, cv2.COLOR_BGR2GRAY)

# 调整SGBM参数（适用于垂直视差）
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=64,    # 视差范围（需根据基线调整）
    blockSize=5,
    P1=8 * 3 * 5**2,
    P2=32 * 3 * 5**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    mode=cv2.STEREO_SGBM_MODE_HH
)

# 计算视差（注意：上下摄像头的视差是垂直的！）
disparity = stereo.compute(gray_top, gray_bottom).astype(np.float32) / 16.0

# -------------------------- 5. 计算深度图 ----------------------------
baseline = abs(T_rel[1][0])  # 基线距离（Y方向）
focal_length = K[0, 0]       # 焦距（像素单位）
depth_map = (baseline * focal_length) / (disparity + 1e-6)  # 避免除以零

# 过滤无效值
depth_map[disparity <= 0] = 0

# -------------------------- 6. 可视化 ----------------------------
# 归一化显示
disparity_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

cv2.imshow("Top Image Rectified", img_top_rect)
cv2.imshow("Bottom Image Rectified", img_bottom_rect)
cv2.imshow("Disparity", disparity_vis)
cv2.imshow("Depth Map", depth_vis)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存结果
cv2.imwrite("disparity.png", disparity_vis)
cv2.imwrite("depth_map.png", depth_vis)
```

---

### **关键步骤说明**
1. **外参处理**  
   - 将欧拉角 `YPR` 转换为旋转矩阵 `R`。  
   - 计算下摄像头 **相对于上摄像头** 的 `R_rel` 和 `T_rel`（注意坐标系转换）。  

2. **立体校正**  
   - 使用 `cv2.stereoRectify` 计算校正映射，确保图像行对齐。  
   - 上下摄像头的视差是 **垂直方向** 的（传统左右双目是水平方向）。  

3. **立体匹配**  
   - `StereoSGBM` 默认搜索水平视差，但上下摄像头的视差是垂直的，因此无需修改算法（校正后的图像已处理）。  
   - 如果效果不佳，尝试调整 `numDisparities` 和 `blockSize`。

4. **深度计算**  
   - 公式：`深度 = (基线 × 焦距) / 视差`  
   - 基线是 `T_rel[1][0]`（Y方向距离），焦距是 `K[0,0]`。  

---

### **参数调整建议**
| 参数 | 作用 | 调整方法 |
|------|------|----------|
| `numDisparities` | 视差搜索范围 | 增大可检测更远距离，但会降低速度 |
| `blockSize` | 匹配块大小 | 增大可抗噪声，但会丢失细节 |
| `baseline` | 基线距离 | 必须准确测量（单位：米） |
| `K` | 内参矩阵 | 通过标定获取真实值 |

---

### **常见问题解决**
- **问题：深度图全黑**  
  - 检查 `disparity` 是否全为负数或零（可能是 `T_rel` 方向错误）。  
  - 验证 `baseline` 和 `focal_length` 的单位一致性（均为米或毫米）。  

- **问题：物体边缘断裂**  
  - 使用 `cv2.StereoSGBM_MODE_HH` 高精度模式。  
  - 后处理时应用加权最小二乘滤波（WLS）：  
    ```python
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo_left)
    filtered_disp = wls_filter.filter(disparity, img_top_rect)
    ```
