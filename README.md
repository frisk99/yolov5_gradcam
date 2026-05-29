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
```python
import time
import mujoco
import mujoco.viewer

# 1. 加载你现成的 XML 文件 (请替换为你的实际文件路径)
xml_path = "your_model.xml"  
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# ==========================================
# 2. 修改特定 joint 的初始位置 (精确赋值)
# ==========================================
joint_name = "你的关节名称"  # 请替换为 XML 中的 joint name
target_value = 1.57         # 请填入你想要的精确数值 (旋转关节为弧度，平移关节为米)

try:
    # 获取关节 ID 和它在 qpos 数组中的地址
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    qpos_addr = model.jnt_qposadr[joint_id]
    
    # 赋精确数值
    data.qpos[qpos_addr] = target_value
    
    # 如果你有多个关节要改，直接复制上面两行并替换名字和数值即可，例如：
    # joint_id_2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint_2")
    # data.qpos[model.jnt_qposadr[joint_id_2]] = 0.5

except KeyError:
    print(f"⚠️ 找不到名为 '{joint_name}' 的关节，请检查你的 XML 文件拼写！")

# 3. 同步内部状态，让修改生效 (非常重要)
mujoco.mj_forward(model, data)

# ==========================================
# 4. 启动 Viewer 进行查看
# ==========================================
with mujoco.viewer.launch_passive(model, data) as viewer:
    
    # 【可选小技巧】如果你只是想静静地查看修改后的初始姿态，
    # 不希望一打开窗口模型就被重力拉倒，可以取消下面这行注释来默认暂停物理引擎：
    # viewer.lock()
    # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PAUSE] = True
    # viewer.unlock()

    print("Viewer 已启动。按空格键可以暂停/恢复物理仿真。")

    while viewer.is_running():
        step_start = time.time()
        
        # 步进物理引擎
        mujoco.mj_step(model, data)
        
        # 同步画面
        viewer.sync()

        # 保持实时速度
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
