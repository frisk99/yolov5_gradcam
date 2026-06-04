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
import os
import argparse
import numpy as np

from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.headless import HeadlessRenderer
from aitviewer.viewer import Viewer


def preview_amass(npz_path, fps, width, height, show_joint_angles):
    seq = SMPLSequence.from_amass(
        npz_data_path=npz_path,
        fps_out=fps,
        name=os.path.basename(npz_path),
        show_joint_angles=show_joint_angles,
    )

    v = Viewer(size=(width, height))
    v.run_animations = True
    v.scene.camera.position = np.array([10.0, 2.5, 0.0])
    v.scene.add(seq)
    v.run()


def export_amass_to_mp4(npz_path, out_path, fps, width, height, show_joint_angles):
    seq = SMPLSequence.from_amass(
        npz_data_path=npz_path,
        fps_out=fps,
        name=os.path.basename(npz_path),
        show_joint_angles=show_joint_angles,
    )

    # 可选：让人体稍微透明一点
    seq.color = seq.color[:3] + (0.8,)

    renderer = HeadlessRenderer(size=(width, height))
    renderer.scene.add(seq)

    # 让相机自动跟随人物，官方示例也是这么做的
    renderer.lock_to_node(seq, (2, 2, 2), smooth_sigma=5.0)

    # 注意：虽然参数名叫 video_dir，但这里传的是完整 mp4 文件路径
    renderer.save_video(video_dir=out_path)

    print(f"Saved video to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, required=True, help="AMASS .npz path")
    parser.add_argument("--out", type=str, default="output.mp4", help="output mp4 path")
    parser.add_argument("--fps", type=float, default=60.0, help="output fps")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--show-joints", action="store_true", help="show joint coordinate systems")
    parser.add_argument("--preview", action="store_true", help="preview only, do not export")
    args = parser.parse_args()

    if args.preview:
        preview_amass(
            npz_path=args.npz,
            fps=args.fps,
            width=args.width,
            height=args.height,
            show_joint_angles=args.show_joints,
        )
    else:
        export_amass_to_mp4(
            npz_path=args.npz,
            out_path=args.out,
            fps=args.fps,
            width=args.width,
            height=args.height,
            show_joint_angles=args.show_joints,
        )


if __name__ == "__main__":
    main()