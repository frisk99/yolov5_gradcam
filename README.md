# YOLO-V5 GRADCAM

I constantly desired to know to which part of an object the object-detection models pay more attention. So I searched for it, but I didn't find any for Yolov5.
Here is my implementation of Grad-cam for YOLO-v5. To load the model I used the yolov5's main codes, and for computing GradCam I used the codes from the gradcam_plus_plus-pytorch repository.
Please follow my GitHub account and star ⭐ the project if this functionality benefits your research or projects.

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
import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl

class FilePreviewApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("File Preview App")
        self.setGeometry(100, 100, 800, 600)

        # 初始化组件
        self.label = QLabel("No file selected", self)
        self.label.setGeometry(20, 20, 760, 400)
        self.label.setStyleSheet("border: 1px solid black;")

        self.select_button = QPushButton("Select File", self)
        self.select_button.setGeometry(20, 440, 120, 40)
        self.select_button.clicked.connect(self.select_file)

        self.run_button = QPushButton("Run Function", self)
        self.run_button.setGeometry(160, 440, 120, 40)
        self.run_button.clicked.connect(self.run_function)

        self.video_widget = QVideoWidget(self)
        self.video_widget.setGeometry(20, 20, 760, 400)
        self.video_widget.hide()

        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.media_player.setVideoOutput(self.video_widget)

        self.selected_file = None

    def select_file(self):
        file_filter = "Media Files (*.png *.jpg *.jpeg *.bmp *.mp4 *.avi *.mov)"
        file_path, _ = QFileDialog.getOpenFileName(self, "Select a file", "", file_filter)

        if file_path:
            self.selected_file = file_path
            self.preview_file(file_path)

    def preview_file(self, file_path):
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            # 预览图片
            self.media_player.stop()
            self.video_widget.hide()
            self.label.show()
            pixmap = QPixmap(file_path)
            self.label.setPixmap(pixmap.scaled(self.label.width(), self.label.height(), aspectRatioMode=True))
        elif file_path.lower().endswith(('.mp4', '.avi', '.mov')):
            # 预览视频
            self.label.hide()
            self.video_widget.show()
            self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(file_path)))
            self.media_player.play()

    def run_function(self):
        if self.selected_file:
            print(f"Running function with file: {self.selected_file}")
            # 在这里添加你的自定义逻辑，将 self.selected_file 作为参数
            self.modify_file(self.selected_file)
        else:
            print("No file selected")

    def modify_file(self, file_path):
        # 假设这是一个修改图片或视频的函数，修改完成后重新预览
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            # 对图片进行一些修改，比如转换为灰度图像
            image = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                print("Error: Could not read the image file.")
                return
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            height, width = gray_image.shape
            bytes_per_line = width
            q_image = QImage(gray_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(q_image)
            self.label.setPixmap(pixmap.scaled(self.label.width(), self.label.height(), aspectRatioMode=True))
            self.label.show()
            self.video_widget.hide()
        elif file_path.lower().endswith(('.mp4', '.avi', '.mov')):
            # 对视频进行一些修改（这里只是示例，无法直接将视频转换为灰度并播放）
            self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(file_path)))
            self.media_player.play()
            self.video_widget.show()
            self.label.hide()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = FilePreviewApp()
    main_window.show()
    sys.exit(app.exec_())
