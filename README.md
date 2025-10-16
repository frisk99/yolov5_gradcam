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

import streamlit as st
import socket
import numpy as np
import cv2
import struct
import time

# --- 页面配置 ---
st.set_page_config(
    page_title="实时图像流播放器",
    page_icon="📹"
)

st.title("📹 C++ 服务器图像流播放")

# --- Socket 连接参数 ---
HOST = '127.0.0.1'  # C++ 服务器的 IP 地址
PORT = 8080         # C++ 服务器的端口

# 使用 Streamlit 的 Session State 来存储 socket 对象，避免每次刷新都重连
if 'sock' not in st.session_state:
    st.session_state.sock = None

def connect_to_server():
    """建立到服务器的连接"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((HOST, PORT))
        st.session_state.sock = sock
        return True
    except ConnectionRefusedError:
        st.error(f"连接被拒绝。请确保 C++ 服务器正在运行于 {HOST}:{PORT}。")
        st.session_state.sock = None
        return False
    except Exception as e:
        st.error(f"连接失败: {e}")
        st.session_state.sock = None
        return False

def recv_all(sock, count):
    """确保接收到指定字节数的数据"""
    buf = b''
    while len(buf) < count:
        new_buf = sock.recv(count - len(buf))
        if not new_buf:
            return None
        buf += new_buf
    return buf

def main():
    # --- 侧边栏控制 ---
    st.sidebar.header("控制面板")
    
    if st.sidebar.button('连接服务器', key='connect'):
        if st.session_state.sock is None:
            if connect_to_server():
                st.sidebar.success("已成功连接到服务器！")
            else:
                st.sidebar.error("连接失败。")
        else:
            st.sidebar.warning("已经连接。如需重连，请先断开。")

    if st.sidebar.button('开始播放', key='play', disabled=(st.session_state.sock is None)):
        st.session_state.is_playing = True

    if st.sidebar.button('停止播放', key='stop'):
        st.session_state.is_playing = False
        
    if st.sidebar.button('断开连接', key='disconnect'):
        if st.session_state.sock:
            st.session_state.sock.close()
            st.session_state.sock = None
            st.session_state.is_playing = False
            st.sidebar.info("已断开连接。")

    # 初始化播放状态
    if 'is_playing' not in st.session_state:
        st.session_state.is_playing = False

    # --- 主显示区域 ---
    image_placeholder = st.empty()
    image_placeholder.info("请先连接服务器，然后点击 '开始播放'。")

    if st.session_state.get('is_playing') and st.session_state.sock:
        try:
            while st.session_state.is_playing:
                # 1. 接收图像大小 (long 类型，8 字节)
                size_data = recv_all(st.session_state.sock, 8)
                if size_data is None:
                    st.warning("与服务器的连接已断开。")
                    st.session_state.is_playing = False
                    st.session_state.sock = None
                    break
                
                # 解包得到图像大小
                image_size = struct.unpack('<q', size_data)[0]

                # 2. 接收图像数据
                image_data = recv_all(st.session_state.sock, image_size)
                if image_data is None:
                    st.warning("与服务器的连接已断开。")
                    st.session_state.is_playing = False
                    st.session_state.sock = None
                    break

                # 3. 解码并显示图像
                # 将字节数据转换为 numpy 数组
                nparr = np.frombuffer(image_data, np.uint8)
                # 从数组解码图像
                img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img_np is not None:
                    # OpenCV 读取的格式是 BGR，需要转换为 RGB 以在网页上正确显示
                    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                    image_placeholder.image(img_rgb, caption="实时视频流", use_column_width=True)
                else:
                    st.error("解码图像失败！")
                
                # 控制刷新率，给 Streamlit 一点时间来渲染
                time.sleep(0.01)

        except (ConnectionResetError, BrokenPipeError):
            st.error("与服务器的连接被重置。请重新连接。")
            st.session_state.sock.close()
            st.session_state.sock = None
            st.session_state.is_playing = False
        except Exception as e:
            st.error(f"发生未知错误: {e}")
            if st.session_state.sock:
                st.session_state.sock.close()
            st.session_state.sock = None
            st.session_state.is_playing = False


if __name__ == '__main__':
    main()

