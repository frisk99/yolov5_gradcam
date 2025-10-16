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
import struct
import io
import time
from PIL import Image

# 初始化 session state
if 'subscribed' not in st.session_state:
    st.session_state.subscribed = False
if 'client_socket' not in st.session_state:
    st.session_state.client_socket = None

def connect_to_server():
    """连接到服务器并将其存储在 session_state 中"""
    if st.session_state.client_socket is None:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(('localhost', 8080))
            st.session_state.client_socket = sock
            st.info("成功连接到服务器。")
        except ConnectionRefusedError:
            st.error("连接被拒绝。请确保C++服务器正在运行。")
            st.session_state.client_socket = None
            st.session_state.subscribed = False

def disconnect_from_server():
    """关闭并清理 socket 连接"""
    if st.session_state.client_socket:
        st.session_state.client_socket.close()
        st.session_state.client_socket = None
        st.session_state.subscribed = False
        st.info("已从服务器断开。")

def send_command(command_id, payload=b''):
    """向服务器发送一个命令"""
    if st.session_state.client_socket is None:
        connect_to_server()
    
    if st.session_state.client_socket:
        try:
            header = struct.pack('!II', command_id, len(payload))
            st.session_state.client_socket.sendall(header + payload)
            return True
        except (BrokenPipeError, ConnectionResetError):
            st.error("与服务器的连接已断开。")
            disconnect_from_server()
            return False
    return False

# --- UI 和逻辑 ---

st.title("持续图像订阅客户端")

st.header("控制面板")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("连接服务器", key="connect", disabled=st.session_state.client_socket is not None):
        connect_to_server()
        # 强制重新运行以更新UI状态
        st.rerun()

with col2:
    if st.button("订阅图像流", key="subscribe", disabled=not st.session_state.client_socket or st.session_state.subscribed):
        if send_command(3):
            st.session_state.subscribed = True
            st.rerun()

with col3:
    if st.button("取消订阅", key="unsubscribe", disabled=not st.session_state.subscribed):
        if send_command(4):
            st.session_state.subscribed = False
            # 在这里我们不立即断开连接，以便可以重新订阅
            st.rerun()

# 图像显示区域
st.header("图像直播")
image_placeholder = st.empty()
info_placeholder = st.empty()

if not st.session_state.subscribed:
    image_placeholder.info("当前未订阅。请连接并订阅以查看图像。")

# 当处于订阅状态时，进入接收循环
while st.session_state.subscribed:
    try:
        sock = st.session_state.client_socket
        if sock is None:
            st.session_state.subscribed = False
            break

        # 接收自定义的图像数据头 (ID=5)
        header_data = sock.recv(8)
        if not header_data:
            raise ConnectionResetError
        
        command_id, payload_len = struct.unpack('!II', header_data)
        
        if command_id == 5: # 确认是图像数据流
            # 接收整个负载
            full_payload = b''
            while len(full_payload) < payload_len:
                chunk = sock.recv(payload_len - len(full_payload))
                if not chunk: raise ConnectionResetError
                full_payload += chunk
            
            # 从负载中解析数据
            offset = 0
            image_id, = struct.unpack('!i', full_payload[offset:offset+4])
            offset += 4
            name_len, = struct.unpack('!I', full_payload[offset:offset+4])
            offset += 4
            name = full_payload[offset:offset+name_len].decode('utf-8')
            offset += name_len
            num_bboxes, = struct.unpack('!I', full_payload[offset:offset+4])
            offset += 4
            bboxes = []
            for _ in range(num_bboxes):
                bbox = struct.unpack('!iiii', full_payload[offset:offset+16])
                bboxes.append(bbox)
                offset += 16
            
            image_size, = struct.unpack('!Q', full_payload[offset:offset+8])
            offset += 8
            
            image_data = full_payload[offset:offset+image_size]

            # 显示图像和信息
            image = Image.open(io.BytesIO(image_data))
            image_placeholder.image(image, caption=f"实时图像: {name}", use_column_width=True)
            info_placeholder.info(f"ID: {image_id}, BBoxes: {bboxes}")

        else:
            # 收到意外的命令
            info_placeholder.warning(f"从服务器收到意外的命令ID: {command_id}")
            # 跳过这个包
            sock.recv(payload_len)

    except (ConnectionResetError, BrokenPipeError):
        st.error("与服务器的连接已断开。")
        disconnect_from_server()
        st.rerun()
    except Exception as e:
        st.error(f"发生错误: {e}")
        disconnect_from_server()
        st.rerun()

# 当循环结束时（因为取消订阅），清理占位符
if not st.session_state.subscribed:
    image_placeholder.info("订阅已停止。")
    info_placeholder.empty()

