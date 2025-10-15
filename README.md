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

# streamlit_client.py
import streamlit as st
import socket
import json
import pandas as pd

# Use session_state to maintain the socket connection across Streamlit reruns
if 'sock' not in st.session_state:
    st.session_state.sock = None
if 'image_data' not in st.session_state:
    st.session_state.image_data = None

def connect_to_server(host, port):
    """Establishes a connection to the socket server."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        # Receive the initial welcome message from the server
        welcome_msg = sock.recv(1024).decode('utf-8')
        st.toast(f"Server says: {welcome_msg}")
        return sock
    except ConnectionRefusedError:
        st.error("Connection refused. Is the server running?")
    except Exception as e:
        st.error(f"An error occurred during connection: {e}")
    return None

def send_command(command):
    """Sends a command to the server and returns the response."""
    if st.session_state.sock:
        try:
            # Add newline character to signal end of message
            st.session_state.sock.sendall(f"{command}\n".encode('utf-8'))
            response = st.session_state.sock.recv(4096).decode('utf-8').strip()
            return response
        except (ConnectionResetError, BrokenPipeError):
            st.error("Connection to server lost. Please reconnect.")
            disconnect_from_server()
            return None
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return None
    else:
        st.warning("Not connected to the server.")
        return None

def disconnect_from_server():
    """Closes the socket connection."""
    if st.session_state.sock:
        st.session_state.sock.close()
    st.session_state.sock = None
    st.session_state.image_data = None # Clear data on disconnect
    st.toast("Disconnected from server.")


# --- Streamlit UI ---

st.set_page_config(layout="wide")
st.title("Socket Client Control Panel")

# --- Sidebar for Connection and Commands ---
with st.sidebar:
    st.header("Server Connection")
    server_host = st.text_input("Server Host", "127.0.0.1")
    server_port = st.number_input("Server Port", 1, 65535, 8080)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Connect", use_container_width=True, disabled=st.session_state.sock is not None):
            st.session_state.sock = connect_to_server(server_host, server_port)
            st.rerun() # Rerun to update button states
    
    with col2:
        if st.button("Disconnect", use_container_width=True, disabled=st.session_state.sock is None):
            disconnect_from_server()
            st.rerun()

    st.divider()

    # --- Add User ---
    with st.expander("Add User Info", expanded=True):
        add_name = st.text_input("Name", "JohnDoe")
        add_image = st.text_input("Image Path", "/images/john.jpg")
        add_id = st.number_input("ID", 1, 99999, 123, format="%d")
        if st.button("Add User"):
            command = f"adduserinfo {add_name} {add_image} {add_id}"
            response = send_command(command)
            if response:
                st.success(f"Server response: `{response}`")

    st.divider()

    # --- Delete User ---
    with st.expander("Delete User Info", expanded=True):
        del_name = st.text_input("Name", "JaneDoe", key="del_name")
        del_id = st.number_input("ID", 1, 99999, 456, format="%d", key="del_id")
        if st.button("Delete User"):
            command = f"deleteusrinfo {del_name} {del_id}"
            response = send_command(command)
            if response:
                st.success(f"Server response: `{response}`")

    st.divider()
    
    # --- Image Subscription ---
    st.header("Image Subscription")
    col3, col4 = st.columns(2)
    with col3:
        if st.button("Subscribe", use_container_width=True):
            response = send_command("subscribeimage")
            if response:
                try:
                    # The response is a JSON string, parse it
                    data = json.loads(response)
                    st.session_state.image_data = data
                    st.toast("Subscribed and received data!")
                except json.JSONDecodeError:
                    st.error(f"Failed to parse JSON from server: {response}")

    with col4:
        if st.button("Unsubscribe", use_container_width=True):
            response = send_command("unsubscribeimage")
            if response:
                st.session_state.image_data = None # Clear data on unsubscribe
                st.success(f"Server response: `{response}`")


# --- Main Panel for Displaying Data ---
st.header("Subscribed Image Data")

if st.session_state.sock is None:
    st.info("Please connect to the server using the sidebar.")
elif st.session_state.image_data is None:
    st.info("No active image subscription. Click 'Subscribe' in the sidebar to get data.")
else:
    st.success("Displaying received data from the server:")
    data = st.session_state.image_data
    
    st.subheader("Raw JSON Data")
    st.json(data)
    
    st.subheader("Formatted Data")
    try:
        ids = data.get("ids", [])
        names = data.get("names", [])
        bboxes_flat = data.get("bboxes", [])

        # Reshape bboxes into groups of 4 (x, y, w, h)
        bboxes = [bboxes_flat[i:i + 4] for i in range(0, len(bboxes_flat), 4)]
        
        max_len = max(len(ids), len(names), len(bboxes))
        ids.extend([None] * (max_len - len(ids)))
        names.extend([None] * (max_len - len(names)))
        bboxes.extend([None] * (max_len - len(bboxes)))

        df = pd.DataFrame({
            "ID": ids,
            "Name": names,
            "Bounding Box (x,y,w,h)": bboxes
        })
        st.dataframe(df, use_container_width=True)

    except Exception as e:
        st.error(f"Error while formatting data: {e}")

