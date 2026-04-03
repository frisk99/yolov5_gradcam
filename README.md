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
'''python
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import urllib.request

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
manager = ConnectionManager()
message_queue: asyncio.Queue | None = None
consumer_task: asyncio.Task | None = None
main_loop: asyncio.AbstractEventLoop | None = None
async def ws_message_consumer():
    print("ws_message_consumer\n")
    while True:
        try:
            payload = await message_queue.get()
            if payload is None:
                break
            t_id = payload.get("t_id")
            data = payload.get("data")
            target_ws = manager.active_connections.get(t_id)
            if target_ws:
                await target_ws.send_json({"from_id": "system_queue", "data": data})
                print(f"forward mseesage to {t_id}")
            message_queue.task_done()
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"forward Error: {e}")
@asynccontextmanager
async def lifespan(app: FastAPI):
    global message_queue, consumer_task, main_loop
    main_loop = asyncio.get_running_loop()
    message_queue = asyncio.Queue()
    consumer_task = asyncio.create_task(ws_message_consumer())
    print("FastAPI Start")
    yield
    print("FastAPI Close")
    if message_queue:
        await message_queue.put(None)
    if consumer_task:
        await consumer_task
    print("FastAPI Close done")
app = FastAPI(lifespan=lifespan)
def sync_send_to_ws(target_id: str, data: dict | str):
    print("sync_send_to_ws")
    if not main_loop or not message_queue:
        print("Error queue not found")
        return
    payload = {"t_id": target_id, "data": data}
    main_loop.call_soon_threadsafe(message_queue.put_nowait, payload)
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            json_data = await websocket.receive_json()
            t_id = json_data.get("t_id")
            data = json_data.get("data")
            if t_id and data:
                target_ws = manager.active_connections.get(t_id)
                if target_ws:
                    await target_ws.send_json({"from_id": client_id, "data": data})
    except WebSocketDisconnect:
        manager.disconnect(client_id)
import time
@app.post("/test_sync_send/{target_id}")
def test_sync_send(target_id: str):
    sync_send_to_ws(target_id=target_id, data="message 1")
    sync_send_to_ws(target_id=target_id, data="message 2")
    sync_send_to_ws(target_id=target_id, data="message 3")
    return {"msg": "sucess"}
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

def trigger_http_endpoint(target_id: str):
    url = f"http://localhost:8000/test_sync_send/{target_id}"
    try:
        req = urllib.request.Request(url, method="POST")
        with urllib.request.urlopen(req) as response:
            res_data = response.read().decode('utf-8')
            print(f"request done: {res_data}\n")
    except Exception as e:
        print(f"Error: {e}")