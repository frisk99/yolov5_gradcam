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

#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>

// 使用原子布尔变量来安全地跨线程共享订阅状态
std::atomic<bool> is_subscribed(false);

// (发送和接收数据的辅助函数保持不变)
bool send_data(int sock, const void* data, size_t size) {
    // MSG_NOSIGNAL 防止在客户端断开连接时程序因 SIGPIPE 信号而崩溃
    ssize_t sent_bytes = send(sock, data, size, MSG_NOSIGNAL);
    if (sent_bytes < 0) {
        // 如果发送失败 (例如，连接已关闭), 返回 false
        return false;
    }
    return sent_bytes == size;
}

bool receive_data(int sock, void* data, size_t size) {
    ssize_t received_bytes = recv(sock, data, size, 0);
    if (received_bytes <= 0) {
        // 如果接收失败或客户端关闭了连接, 返回 false
        return false;
    }
    return received_bytes == size;
}


// 图片发送线程函数
void image_sender_thread(int client_socket) {
    std::cout << "Image sender thread started." << std::endl;

    // 准备要发送的图片数据 (只准备一次)
    int32_t image_id = 123;
    std::string name = "streaming_image.jpg";
    int32_t num_bboxes = 1;
    std::vector<int32_t> bboxes = {10, 20, 100, 150};
    cv::Mat image = cv::imread("sample_image.jpg", cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Could not read the image for streaming." << std::endl;
        is_subscribed = false; // 停止流
        return;
    }
    std::vector<uchar> encoded_image;
    cv::imencode(".jpg", image, encoded_image);
    int64_t image_size = encoded_image.size();
    
    // 定义一个新的命令ID用于图像数据流，以便客户端区分
    int32_t image_data_command_id = 5;

    while (is_subscribed) {
        // --- 序列化并发送数据 ---
        // 1. 发送命令ID (5) 和总负载长度
        int32_t payload_length = sizeof(image_id) + sizeof(int32_t) + name.length() + sizeof(num_bboxes) + bboxes.size() * sizeof(int32_t) + sizeof(image_size) + image_size;
        if (!send_data(client_socket, &image_data_command_id, sizeof(image_data_command_id))) break;
        if (!send_data(client_socket, &payload_length, sizeof(payload_length))) break;
        
        // 2. 发送具体数据
        if (!send_data(client_socket, &image_id, sizeof(image_id))) break;
        int32_t name_len = name.length();
        if (!send_data(client_socket, &name_len, sizeof(name_len))) break;
        if (!send_data(client_socket, name.c_str(), name_len)) break;
        if (!send_data(client_socket, &num_bboxes, sizeof(num_bboxes))) break;
        if (!send_data(client_socket, bboxes.data(), bboxes.size() * sizeof(int32_t))) break;
        if (!send_data(client_socket, &image_size, sizeof(image_size))) break;
        if (!send_data(client_socket, encoded_image.data(), image_size)) break;

        std::cout << "Sent one image frame." << std::endl;

        // 等待一段时间再发送下一帧，避免网络拥塞
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << "Image sender thread stopped." << std::endl;
}

// (adduserinfo 和 deleteusrinfo 函数保持不变)
void handle_adduserinfo(int client_socket) { /* ... */ }
void handle_deleteusrinfo(int client_socket) { /* ... */ }

// 主循环，用于接收客户端指令
void command_listener_loop(int client_socket) {
    std::thread sender;

    while (true) {
        int32_t command_id;
        // 使用 MSG_PEEK 来检查数据，但不从缓冲区移除，以判断连接是否仍然有效
        if (recv(client_socket, &command_id, sizeof(command_id), MSG_PEEK) <= 0) {
            std::cout << "Client disconnected." << std::endl;
            is_subscribed = false; // 确保图片发送线程会停止
            break;
        }
        
        // 真正地接收数据
        if (!receive_data(client_socket, &command_id, sizeof(command_id))) break;

        // 第二个header字段（payload_length）由每个handler自行处理
        int32_t payload_length;

        switch (command_id) {
            case 1: // adduserinfo
                receive_data(client_socket, &payload_length, sizeof(payload_length));
                handle_adduserinfo(client_socket);
                break;
            case 2: // deleteusrinfo
                receive_data(client_socket, &payload_length, sizeof(payload_length));
                handle_deleteusrinfo(client_socket);
                break;
            case 3: // subscribeimage
                receive_data(client_socket, &payload_length, sizeof(payload_length)); // 读取空的payload length
                std::cout << "Received subscribeimage request." << std::endl;
                if (!is_subscribed) {
                    is_subscribed = true;
                    // 如果之前的线程已结束，则创建一个新的
                    if (sender.joinable()) {
                        sender.join();
                    }
                    sender = std::thread(image_sender_thread, client_socket);
                }
                break;
            case 4: // unsubscribeimage
                receive_data(client_socket, &payload_length, sizeof(payload_length)); // 读取空的payload length
                std::cout << "Received unsubscribeimage request." << std::endl;
                is_subscribed = false;
                if (sender.joinable()) {
                    sender.join();
                }
                break;
            default:
                std::cerr << "Unknown command: " << command_id << std::endl;
                // 跳过未知的负载
                receive_data(client_socket, &payload_length, sizeof(payload_length));
                std::vector<char> unknown_payload(payload_length);
                receive_data(client_socket, unknown_payload.data(), payload_length);
        }
    }

    // 清理
    is_subscribed = false;
    if (sender.joinable()) {
        sender.join();
    }
}


int main() {
    int server_fd;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);

    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt));
    
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(8080);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }

    if (listen(server_fd, 3) < 0) {
        perror("listen");
        exit(EXIT_FAILURE);
    }
    
    std::cout << "Server listening on port 8080" << std::endl;

    while (true) {
        int new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen);
        if (new_socket < 0) {
            perror("accept");
            continue; // 继续等待下一个连接
        }
        
        std::cout << "New client connected. Starting command listener..." << std::endl;
        // 为每个客户端创建一个新线程来处理指令
        // （为简单起见，此示例一次只处理一个客户端，但可以轻松扩展为多客户端）
        command_listener_loop(new_socket);
        close(new_socket);
        std::cout << "Client session ended." << std::endl;
    }

    close(server_fd);
    return 0;
}
