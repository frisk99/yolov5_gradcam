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
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <cstdint> // 用于 int32_t, uint64_t 等

// --- 协议定义 ---
// 使用 #pragma pack 确保结构体在内存中是紧凑的，没有额外的填充字节
#pragma pack(push, 1)
// 客户端发来的请求包 (固定20字节)
struct Request {
    int32_t function_id;
    double arg1;
    double arg2;
};

// 服务器发回的响应头 (固定8字节)
struct ResponseHeader {
    int32_t status;     // 0 = OK, 1 = Error
    int32_t data_len;   // 后面跟随的数据长度
};
#pragma pack(pop)


// --- 服务器端需要暴露的函数 ---
double add(double a, double b) {
    return a + b;
}

double subtract(double a, double b) {
    return a - b;
}

std::string get_server_name() {
    return "Simple C++ RPC Server";
}

// --- 网络辅助函数 ---
// 从 socket 安全地读取指定长度的数据
bool recv_all(int sock, void* buffer, size_t length) {
    char* ptr = static_cast<char*>(buffer);
    while (length > 0) {
        int bytes_received = recv(sock, ptr, length, 0);
        if (bytes_received <= 0) {
            // 连接关闭或发生错误
            return false;
        }
        ptr += bytes_received;
        length -= bytes_received;
    }
    return true;
}

// 向 socket 安全地发送指定长度的数据
bool send_all(int sock, const void* buffer, size_t length) {
    const char* ptr = static_cast<const char*>(buffer);
    while (length > 0) {
        int bytes_sent = send(sock, ptr, length, 0);
        if (bytes_sent < 0) {
            // 发生错误
            return false;
        }
        ptr += bytes_sent;
        length -= bytes_sent;
    }
    return true;
}

// double 类型的主机序和网络序转换
double ntohd(double val) {
    uint64_t temp;
    static_assert(sizeof(double) == sizeof(uint64_t), "Size of double is not 64 bits");
    memcpy(&temp, &val, sizeof(double));
    temp = be64toh(temp); // be64toh: Big-Endian 64 to Host
    memcpy(&val, &temp, sizeof(double));
    return val;
}

double htond(double val) {
    uint64_t temp;
    static_assert(sizeof(double) == sizeof(uint64_t), "Size of double is not 64 bits");
    memcpy(&temp, &val, sizeof(double));
    temp = htobe64(temp); // htobe64: Host to Big-Endian 64
    memcpy(&val, &temp, sizeof(double));
    return val;
}


int main() {
    int server_fd, new_socket;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);

    // 创建 socket 文件描述符
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }
    
    // 绑定 socket 到端口 8080
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(8080);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }

    // 监听端口
    if (listen(server_fd, 3) < 0) {
        perror("listen");
        exit(EXIT_FAILURE);
    }

    std::cout << "Binary RPC Server is listening on port 8080..." << std::endl;

    // 接受一个客户端连接
    if ((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen)) < 0) {
        perror("accept");
        exit(EXIT_FAILURE);
    }
    std::cout << "Client connected." << std::endl;

    // 主循环: 处理来自客户端的请求
    while (true) {
        Request req;
        // 接收一个完整的20字节请求包
        if (!recv_all(new_socket, &req, sizeof(Request))) {
            std::cout << "Client disconnected." << std::endl;
            break;
        }

        // 将接收到的数据从网络字节序转换为主机字节序
        req.function_id = ntohl(req.function_id);
        req.arg1 = ntohd(req.arg1);
        req.arg2 = ntohd(req.arg2);

        ResponseHeader resp_header;
        resp_header.status = 0; // 默认成功

        // 根据 function_id 调用相应的函数
        switch (req.function_id) {
            case 1: { // add
                double result = add(req.arg1, req.arg2);
                result = htond(result); // 转换结果为网络字节序
                resp_header.data_len = sizeof(double);
                resp_header.status = htonl(resp_header.status);
                resp_header.data_len = htonl(resp_header.data_len);
                send_all(new_socket, &resp_header, sizeof(ResponseHeader));
                send_all(new_socket, &result, sizeof(double));
                break;
            }
            case 2: { // subtract
                double result = subtract(req.arg1, req.arg2);
                result = htond(result);
                resp_header.data_len = sizeof(double);
                resp_header.status = htonl(resp_header.status);
                resp_header.data_len = htonl(resp_header.data_len);
                send_all(new_socket, &resp_header, sizeof(ResponseHeader));
                send_all(new_socket, &result, sizeof(double));
                break;
            }
            case 3: { // get_server_name
                std::string name = get_server_name();
                resp_header.data_len = name.length();
                resp_header.status = htonl(resp_header.status);
                resp_header.data_len = htonl(resp_header.data_len);
                send_all(new_socket, &resp_header, sizeof(ResponseHeader));
                send_all(new_socket, name.c_str(), name.length());
                break;
            }
            default: { // 未知函数
                std::string error_msg = "Unknown function ID";
                resp_header.status = 1; // 错误状态
                resp_header.data_len = error_msg.length();
                resp_header.status = htonl(resp_header.status);
                resp_header.data_len = htonl(resp_header.data_len);
                send_all(new_socket, &resp_header, sizeof(ResponseHeader));
                send_all(new_socket, error_msg.c_str(), error_msg.length());
                break;
            }
        }
    }

    // 关闭连接
    close(new_socket);
    close(server_fd);

    return 0;
}
