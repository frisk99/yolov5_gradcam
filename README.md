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
#include <cstring> // 用于 memcpy
#include <algorithm> // 用于 std::reverse

// --- 字节序处理辅助函数 ---

// 检查主机字节序是否为小端
bool is_little_endian() {
    uint16_t x = 1;
    return *(uint8_t*)&x == 1;
}

// double 类型的主机序和网络序（大端）转换
double htond(double val) {
    if (is_little_endian()) {
        uint64_t temp;
        memcpy(&temp, &val, sizeof(double));
        temp = __builtin_bswap64(temp); // GCC/Clang 内置函数，效率高
        memcpy(&val, &temp, sizeof(double));
    }
    return val;
}

double ntohd(double val) {
    return htond(val); // 转换是可逆的
}


// --- 网络通信辅助函数 ---

// 从 socket 安全地读取指定长度的数据
bool recv_all(int sock, void* buffer, size_t length) {
    char* ptr = static_cast<char*>(buffer);
    while (length > 0) {
        ssize_t bytes_received = recv(sock, ptr, length, 0);
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
        ssize_t bytes_sent = send(sock, ptr, length, 0);
        if (bytes_sent < 0) {
            // 发生错误
            return false;
        }
        ptr += bytes_sent;
        length -= bytes_sent;
    }
    return true;
}

// 接收一个字符串 (先收长度，再收内容)
bool recv_string(int sock, std::string& s) {
    uint32_t len_net;
    if (!recv_all(sock, &len_net, sizeof(len_net))) return false;
    uint32_t len = ntohl(len_net);
    if (len > 0) { // 只有当长度大于0时才接收
        std::vector<char> buffer(len);
        if (!recv_all(sock, buffer.data(), len)) return false;
        s.assign(buffer.begin(), buffer.end());
    } else {
        s.clear();
    }
    return true;
}

// 发送一个字符串
bool send_string(int sock, const std::string& s) {
    uint32_t len_net = htonl(s.length());
    if (!send_all(sock, &len_net, sizeof(len_net))) return false;
    if (!s.empty()) {
        if (!send_all(sock, s.c_str(), s.length())) return false;
    }
    return true;
}


// --- 服务器端需要暴露的函数 ---

double add(double a, double b) {
    return a + b;
}

std::string concat(const std::string& s1, const std::string& s2) {
    return s1 + " " + s2;
}

std::string get_server_name() {
    return "Advanced C++ RPC Server";
}


int main() {
    // --- Socket 初始化 ---
    int server_fd, new_socket;
    struct sockaddr_in address;
    int addrlen = sizeof(address);

    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }
    
    // 允许地址重用，避免 "Address already in use" 错误
    int opt = 1;
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))) {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }
    
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

    std::cout << "Advanced RPC Server is listening on port 8080..." << std::endl;

    if ((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen)) < 0) {
        perror("accept");
        exit(EXIT_FAILURE);
    }
    std::cout << "Client connected." << std::endl;

    // --- 主循环：处理请求 ---
    while (true) {
        uint32_t function_id_net;
        // 1. 首先只接收4字节的 function_id
        if (!recv_all(new_socket, &function_id_net, sizeof(uint32_t))) {
            std::cout << "Client disconnected." << std::endl;
            break;
        }
        uint32_t function_id = ntohl(function_id_net);
        
        // 定义响应头
        struct ResponseHeader {
            int32_t status; // 0 for OK, 1 for Error
        };
        ResponseHeader resp_header;
        resp_header.status = 0; // 默认为成功

        // 2. 根据 function_id，接收对应参数并执行函数
        switch (function_id) {
            case 1: { // add(double, double)
                double arg1_net, arg2_net;
                if (!recv_all(new_socket, &arg1_net, sizeof(double)) || !recv_all(new_socket, &arg2_net, sizeof(double))) {
                    std::cerr << "Failed to receive double arguments." << std::endl;
                    goto connection_lost;
                }
                
                double result = add(ntohd(arg1_net), ntohd(arg2_net));
                
                // 发送响应
                resp_header.status = htonl(resp_header.status);
                double result_net = htond(result);
                send_all(new_socket, &resp_header, sizeof(ResponseHeader));
                send_all(new_socket, &result_net, sizeof(double));
                break;
            }
            case 2: { // concat(string, string)
                std::string s1, s2;
                if (!recv_string(new_socket, s1) || !recv_string(new_socket, s2)) {
                    std::cerr << "Failed to receive string arguments." << std::endl;
                    goto connection_lost;
                }
                
                std::string result = concat(s1, s2);
                
                // 发送响应
                resp_header.status = htonl(resp_header.status);
                send_all(new_socket, &resp_header, sizeof(ResponseHeader));
                send_string(new_socket, result);
                break;
            }
            case 3: { // get_server_name()
                std::string name = get_server_name();
                
                // 发送响应
                resp_header.status = htonl(resp_header.status);
                send_all(new_socket, &resp_header, sizeof(ResponseHeader));
                send_string(new_socket, name);
                break;
            }
            default: {
                std::cerr << "Received unknown function ID: " << function_id << std::endl;
                resp_header.status = 1; // 设置错误状态
                
                std::string error_msg = "Unknown function ID";
                
                // 发送错误响应
                resp_header.status = htonl(resp_header.status);
                send_all(new_socket, &resp_header, sizeof(ResponseHeader));
                send_string(new_socket, error_msg);
                break;
            }
        }
    }

connection_lost: // goto 标签，用于跳出循环
    std::cout << "Closing client connection..." << std::endl;
    close(new_socket);
    close(server_fd);

    return 0;
}
