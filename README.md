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
#include <vector>
#include <string>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <dirent.h>
#include <algorithm>
#include <cstdint> // For fixed-width integers

#include <opencv2/opencv.hpp>

// 确保结构体在内存中是紧凑的，没有填充字节
// 这对于与 Python struct 模块的兼容性至关重要
#pragma pack(push, 1)
struct PacketHeader {
    int32_t bbox_x;
    int32_t bbox_y;
    int32_t bbox_width;
    int32_t bbox_height;
    int64_t image_size;
};
#pragma pack(pop)

// 函数：发送一个完整的消息
bool send_all(int socket, const void* buffer, size_t length) {
    const char* ptr = (const char*)buffer;
    while (length > 0) {
        int bytes_sent = send(socket, ptr, length, 0);
        if (bytes_sent < 1) {
            return false;
        }
        ptr += bytes_sent;
        length -= bytes_sent;
    }
    return true;
}

int main() {
    // --- 1. 获取图片列表 (与之前相同) ---
    std::string image_dir = "images/";
    std::vector<std::string> image_paths;
    // ... (此处代码与上一版完全相同，故省略)
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(image_dir.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            std::string filename = ent->d_name;
            if (filename.find(".jpg") != std::string::npos || filename.find(".png") != std::string::npos) {
                image_paths.push_back(image_dir + filename);
            }
        }
        closedir(dir);
        std::sort(image_paths.begin(), image_paths.end());
    } else {
        perror("Could not open image directory");
        return 1;
    }
    if (image_paths.empty()) {
        std::cerr << "No images found in directory." << std::endl;
        return 1;
    }

    // --- 2. 创建并配置 Socket (与之前相同) ---
    int server_fd, new_socket;
    struct sockaddr_in address;
    int addrlen = sizeof(address);
    // ... (此处代码与上一版完全相同，故省略)
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket failed");
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
    std::cout << "Server listening on port 8080..." << std::endl;
    if ((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen)) < 0) {
        perror("accept");
        exit(EXIT_FAILURE);
    }
    std::cout << "Client connected." << std::endl;


    // --- 3. 循环发送图片和Bbox数据 ---
    int image_index = 0;
    while (true) {
        std::string current_image_path = image_paths[image_index];
        cv::Mat image = cv::imread(current_image_path);
        if (image.empty()) {
            // ... (与之前相同)
            image_index = (image_index + 1) % image_paths.size();
            continue;
        }

        // 将图片编码为 JPEG 格式
        std::vector<uchar> encoded_image;
        cv::imencode(".jpg", image, encoded_image, {cv::IMWRITE_JPEG_QUALITY, 90});
        
        // *** 新增：创建并填充 Header ***
        PacketHeader header;
        
        // 为了演示，我们创建一个动态变化的虚拟Bbox
        // 在实际应用中，这些值应来自您的目标检测算法
        header.bbox_x = 50 + (image_index % 10) * 10;
        header.bbox_y = 50;
        header.bbox_width = 100 + (image_index % 5) * 5;
        header.bbox_height = 100;
        header.image_size = encoded_image.size();

        // *** 修改发送逻辑 ***
        // 1. 发送固定大小的 Header
        if (!send_all(new_socket, &header, sizeof(PacketHeader))) {
            std::cerr << "Failed to send header. Client disconnected." << std::endl;
            break;
        }

        // 2. 发送可变大小的图片数据
        if (!send_all(new_socket, encoded_image.data(), header.image_size)) {
            std::cerr << "Failed to send image data. Client disconnected." << std::endl;
            break;
        }

        std::cout << "Sent " << current_image_path << " (Image: " << header.image_size 
                  << " bytes, Bbox: x=" << header.bbox_x << ")" << std::endl;

        image_index = (image_index + 1) % image_paths.size();
        usleep(100000); // 100ms
    }

    close(new_socket);
    close(server_fd);
    return 0;
}
