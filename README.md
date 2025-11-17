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
#include <unistd.h>
#include <sys/wait.h>
#include <cstring>
#include <cerrno>
#include <stdexcept>
#include <optional>  // <<< 1. 包含 optional
#include <chrono>    // for main
#include <thread>    // for main
#include <memory>

class PipeUtils {
public:
    // 确保写入所有数据
    static bool send_data(int fd, const void* data, size_t size) {
        const char* ptr = static_cast<const char*>(data);
        size_t total = 0;
        while (total < size) {
            ssize_t written = write(fd, ptr + total, size - total);
            if (written < 0) {
                if (errno == EINTR) continue;
                if (errno == EPIPE) {
                    std::cerr << "Pipe write error: Broken pipe (EPIPE)" << std::endl;
                } else {
                    perror("Pipe write error");
                }
                return false;
            }
            total += written;
        }
        return true;
    }

    // 确保读取所有数据
    static ssize_t recv_data(int fd, void* data, size_t size) {
        char* ptr = static_cast<char*>(data);
        size_t total = 0;
        while (total < size) {
            ssize_t read_bytes = read(fd, ptr + total, size - total);
            if (read_bytes < 0) {
                if (errno == EINTR) continue;
                perror("Pipe read error");
                return -1; // -1 表示错误
            }
            if (read_bytes == 0) {
                if (total == 0) {
                    return 0; // 0 表示 EOF
                }
                std::cerr << "Pipe closed unexpectedly" << std::endl;
                return -1;
            }
            total += read_bytes;
        }
        return total; // 返回读取的字节数
    }
};


// --- 核心工作类 ---
class ImageProcessor {
public:
    ImageProcessor() : child_pid_(-1), fd_write_to_child_(-1), fd_read_from_child_(-1) {
        // 构造函数现在只调用 start_worker
        if (!start_worker()) {
            throw std::runtime_error("Failed to start initial worker process");
        }
    }

    ~ImageProcessor() {
        // 析构函数现在只调用 stop_worker
        stop_worker();
    }

    std::optional<std::vector<float>> process(const std::vector<unsigned char>& image) {
        
        // 尝试 2 次：1 次正常，1 次在重启后
        const int MAX_ATTEMPTS = 2; 
        for (int attempt = 1; attempt <= MAX_ATTEMPTS; ++attempt) {
            
            // 1. 发送数据给子进程
            if (!send_image(fd_write_to_child_, image)) {
                std::cerr << "[Parent] Send failed on attempt " << attempt << std::endl;
                if (attempt == MAX_ATTEMPTS) return std::nullopt; // 最终失败
                if (!restart_worker()) return std::nullopt; // 重启失败
                continue; // 重试
            }

            // 2. 读取子进程结果
            std::optional<std::vector<float>> results = recv_results(fd_read_from_child_);
            if (!results.has_value()) {
                std::cerr << "[Parent] Recv failed on attempt " << attempt << std::endl;
                if (attempt == MAX_ATTEMPTS) return std::nullopt; // 最终失败
                if (!restart_worker()) return std::nullopt; // 重启失败
                continue; // 重试 (注意：重试会重新发送数据)
            }

            return results; // 成功！
        }
        
        return std::nullopt; // 理论上不会到这里
    }

private:
    pid_t child_pid_;
    int fd_write_to_child_;
    int fd_read_from_child_;

    // --- 提取出的工作者启动逻辑 ---
    bool start_worker() {
        int pipe_p2c[2]; // Parent -> Child
        int pipe_c2p[2]; // Child -> Parent

        if (pipe(pipe_p2c) == -1 || pipe(pipe_c2p) == -1) {
            perror("pipe failed");
            return false;
        }

        child_pid_ = fork();

        if (child_pid_ == -1) {
            perror("fork failed");
            return false;
        }

        if (child_pid_ == 0) {
            // === 子进程逻辑 ===
            close(pipe_p2c[1]); // P->C 关写
            close(pipe_c2p[0]); // C->P 关读
            run_child_loop(pipe_p2c[0], pipe_c2p[1]);
            close(pipe_p2c[0]);
            close(pipe_c2p[1]);
            exit(0); 
        } else {
            // === 父进程逻辑 ===
            close(pipe_p2c[0]); // P->C 关读
            close(pipe_c2p[1]); // C->P 关写
            fd_write_to_child_ = pipe_p2c[1];
            fd_read_from_child_ = pipe_c2p[0];
            std::cout << "[Parent] Worker process started with PID: " << child_pid_ << std::endl;
            return true;
        }
    }

    // --- 提取出的工作者停止逻辑 ---
    void stop_worker() {
        if (child_pid_ <= 0) return; // 已经停止

        std::cout << "[Parent] Shutting down worker PID: " << child_pid_ << std::endl;
        
        // 1. 关闭管道，通知子进程退出
        close(fd_write_to_child_);
        close(fd_read_from_child_);

        // 2. 等待子进程退出
        waitpid(child_pid_, nullptr, 0);
        
        // 3. 重置状态
        child_pid_ = -1;
        fd_write_to_child_ = -1;
        fd_read_from_child_ = -1;
    }

    // --- 新的重启函数 ---
    bool restart_worker() {
        std::cerr << "[Parent] --- RESTARTING WORKER ---" << std::endl;
        stop_worker(); // 清理旧的
        return start_worker(); // 启动新的
    }

    // 父进程辅助：发送图片 (不变)
    bool send_image(int fd, const std::vector<unsigned char>& img) {
        uint64_t size = img.size();
        if (!PipeUtils::send_data(fd, &size, sizeof(size))) return false;
        if (size > 0) {
            if (!PipeUtils::send_data(fd, img.data(), size)) return false;
        }
        return true;
    }

    // 父进程辅助：接收结果 (修改为返回 optional)
    std::optional<std::vector<float>> recv_results(int fd) {
        uint32_t count = 0;
        // <= 0 同时捕获 错误(-1) 和 EOF(0)
        if (PipeUtils::recv_data(fd, &count, sizeof(count)) <= 0) {
            std::cerr << "[Parent] Failed to read result count." << std::endl;
            return std::nullopt;
        }
        
        std::vector<float> results(count);
        if (count > 0) {
            if (PipeUtils::recv_data(fd, results.data(), count * sizeof(float)) <= 0) {
                std::cerr << "[Parent] Failed to read result data." << std::endl;
                return std::nullopt;
            }
        }
        return results; // 成功
    }

    // 子进程工作循环 (不变)
    void run_child_loop(int fd_read, int fd_write) {
        while (true) {
            uint64_t img_size = 0;
            ssize_t read_ret = PipeUtils::recv_data(fd_read, &img_size, sizeof(img_size));

            if (read_ret == 0) {
                std::cout << "[Child] Parent closed pipe. Exiting loop." << std::endl;
                break; 
            }
            if (read_ret < 0) {
                std::cerr << "[Child] Pipe read error. Exiting." << std::endl;
                break;
            }

            // (此处省略读取图像、处理、发送结果的代码，与上一版相同)
            // 2. 读取图像数据
            std::vector<unsigned char> img(img_size);
            if (img_size > 0) {
                if (PipeUtils::recv_data(fd_read, img.data(), img_size) < 0) {
                    std::cerr << "[Child] Pipe read (data) error. Exiting." << std::endl;
                    break;
                }
            }
            
            // 3. >>> 真正的数据处理算法在这里 <<<
            std::vector<float> result = internal_algorithm(img);

            // 4. 发送结果
            uint32_t res_count = result.size();
            if (!PipeUtils::send_data(fd_write, &res_count, sizeof(res_count))) {
                 std::cerr << "[Child] Pipe write error. Exiting." << std::endl;
                 break;
            }
            if (res_count > 0) {
                if (!PipeUtils::send_data(fd_write, result.data(), res_count * sizeof(float))) {
                    std::cerr << "[Child] Pipe write (data) error. Exiting." << std::endl;
                    break;
                }
            }
        } 
    }

    // 模拟图像处理算法 (不变)
    std::vector<float> internal_algorithm(const std::vector<unsigned char>& img) {
        // ... (模拟代码) ...
        std::vector<float> res;
        res.push_back(10.5f);
        res.push_back(20.5f);
        res.push_back(static_cast<float>(img.size()));
        return res;
    }
};