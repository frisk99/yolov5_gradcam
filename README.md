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




```cpp
#include <stdio.h>
#include <unistd.h>  // 包含 sleep 函数
#include <pthread.h> // 包含 pthread 函数

// 定义回调函数类型
typedef void (*callback_t)(int result);

// 回调函数实现
void on_async_operation_complete(int result) {
    printf("异步操作完成，结果：%d\n", result);
}

// 异步操作线程函数
void* async_operation_thread(void* arg) {
    callback_t callback = (callback_t)arg;

    sleep(2);  // 模拟耗时的异步操作

    int result = 42;  // 假设这是异步操作的结果

    if (callback) {
        callback(result);  // 调用回调函数，传递结果
    }

    return NULL;
}

// 注册异步操作
void async_operation(callback_t callback) {
    pthread_t thread;
    pthread_create(&thread, NULL, async_operation_thread, (void*)callback);
    pthread_detach(thread);  // 分离线程，自动回收
}

int main() {
    printf("开始异步操作...\n");

    // 注册并执行异步操作，将回调函数传递进去
    //async_operation(on_async_operation_complete);
    pthread_t thread;
    pthread_create(&thread, NULL, async_operation_thread, (void*)on_async_operation_complete);
    pthread_detach(thread);  // 分离线程，自动回收

    // 主线程继续做其他事情
    printf("主线程正在执行其他任务...\n");

    // 为了防止主线程结束得太快，可以增加一个 sleep，方便观察输出
    sleep(3);
    return 0;
}

```cpp
#include <stdio.h>
#include <unistd.h>  // 包含 sleep 函数
#include <pthread.h> // 包含 pthread 函数

// 直接处理结果的函数
void handle_result(int result) {
    printf("异步操作完成，结果：%d\n", result);
}

// 异步操作线程函数，直接调用处理函数
void* async_operation_thread(void* arg) {
    sleep(2);  // 模拟耗时的异步操作
    int result = 42;  // 假设这是异步操作的结果

    // 直接处理结果，而不是通过函数指针
    handle_result(result);

    return NULL;
}

int main() {
    pthread_t thread;

    // 创建线程执行异步操作
    pthread_create(&thread, NULL, async_operation_thread, NULL);
    pthread_join(thread, NULL);
    printf("主线程正在执行其他任务...\n");


    return 0;
}

