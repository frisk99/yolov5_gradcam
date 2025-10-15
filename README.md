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
// socket_server_no_json.cpp
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <thread>
#include <mutex>
#include <cstring>

// For socket programming
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h> // for close()
#include <arpa/inet.h>

#define PORT 8080
#define BUFFER_SIZE 4096

// A mutex for thread-safe cout
std::mutex cout_mutex;

// Forward declarations
void adduserinfo(const std::vector<std::string>& args);
void deleteusrinfo(const std::vector<std::string>& args);
std::string subscribeimage();
void unsubscribeimage();

// Client handler function
void handle_client(int client_socket) {
    char buffer[BUFFER_SIZE] = {0};
    
    const char* welcome_msg = "Connection to server successful.\n";
    send(client_socket, welcome_msg, strlen(welcome_msg), 0);

    while (true) {
        memset(buffer, 0, BUFFER_SIZE);
        int bytes_received = recv(client_socket, buffer, BUFFER_SIZE, 0);

        if (bytes_received <= 0) {
            {
                std::lock_guard<std::mutex> lock(cout_mutex);
                std::cout << "Client disconnected. Socket fd: " << client_socket << std::endl;
            }
            close(client_socket);
            return;
        }
        
        std::string received_data(buffer, bytes_received);
        std::stringstream ss(received_data);
        std::string command;
        ss >> command;

        std::vector<std::string> args;
        std::string arg;
        while (ss >> arg) {
            args.push_back(arg);
        }

        {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "Received command from client " << client_socket << ": " << command << std::endl;
        }

        std::string response = "OK\n";

        if (command == "adduserinfo") {
            if (args.size() == 3) {
                adduserinfo(args);
            } else {
                response = "ERROR: adduserinfo requires 3 arguments (name, image, id).\n";
            }
        } else if (command == "deleteusrinfo") {
            if (args.size() == 2) {
                deleteusrinfo(args);
            } else {
                response = "ERROR: deleteusrinfo requires 2 arguments (name, id).\n";
            }
        } else if (command == "subscribeimage") {
            response = subscribeimage() + "\n";
        } else if (command == "unsubscribeimage") {
            unsubscribeimage();
        } else {
            response = "ERROR: Unknown command '" + command + "'\n";
        }

        send(client_socket, response.c_str(), response.length(), 0);
    }
}

// --- Function Implementations ---

void adduserinfo(const std::vector<std::string>& args) {
    std::string name = args[0];
    std::string image = args[1];
    int id = std::stoi(args[2]);
    
    std::lock_guard<std::mutex> lock(cout_mutex);
    std::cout << "[INFO] Adding user. Name: " << name << ", Image: " << image << ", ID: " << id << std::endl;
}

void deleteusrinfo(const std::vector<std::string>& args) {
    std::string name = args[0];
    int id = std::stoi(args[1]);

    std::lock_guard<std::mutex> lock(cout_mutex);
    std::cout << "[INFO] Deleting user. Name: " << name << ", ID: " << id << std::endl;
}

/**
 * @brief Manually constructs a JSON string with image data.
 * This function creates a JSON object without any external libraries.
 * @return A std::string containing the JSON data.
 */
std::string subscribeimage() {
    std::lock_guard<std::mutex> lock(cout_mutex);
    std::cout << "[INFO] Client subscribed to image stream. Sending data." << std::endl;
    
    // Data to be sent
    std::vector<int> ids = {101, 102, 103};
    std::vector<std::string> names = {"person_A", "car_B", "person_C"};
    std::vector<int> bboxes = {10, 20, 50, 60, 100, 120, 80, 40, 200, 210, 75, 75};

    std::stringstream json_ss;
    
    json_ss << "{";
    
    // 1. Add integer array "ids"
    json_ss << "\"ids\":[";
    for (size_t i = 0; i < ids.size(); ++i) {
        json_ss << ids[i];
        if (i < ids.size() - 1) {
            json_ss << ",";
        }
    }
    json_ss << "],";
    
    // 2. Add string array "names"
    json_ss << "\"names\":[";
    for (size_t i = 0; i < names.size(); ++i) {
        json_ss << "\"" << names[i] << "\""; // Strings in JSON must be in double quotes
        if (i < names.size() - 1) {
            json_ss << ",";
        }
    }
    json_ss << "],";
    
    // 3. Add integer array "bboxes"
    json_ss << "\"bboxes\":[";
    for (size_t i = 0; i < bboxes.size(); ++i) {
        json_ss << bboxes[i];
        if (i < bboxes.size() - 1) {
            json_ss << ",";
        }
    }
    json_ss << "]";
    
    json_ss << "}";
    
    return json_ss.str();
}

void unsubscribeimage() {
    std::lock_guard<std::mutex> lock(cout_mutex);
    std::cout << "[INFO] Client unsubscribed from image stream." << std::endl;
}


int main() {
    int server_fd;
    struct sockaddr_in address;
    int opt = 1;
    socklen_t addrlen = sizeof(address);

    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }

    if (listen(server_fd, 3) < 0) {
        perror("listen");
        exit(EXIT_FAILURE);
    }

    std::cout << "Server listening on port " << PORT << std::endl;

    while (true) {
        int new_socket = accept(server_fd, (struct sockaddr *)&address, &addrlen);
        if (new_socket < 0) {
            perror("accept");
            continue;
        }
        
        {
            std::lock_guard<std::mutex> lock(cout_mutex);
            char client_ip[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, &address.sin_addr, client_ip, INET_ADDRSTRLEN);
            std::cout << "New connection from " << client_ip << ", socket fd: " << new_socket << std::endl;
        }

        std::thread client_thread(handle_client, new_socket);
        client_thread.detach();
    }

    close(server_fd);
    return 0;
}

