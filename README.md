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
```python
import os
import shutil

def copy_matching_files(src_dir_a, src_dir_b, dest_dir_c):
    """
    Copy files from source directory A to destination directory C if they have the same name as files in source directory B.

    Parameters:
    src_dir_a (str): The source directory A containing files to be checked and copied.
    src_dir_b (str): The source directory B containing files to match.
    dest_dir_c (str): The destination directory C to copy matched files to.
    """
    # Get list of files in source directory B
    files_b = os.listdir(src_dir_b)

    # Ensure the destination directory exists
    os.makedirs(dest_dir_c, exist_ok=True)

    # Copy matching files from source directory A to destination directory C
    for file in files_b:
        src_path = os.path.join(src_dir_a, file)
        dest_path = os.path.join(dest_dir_c, file)
        # Check if the file exists in source directory A
        if os.path.isfile(src_path):
            shutil.copy(src_path, dest_path)

    print(f"Copied matching files from {src_dir_a} to {dest_dir_c}")

# Example usage:
source_directory_a = 'path_to_source_directory_a'
source_directory_b = 'path_to_source_directory_b'
destination_directory_c = 'path_to_destination_directory_c'
copy_matching_files(source_directory_a, source_directory_b, destination_directory_c)