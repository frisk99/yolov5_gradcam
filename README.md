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




```python

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.segmentation import fcn_resnet50
from PIL import Image
import numpy as np
from tqdm import tqdm

# 自定义语义分割数据集
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, target_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.image_dir, self.image_files[idx])).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, self.mask_files[idx])).convert("L")  # single-channel

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            mask = self.target_transform(mask)

        return img, mask.long()

# 参数配置
NUM_CLASSES = 21  # 21 类，含背景
BATCH_SIZE = 4
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 图像和mask的转换
image_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet 均值
                         [0.229, 0.224, 0.225])  # ImageNet 方差
])

mask_transform = transforms.Compose([
    transforms.Resize((512, 512), interpolation=Image.NEAREST),
    transforms.PILToTensor(),  # 输出 (1,H,W)
    transforms.Lambda(lambda x: x.squeeze(0))  # 转为 (H, W)
])

# 构建 DataLoader
train_dataset = SegmentationDataset("your_dataset/train/images", "your_dataset/train/masks",
                                    transform=image_transform, target_transform=mask_transform)
val_dataset = SegmentationDataset("your_dataset/val/images", "your_dataset/val/masks",
                                  transform=image_transform, target_transform=mask_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# 加载模型并替换 classifier
model = fcn_resnet50(pretrained=True)

# 冻结主干网络的所有层，保持原来训练好的特征提取能力
for param in model.backbone.parameters():
    param.requires_grad = False

# 只训练 classifier 和 aux_classifier 部分
for param in model.classifier.parameters():
    param.requires_grad = True
for param in model.aux_classifier.parameters():
    param.requires_grad = True

# 优化器
optimizer = optim.Adam([
    {'params': model.classifier.parameters()},
    {'params': model.aux_classifier.parameters()}
], lr=LEARNING_RATE)

# 损失函数（ignore_index=255 是语义分割常规做法）
criterion = nn.CrossEntropyLoss(ignore_index=255)

# mIoU 计算函数
def compute_miou(preds, labels, num_classes, ignore_index=255):
    """
    计算每一类 IoU，并返回 mean IoU（忽略 ignore_index）
    preds, labels: tensor, shape = (N, H, W)
    """
    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    ious = []
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        pred_inds = preds == cls
        label_inds = labels == cls
        intersection = np.logical_and(pred_inds, label_inds).sum()
        union = np.logical_or(pred_inds, label_inds).sum()

        if union == 0:
            ious.append(np.nan)  # 当前 batch 中该类没有出现
        else:
            ious.append(intersection / union)
    # 返回平均 mIoU（忽略 NaN）
    return np.nanmean(ious)

# 训练 & 验证函数
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for imgs, masks in tqdm(loader, desc="Training"):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        output = model(imgs)['out']
        loss = criterion(output, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, num_classes=21):
    model.eval()
    miou_scores = []

    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Validating"):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            output = model(imgs)['out']
            preds = torch.argmax(output, dim=1)

            # 只取有效区域参与 mIoU
            valid = masks != 255
            if valid.sum() == 0:
                continue
            miou = compute_miou(preds[valid], masks[valid], num_classes)
            miou_scores.append(miou)

    return np.nanmean(miou_scores)

# 训练主循环
for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    val_miou = evaluate(model, val_loader)
    print(f"Train Loss: {train_loss:.4f} | Val mIoU: {val_miou:.4f}")

    # 保存模型
    torch.save(model.state_dict(), f"fcn_epoch{epoch+1}.pth")