import os
import shutil
from sklearn.model_selection import train_test_split

# 数据集路径
image_dir = 'datasets/images/all'
label_dir = 'datasets/labels/all'

# 将数据集划分为训练集和验证集
images = os.listdir(image_dir)
train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)

# 创建必要的目录
os.makedirs('datasets/images/train', exist_ok=True)
os.makedirs('datasets/images/val', exist_ok=True)
os.makedirs('datasets/labels/train', exist_ok=True)
os.makedirs('datasets/labels/val', exist_ok=True)

# 移动图像和标签到对应目录
for image in train_images:
    shutil.move(os.path.join(image_dir, image), 'datasets/images/train/')
    shutil.move(os.path.join(label_dir, image.replace('.jpg', '.txt')), 'datasets/labels/train/')

for image in val_images:
    shutil.move(os.path.join(image_dir, image), 'datasets/images/val/')
    shutil.move(os.path.join(label_dir, image.replace('.jpg', '.txt')), 'datasets/labels/val/')
