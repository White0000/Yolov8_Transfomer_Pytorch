import os
import cv2

# 设置数据集路径
train_path = 'E:/PyDev/yolo/.venv/dataset/IIIT5K/train'


# 定义一个函数来检查和修复标签文件
def check_and_fix_labels(image_folder):
    for filename in os.listdir(image_folder):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            # 获取图像的基础名称，不包含扩展名
            base_name = os.path.splitext(filename)[0]
            # 构造图像和标签文件路径
            image_path = os.path.join(image_folder, filename)
            label_file = os.path.join(image_folder, f"{base_name}.txt")

            # 检查图像是否可以成功加载
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法加载图像: {image_path}")
                continue

            # 获取图像的宽度和高度
            height, width = image.shape[:2]

            # 如果标签文件不存在或为空，则创建一个默认标签文件
            if not os.path.exists(label_file) or os.stat(label_file).st_size == 0:
                with open(label_file, 'w') as file:
                    # 写入一个默认的标签内容（例如，class=0，中心在图像中央，宽高为0.1）
                    default_content = "0 0.5 0.5 0.1 0.1\n"
                    file.write(default_content)
                print(f"创建或修正了标签文件: {label_file}，内容为默认标签")
                continue

            # 检查标签文件的内容
            with open(label_file, 'r') as file:
                lines = file.readlines()

            valid_lines = []
            for line in lines:
                parts = line.strip().split()
                # 检查是否包含正确数量的元素
                if len(parts) == 5:
                    class_id, x_center, y_center, w, h = parts
                    try:
                        # 检查数值是否在合理范围内
                        class_id = int(class_id)
                        x_center = float(x_center)
                        y_center = float(y_center)
                        w = float(w)
                        h = float(h)
                        if 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= w <= 1 and 0 <= h <= 1:
                            valid_lines.append(f"{class_id} {x_center} {y_center} {w} {h}")
                        else:
                            print(f"标签值超出范围，在文件: {label_file}")
                    except ValueError:
                        print(f"标签内容格式错误，在文件: {label_file}")
                else:
                    print(f"标签文件格式错误，在文件: {label_file}")

            # 如果修正后的标签不符合要求或为空，则写一个默认的标签内容
            if not valid_lines:
                with open(label_file, 'w') as file:
                    # 写入一个默认的标签内容（例如，class=0，中心在图像中央，宽高为0.1）
                    default_content = "0 0.5 0.5 0.1 0.1\n"
                    file.write(default_content)
                print(f"标签文件无效，已重写为默认内容: {label_file}")
            else:
                # 如果有有效标签，则写回修正后的内容
                with open(label_file, 'w') as file:
                    file.write("\n".join(valid_lines))
                print(f"修正了标签文件: {label_file}")


print("开始检查和修复标签文件...")
check_and_fix_labels(train_path)
print("标签文件检查和修复完成！")
