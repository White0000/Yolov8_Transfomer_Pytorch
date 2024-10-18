import torch
from ultralytics import YOLO

# 加载YOLOv8预训练模型
model = YOLO('yolov8n.pt')  # 你也可以选择yolov8s.pt或者更大的模型

# 开始训练
results = model.train(
    data='data.yaml',  # 指向data.yaml配置文件
    epochs=100,        # 训练轮数
    batch_size=16,     # 批次大小（根据显存可调）
    imgsz=640,         # 输入图像的大小
    device=0,          # 使用GPU进行训练
    workers=4          # 数据加载的CPU线程数
)

# 验证模型性能
results = model.val()

# 保存最佳模型权重
model.save('best_yolov8_weights.pt')

# 导出模型为ONNX格式（可用于推理加速）
model.export(format='onnx')
