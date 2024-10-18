import sys
import threading
import time
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QTextEdit, QFileDialog, QVBoxLayout, \
    QProgressBar, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import pyqtSignal, QObject
from paddleocr import PaddleOCR
import os
from datetime import datetime
import matplotlib.pyplot as plt


torch_available = False
try:
    import torch
    from ultralytics import YOLO
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    torch_available = True
    print("CUDA 是否可用:", torch.cuda.is_available())
    print("CUDA 版本:", torch.version.cuda)
    print("当前设备名称:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "没有检测到 GPU")
except (OSError, ImportError) as e:
    print(f"PyTorch 加载错误: {e}")
    torch_available = False

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 初始化 PaddleOCR，用于中英文识别
ocr = PaddleOCR(use_angle_cls=True, lang="ch")

# 初始化 TrOCR
trocr_processor = None
trocr_model = None
if torch_available:
    try:
        trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        trocr_model.to("cuda" if torch.cuda.is_available() else "cpu")
    except Exception as e:
        print(f"TrOCR 模型加载错误: {e}")

# 定义信号类，用于线程间通信
class WorkerSignals(QObject):
    result = pyqtSignal(str)
    progress = pyqtSignal(int)
    console_output = pyqtSignal(str)
    error = pyqtSignal(str)

class TextRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.yolo_model_path = 'v8model/yolov8n.pt'  # 替换YOLO模型路径
        self.signals = WorkerSignals()
        self.signals.result.connect(self.show_result)
        self.signals.progress.connect(self.update_progress)
        self.signals.console_output.connect(self.show_console_output)
        self.signals.error.connect(self.show_error_dialog)

        self.image_path = None
        self.image = None
        self.selected_roi = None
        self.recognized_text = ""

    def initUI(self):
        self.setWindowTitle("图片文字识别系统")
        self.label = QLabel(self)
        self.label.setText("请选择一张图片")

        self.button = QPushButton('选择图片', self)
        self.button.clicked.connect(self.open_image)

        self.grayscale_button = QPushButton('灰度化', self)
        self.grayscale_button.clicked.connect(self.apply_grayscale)
        self.grayscale_button.setEnabled(False)

        self.binarize_button = QPushButton('二值化', self)
        self.binarize_button.clicked.connect(self.apply_binarization)
        self.binarize_button.setEnabled(False)

        self.denoise_button = QPushButton('去噪', self)
        self.denoise_button.clicked.connect(self.apply_denoise)
        self.denoise_button.setEnabled(False)

        self.select_area_button = QPushButton('选择识别区域', self)
        self.select_area_button.clicked.connect(self.select_area)
        self.select_area_button.setEnabled(False)

        self.train_button = QPushButton('训练模型', self)
        self.train_button.clicked.connect(self.train_model)

        self.test_button = QPushButton('测试模型', self)
        self.test_button.clicked.connect(self.test_model)

        self.process_button = QPushButton('开始识别', self)
        self.process_button.clicked.connect(self.start_recognition)
        self.process_button.setEnabled(False)

        self.result_text = QTextEdit(self)
        self.console_text = QTextEdit(self)
        self.console_text.setReadOnly(True)

        self.export_button = QPushButton('导出为TXT', self)
        self.export_button.clicked.connect(self.export_result)
        self.export_button.setEnabled(False)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.button)
        layout.addWidget(self.grayscale_button)
        layout.addWidget(self.binarize_button)
        layout.addWidget(self.denoise_button)
        layout.addWidget(self.select_area_button)
        layout.addWidget(self.train_button)
        layout.addWidget(self.test_button)
        layout.addWidget(self.process_button)
        layout.addWidget(self.result_text)
        layout.addWidget(self.console_text)
        layout.addWidget(self.export_button)
        layout.addWidget(self.progress_bar)
        self.setLayout(layout)

    def open_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "选择图片", "",
                                                   "Image Files (*.png *.jpg *.bmp);;All Files (*)", options=options)
        if file_name:
            self.image_path = file_name
            self.image = cv2.imread(file_name)
            self.original_image = self.image.copy()
            if self.image is not None:
                self.display_image(self.image)
                self.grayscale_button.setEnabled(True)
            else:
                self.result_text.setText("无法读取图片，请选择有效的图片文件")

    def display_image(self, img):
        if img is not None:
            if len(img.shape) == 3:
                height, width, channel = img.shape
                bytes_per_line = 3 * width
                qImg = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            elif len(img.shape) == 2:
                height, width = img.shape
                bytes_per_line = width
                qImg = QImage(img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            else:
                self.show_error_dialog("无法显示图像，图像通道数不正确")
                return
            self.label.setPixmap(QPixmap.fromImage(qImg))

    def apply_grayscale(self):
        if self.image is not None:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.display_image(self.image)
            self.grayscale_button.setEnabled(False)
            self.binarize_button.setEnabled(True)

    def apply_binarization(self):
        if self.image is not None:
            self.image = cv2.adaptiveThreshold(self.image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY, 11, 2)
            self.display_image(self.image)
            self.binarize_button.setEnabled(False)
            self.denoise_button.setEnabled(True)

    def apply_denoise(self):
        if self.image is not None:
            self.image = cv2.fastNlMeansDenoising(self.image, None, 30, 7, 21)
            self.display_image(self.image)
            self.denoise_button.setEnabled(False)
            self.select_area_button.setEnabled(True)

    def select_area(self):
        if self.image is not None:
            roi = cv2.selectROI("选择识别区域", self.image, showCrosshair=True)
            x, y, w, h = roi
            if w > 0 and h > 0:
                self.selected_roi = self.image[y:y+h, x:x+w]
                self.display_image_with_rectangle(self.image, (x, y, x+w, y+h), color=(0, 0, 255))
                self.process_button.setEnabled(True)
            else:
                self.show_error_dialog("选择的区域无效，请重新选择一个有效的区域。")
            cv2.destroyAllWindows()

    def display_image_with_rectangle(self, img, rect, color):
        x1, y1, x2, y2 = rect
        img_copy = img.copy()
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        self.display_image(img_copy)

    def train_model(self):
        # 检查 PyTorch 是否可用，如果不可用则显示错误对话框并退出函数
        if not torch_available:
            self.show_error_dialog("PyTorch 未正确加载，无法训练模型。")
            return

        # 在新线程中启动训练过程，以保持用户界面的响应性
        train_thread = threading.Thread(target=self.train_yolo_model)
        train_thread.start()

    def train_yolo_model(self):
        try:
            # 使用指定的路径加载 YOLO 模型
            model = YOLO(self.yolo_model_path)
            self.signals.console_output.emit("开始训练模型...\n")

            # 使用给定的参数训练模型
            results = model.train(
                data='data.yaml',  # 数据集配置文件的路径
                epochs=50,  # 训练的轮次
                batch=16,  # 训练的批大小
                imgsz=640,  # 模型使用的图像大小
                device=0 if torch.cuda.is_available() else 'cpu'  # 如果可用则使用 GPU，否则使用 CPU
            )

            # 保存训练好的模型权重
            model.save(self.yolo_model_path)
            self.signals.console_output.emit("训练完成！模型已保存！\n")
            self.signals.progress.emit(100)

            # 绘制训练结果，如准确率和损失
            self.plot_training_results(results)
        except Exception as e:
            # 如果训练过程中发生错误，发送错误信息
            self.signals.error.emit(f"训练时出现错误: {str(e)}")

    def test_model(self):
        # 检查 PyTorch 是否可用，如果不可用则显示错误对话框并退出函数
        if not torch_available:
            self.show_error_dialog("PyTorch 未正确加载，无法测试模型。")
            return

        # 在新线程中启动测试过程，以保持用户界面的响应性
        test_thread = threading.Thread(target=self.test_yolo_model)
        test_thread.start()

    def test_yolo_model(self):
        try:
            # 使用指定的路径加载 YOLO 模型
            model = YOLO(self.yolo_model_path)

            # 验证模型并收集结果
            results = model.val()
            self.signals.console_output.emit(f"测试完成！\n{results}")
        except Exception as e:
            # 如果测试过程中发生错误，发送错误信息
            self.signals.error.emit(f"测试时出现错误: {str(e)}")

    def plot_training_results(self, results):
        # 绘制训练损失和验证准确率随轮次的变化图
        epochs = range(len(results['train_loss']))
        plt.figure()
        plt.plot(epochs, results['train_loss'], 'r', label='Training loss')
        plt.plot(epochs, results['val_accuracy'], 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy/loss')
        plt.xlabel('Epochs')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig('training_performance.png')
        plt.show()

    def start_recognition(self):
        # 在新线程中启动图像识别过程，以保持用户界面的响应性
        recognition_thread = threading.Thread(target=self.process_image)
        recognition_thread.start()

    def process_image(self):
        if self.selected_roi is not None:
            start_time = time.time()
            self.signals.progress.emit(10)  # 更新进度条

            try:
                # 使用 PaddleOCR 识别选定区域的文本
                result_text = self.recognize_text_with_paddleocr(self.selected_roi)
                self.recognized_text = result_text
                elapsed_time = time.time() - start_time
                self.signals.console_output.emit(f"识别完成！用时: {elapsed_time:.2f}秒\n")
                self.signals.result.emit("识别完成！")
                self.signals.progress.emit(100)  # 将进度设置为完成
                self.export_button.setEnabled(True)  # 识别完成后启用导出按钮
            except Exception as e:
                # 如果识别过程中发生错误，发送错误信息
                self.signals.error.emit(f"识别时出现错误: {str(e)}")

    def recognize_text_with_paddleocr(self, image):
        # 使用 PaddleOCR 识别给定图像中的文本
        result = ocr.ocr(image, cls=True)
        text = ''
        for line in result:
            for word_info in line:
                text += word_info[1][0] + ' '  # 追加识别的单词
        return text.strip()

    def recognize_text_with_transformer(self, image):
        if trocr_processor and trocr_model:
            # 将图像转换为张量并移动到适当的设备（GPU/CPU）
            image_tensor = torch.tensor(image).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")

            # 使用 transformer 模型从图像生成文本
            generated_ids = trocr_model.generate(image_tensor)
            generated_text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return generated_text
        return "TrOCR 模型不可用"

    def export_result(self):
        if self.recognized_text.strip():
            current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_name, _ = QFileDialog.getSaveFileName(self, "导出为TXT", f"识别结果_{current_time}.txt", "Text Files (*.txt);;All Files (*)")
            if file_name:
                try:
                    with open(file_name, 'w', encoding='utf-8') as file:
                        file.write(self.recognized_text)
                    self.signals.console_output.emit("导出成功！")
                except Exception as e:
                    self.show_error_dialog(f"导出时出现错误: {str(e)}")
        else:
            self.show_error_dialog("没有识别结果可以导出")

    def show_result(self, result):
        self.result_text.setText(result)

    def show_console_output(self, message):
        self.console_text.append(message)

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def show_error_dialog(self, message):
        error_dialog = QMessageBox()
        error_dialog.setIcon(QMessageBox.Critical)
        error_dialog.setWindowTitle("错误")
        error_dialog.setText(message)
        error_dialog.exec_()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TextRecognitionApp()
    window.show()
    sys.exit(app.exec_())
