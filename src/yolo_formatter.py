import os
import re
import logging
from pathlib import Path

from copy_dir import DirCopier

def setup_logging(level):
    logger = logging.getLogger()
    logger.setLevel(level)
    return logger

class YOLOFormatter:
    def __init__(self, classes_file=None, logger=None):
        self.classes = self.read_classes(classes_file=classes_file, logger=logger)
        self.logger = logger or logging.getLogger(__name__)

    def generate_yolo_label(self, class_name, bbox, logger=None):
        """生成YOLO格式标签。

        参数:
        - class_name: 要生成标签的类别名
        - bbox: (class_id, x, y, w, h) 其中后四个值已经是YOLO归一化格式
        返回: 一行 YOLO 格式字符串
        """
        try:
            class_id = self.classes.index(class_name)
        except ValueError:
            if logger:
                logger.error(f"class_name '{class_name}' 不在 classes 列表中")
            raise ValueError(f"class_name '{class_name}' 不在 classes 列表中")
        
        # bbox 中的坐标已经是YOLO格式（归一化），直接使用
        x, y, w, h = bbox
        return f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"

    def extract_class_from_filename(self, filename, logger=None):
        """从文件名中提取类别名，假设类别名是文件名前缀"""
        # 使用正则表达式提取前缀（所有非数字字符）
        match = re.match(r'(\D+)', filename)
        if not match:
            if logger:
                logger.error(f"无法从文件名 '{filename}' 提取类别")
            raise ValueError(f"无法从文件名 '{filename}' 提取类别")

        class_name = match.group(1)

        if class_name in self.classes:
            return class_name
        else:
            if logger:
                logger.error(f"无法从文件名 '{filename}' 提取有效类别，'{class_name}' 不在类别列表中")
            raise ValueError(f"无法从文件名 '{filename}' 提取有效类别，'{class_name}' 不在类别列表中")
        
    def read_classes(self, classes_file=None, logger=None):
        """读取classes.txt，返回类别列表。将原 ClassManager.read_classes 改为普通函数。
        如果提供 logger，会在找不到关键类别时发出警告。
            """
        if classes_file is None:
            classes_file = "../config/classes.txt"
        try:
            with open(classes_file, 'r', encoding='utf-8') as f:
                classes = [line.strip() for line in f if line.strip()]
            
            return classes
        except FileNotFoundError:
            raise FileNotFoundError(f"未找到类别文件: {classes_file}")
        
        
    def start_formatting(self, input_dir, output_dir, logger=None):
        """开始格式化YOLO标签文件。
        
        标签格式: class_id x y w h (都是YOLO格式，第一个是类别索引，后四个是归一化坐标)
        """
        if logger is None:
            logger = logging.getLogger(__name__)

        for label in os.listdir(input_dir):
            if not label.endswith('.txt'):
                continue
            
            label_name = self.extract_class_from_filename(label, logger=logger)
            self.logger.debug(f"Processing label file: {label} for class: {label_name}")
            
            input_path = os.path.join(input_dir, label)
            output_path = os.path.join(output_dir, label)
            
            with open(input_path, 'r', encoding='utf-8') as f_in, \
                 open(output_path, 'w', encoding='utf-8') as f_out:
                
                for line in f_in:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        logger.warning(f"标签文件 {label} 中的行格式不正确: {line.strip()}")
                        continue
                    
                    # 第一个是类别索引，后四个是x, y, w, h（已经是YOLO格式）
                    class_id, x, y, w, h = map(float, parts)
                    bbox = (x, y, w, h)
                    
                    label_name = self.extract_class_from_filename(label, logger=logger)
                    
                    yolo_line = self.generate_yolo_label(label_name, bbox, logger=logger)
                    f_out.write(yolo_line + "\n")


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    path = Path(__file__).parent.parent 
    
    classes_file = os.path.join(path, 'config', 'classes.txt')

    yolo_formatter = YOLOFormatter(classes_file=classes_file, logger=logger)
    
    #完全copy original database labels to new location
    input_file_path = os.path.join(path, 'dataset')
    output_file_path = os.path.join(path,  'dataset_yolo')
    dir_copier = DirCopier(input_file_path, output_file_path, logger=logger)
    dir_copier.copy_directory()

    input_file_path = os.path.join(path, 'dataset', 'images', 'train')
    output_file_path = os.path.join(path, 'dataset_yolo', 'images', 'train')

    dir_copier = DirCopier(input_file_path, output_file_path, logger=logger)
    dir_copier.copy_directory_with_symlinks()

    input_dir = os.path.join(path, 'dataset', 'labels', 'train')
    output_dir = os.path.join(path, 'dataset_yolo', 'labels', 'train')
    
    yolo_formatter.start_formatting(input_dir, output_dir, logger=logger)

if __name__ == "__main__":
    main()


