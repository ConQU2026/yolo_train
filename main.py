import cv2
import numpy as np
import os
import random
from pathlib import Path
import logging

from config_parser import ParseConfig

def setup_logging(level):
    logger = logging.getLogger()
    logger.setLevel(level)
    return logger


class ClassManager:
    def __init__(self, classes_file, logger):
        self.classes = self.read_classes(classes_file)

    def read_classes(self, classes_file=CLASSES_FILE):
        """读取classes.txt，返回类别列表"""
        try:
            with open(classes_file, 'r', encoding='utf-8') as f:
                classes = [line.strip() for line in f if line.strip()]
            # 检查关键类别是否存在
            for key_cls in CLASSES_COUNT.keys():
                if key_cls not in classes:
                    self.logger.warning(f"classes.txt中未找到 {key_cls}，请确认文件内容")
            return classes
        except FileNotFoundError:
            raise FileNotFoundError(f"未找到类别文件: {classes_file}")
    
class DatasetGenerator:
    def __init__(self, classes):
        self.classes = classes

    def generate_yolo_label(self, class_name, bbox, background_shape):
        """生成YOLO格式标签"""
        class_id = classes.index(class_name)
        bg_h, bg_w = background_shape[:2]
        x, y, w, h = bbox
        
        center_x = max(0, min(1, (x + w/2) / bg_w))
        center_y = max(0, min(1, (y + h/2) / bg_h))
        norm_w = max(0, min(1, w / bg_w))
        norm_h = max(0, min(1, h / bg_h))
        
        return f"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}"



def main():
    
    logger = setup_logging(logging.INFO)
    
    # 读取类别并生成配置
    classes_manager = ClassManager()
    classes = classes_manager.classes
    CLASSES_COUNT = {cls: DATA_NUMBERS for cls in classes}
    
    total_samples = sum(CLASSES_COUNT.values())
    br_sum = sum(br_ratio)
    if br_sum == 0:
        raise ValueError("B/R比例不能全为0，请调整br_ratio参数")
    b_total = (total_samples * br_ratio[0]) // br_sum
    r_total = total_samples - b_total
    
    # 打印任务信息（含模糊配置）
    logger.info(f"=== 任务配置 ===")
    logger.info(f"总样本数: {total_samples} | B文件夹: {b_total}个 | R文件夹: {r_total}个")
    logger.info(f"背景尺寸: {BACKGROUND_SIZE[0]}×{BACKGROUND_SIZE[1]} | 旋转角度: {ROTATION_ANGLE_RANGE}度")
    logger.info(f"仿射扭曲幅度: {AFFINE_DISTORTION_RANGE} | 最小缩放比例: {SCALE_MIN}")
    logger.info(f"模糊配置: 触发概率{BLUR_PROBABILITY*100}% | 核大小范围{BLUR_KERNEL_RANGE}（奇数）")
    logger.info(f"输出路径: {os.path.abspath(OUTPUT_DIR)}")
    logger.info(f"===============")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    generated = {cls: 0 for cls in CLASSES_COUNT}
    b_generated = 0
    r_generated = 0
    sample_idx = 0
    processor = ImageProcessor()
    dataset_gen = DatasetGenerator(classes)
    
    while sum(generated.values()) < total_samples:
        try:
            
            # 选择B/R文件夹
            if (b_generated < b_total) and (r_generated >= r_total or random.random() < br_ratio[0]/br_sum):
                selected_folder = os.path.join(KFS_FOLDER, "B")
                folder_type = "B"
            else:
                selected_folder = os.path.join(KFS_FOLDER, "R")
                folder_type = "R"
            
            # 提取类别
            filename = os.path.basename(foreground_path)
            class_name = ClassManager.extract_class_from_filename(filename, classes)
            if generated[class_name] >= classes_count[class_name]:
                continue
            
            # 图像变换（含模糊）
            try:
                transformed_fg = processor.random_transform(foreground)
            except Exception as e:
                logger.error(f"图像变换失败 {filename}: {str(e)}，跳过")
                continue
            
            
            # 保存结果
            output_img = os.path.join(OUTPUT_DIR, f"sample_{sample_idx}.jpg")
            output_txt = os.path.join(OUTPUT_DIR, f"sample_{sample_idx}.txt")
            cv2.imwrite(output_img, result)
            yolo_label = dataset_gen.generate_yolo_label(class_name, bbox, background.shape)
            with open(output_txt, 'w', encoding='utf-8') as f:
                f.write(yolo_label + "\n")
            
            # 更新计数与进度
            generated[class_name] += 1
            if folder_type == "B":
                b_generated += 1
            else:
                r_generated += 1
            sample_idx += 1
            
            progress = sum(generated.values()) / total_samples * 100
            logger.info(f"进度: {progress:.1f}% | 已生成 {sample_idx}/{total_samples} 个 | 类别: {class_name} | 来源: {folder_type}")
        
        except Exception as e:
            logger.error(f"处理出错: {str(e)}，继续尝试下一个样本...")
    
    logger.info(f"所有样本生成完成！共生成 {sample_idx} 个样本，保存于: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()