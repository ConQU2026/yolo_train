from image_processor import ImageProcessor
import os
from pathlib import Path
from copy_dir import DirCopier
import logging
from parse_config import ParseConfig
import cv2
import tqdm
import sys
import random

sys.path.append(os.path.join(Path(__file__).parent.resolve().parent, 'src'))


def setup_logging(level):
    logger = logging.getLogger()
    logger.setLevel(level)
    return logger   

def main():
    path = Path(__file__).parent.resolve()
    logger = setup_logging(logging.INFO)
    
    config  = ParseConfig(os.path.join(path.parent, 'config', 'config.yaml')).config
    blur_probability = config.get('BLUR_PROBABILITY', 0.5) 
    kernel_range = config.get('KERNEL_RANGE', (5, 45))
    img_size = config.get('IMG_SIZE', (640, 640))
    scale_min = config.get('SCALE_MIN', 0.3)
    rotation_angle_range = config.get('ROTATION_ANGLE_RANGE', (-60, 60))
    perspective_range = config.get('PERSPECTIVE_RANGE', 0.1) 
    
    
    config_params = {
        'BLUR_PROBABILITY': blur_probability,
        'KERNEL_RANGE': kernel_range,
        'IMG_SIZE': img_size,
        'SCALE_MIN': scale_min,
        'ROTATION_ANGLE_RANGE': rotation_angle_range,
        'PERSPECTIVE_RANGE': perspective_range
    }
    
    
    src_directory = os.path.join(path.parent, 'dataset')
    dst_directory = os.path.join(path.parent, 'transformed_dataset')

    if not os.path.exists(src_directory):
        logger.error(f"Source directory does not exist: {src_directory}")
        return
    
    copier = DirCopier(src_path=src_directory, dst_path=dst_directory, logger=logger)

    #完全复制文件夹，不对原数据集进行修改
    copier.copy_directory()
    logger.info("Dataset copied successfully.")
    logger.info("Starting image transformations...")
    
    """
    - database
        - images
            - train
        - labels
            - train
    """
    iamges_dir = "images/train"
    labels_dir = "labels/train"
    
    for root, dirs, files in os.walk(os.path.join(dst_directory, iamges_dir)):
        
        # 处理每个图像文件
        for file in tqdm.tqdm(files, desc="Processing images"):
            if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                continue
            
            image_path = os.path.join(root, file)
            relative_path = os.path.relpath(image_path, os.path.join(dst_directory, iamges_dir))
            label_path = os.path.join(dst_directory, labels_dir, os.path.splitext(relative_path)[0] + '.txt')
            
            # tqdm.tqdm.write(f"Processing image: {image_path} with label: {label_path}")
            logger.debug(f"Processing image: {image_path} with label: {label_path}")
            
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                tqdm.tqdm.write(f"无法读取图像: {image_path}")
                logger.error(f"无法读取图像: {image_path}")
                continue
            
            processor = ImageProcessor(image, **config_params)
            
            transformed_image, M_final, output_shape = processor.random_transform()
            
            cv2.imwrite(image_path, transformed_image)
            # tqdm.tqdm.write(f"Transformed and saved image: {image_path}")
            logger.debug(f"已保存变换后的图像: {image_path}")

            original_coords = processor.load_yolo_labels(label_path)
            if not original_coords:
                logger.warning("未加载到原始标签，无法绘制。")

            transformed_coords = processor.transform_yolo_coords(original_coords, M_final, output_shape)
            
            with open(label_path, 'w', encoding='utf-8') as f_out:
                for class_id, x, y, w, h in transformed_coords:
                    f_out.write(f"{int(class_id)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                    logger.debug(f"写入变换后标签: {class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f} 到文件: {label_path}")
            

            # if os.path.exists(label_path):
            #     processor.valid_result(transformed_image, M_final, output_shape, label_path)
            # else:
            #     logger.warning(f"标签文件不存在: {label_path}")
                
        logger.info(f"已处理目录: {root}")
                
    
    
if __name__ == "__main__":
    main()
    
    