import cv2
import sys
import os
from pathlib import Path

path = Path(__file__).parent.resolve()
sys.path.append(os.path.join(path.parent, 'src'))

from image_processor import ImageProcessor
from parse_config import ParseConfig

def main():
    path = Path(__file__).parent.resolve()
    
    test_image_path = os.path.join(path, 'ablue001.jpg')
    test_label_path = os.path.join(path, 'ablue001.txt')
    
    config_path = os.path.join(path.parent, 'config', 'config.yaml')
    config = ParseConfig(config_path).config

    blur_probability = config.get('BLUR_PROBABILITY', 0.5) # 假设默认值
    kernel_range = config.get('KERNEL_RANGE', (5, 45))
    img_size = config.get('IMG_SIZE', (640, 640))
    scale_min = config.get('SCALE_MIN', 0.3)
    rotation_angle_range = config.get('ROTATION_ANGLE_RANGE', (-60, 60))
    perspective_range = config.get('PERSPECTIVE_RANGE', 0.1)

    config = {
        'BLUR_PROBABILITY': blur_probability,
        'KERNEL_RANGE': kernel_range,
        'IMG_SIZE': img_size,
        'SCALE_MIN': scale_min,
        'ROTATION_ANGLE_RANGE': rotation_angle_range,
        'PERSPECTIVE_RANGE': perspective_range
    }
    
    image = cv2.imread(test_image_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        logger.error(f"无法读取图像: {test_image_path}")
        return

    processor = ImageProcessor(image, **config)
    
    # 1. 执行变换
    transformed_image, M_final, output_shape = processor.random_transform()
    
    # 2. 执行验证
    processor.valid_result(transformed_image, M_final, output_shape, test_label_path)



if __name__ == "__main__":
    main()
