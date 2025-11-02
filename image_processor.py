from parse_config import ParseConfig
import cv2
import random
import numpy as np
import logging
import time

def setup_logging(level):
    logger = logging.getLogger()
    logger.setLevel(level)
    return logger


class ImageProcessor:


    def __init__(self, image, logger=None, **config):
        self.logger = logger
        self.image = image
        
        self.blur_prob = config['BLUR_PROBABILITY']
        self.kernel_range = config['KERNEL_RANGE']
        self.img_size = config['IMG_SIZE']
        self.scale_min = config['SCALE_MIN']
        self.rotation_angle_range = config['ROTATION_ANGLE_RANGE']
        self.affine_distortion_range = config['AFFINE_DISTORTION_RANGE']


    def add_random_blur(self, transformed_image):
        """对图像应用随机模糊"""
        # 1. 按概率决定是否应用模糊
        if random.random() > self.blur_prob:
            return transformed_image

        # 2. 处理模糊核大小（确保为奇数，且在用户指定范围内）
        min_kernel, max_kernel = self.kernel_range
        
        # 若用户输入非奇数，自动调整为最近的奇数
        if min_kernel % 2 == 0:
            min_kernel += 1
        if max_kernel % 2 == 0:
            max_kernel -= 1
            
        # 确保核大小范围有效
        if min_kernel > max_kernel:
            min_kernel, max_kernel = max_kernel, min_kernel
        if min_kernel < 3:
            min_kernel = 3  
        
        # 3. 随机选择核大小（从奇数列表中选择）
        kernel_sizes = list(range(min_kernel, max_kernel + 1, 2))
        kernel_size = random.choice(kernel_sizes)
        
        # 4. 应用高斯模糊
        blurred_image = cv2.GaussianBlur(transformed_image, (kernel_size, kernel_size), 0)
        
        # 5. 若原图有alpha通道
        if transformed_image.shape[2] == 4:
            # 提取原图alpha通道，合并到模糊后的图像
            alpha = transformed_image[:, :, 3]
            blurred_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2BGRA)
            blurred_image[:, :, 3] = alpha
        return blurred_image


    def random_transform(self):
        """对图像进行随机变换（含新增的模糊处理）"""
        if len(self.image.shape) < 2:
            raise ValueError("图像尺寸异常，无法进行变换")
        rows, cols = self.image.shape[:2]
        max_w, max_h = self.img_size

        # 1. 随机缩放
        scale_w = max_w / cols
        scale_h = max_h / rows
        max_allowed_scale = min(scale_w, scale_h)
        scale = random.uniform(self.scale_min, max_allowed_scale)
        scaled_cols = int(cols * scale)
        scaled_rows = int(rows * scale)
        scaled = cv2.resize(self.image, (scaled_cols, scaled_rows), interpolation=cv2.INTER_AREA)
        
        # 2. 随机旋转
        angle = random.uniform(*self.rotation_angle_range)
        M_rot = cv2.getRotationMatrix2D((scaled_cols/2, scaled_rows/2), angle, 1)
        cos_val = abs(M_rot[0, 0])
        sin_val = abs(M_rot[0, 1])
        new_cols = int((scaled_rows * sin_val) + (scaled_cols * cos_val))
        new_rows = int((scaled_rows * cos_val) + (scaled_cols * sin_val))
        if new_cols > max_w or new_rows > max_h:
            scale_after_rot = min(max_w / new_cols, max_h / new_rows)
            new_cols = int(new_cols * scale_after_rot)
            new_rows = int(new_rows * scale_after_rot)
            M_rot[0, 0] *= scale_after_rot
            M_rot[0, 1] *= scale_after_rot
            M_rot[1, 0] *= scale_after_rot
            M_rot[1, 1] *= scale_after_rot
        M_rot[0, 2] += (new_cols / 2) - (scaled_cols / 2)
        M_rot[1, 2] += (new_rows / 2) - (scaled_rows / 2)
        rotated = cv2.warpAffine(scaled, M_rot, (new_cols, new_rows))
        
        # 3. 随机仿射变换
        rows_rot, cols_rot = rotated.shape[:2]
        src_points = np.float32([[0, 0], [cols_rot-1, 0], [0, rows_rot-1]])
        distort_w = cols_rot * self.affine_distortion_range
        distort_h = rows_rot * self.affine_distortion_range
        dst_points = np.float32([
            [random.uniform(0, distort_w), random.uniform(0, distort_h)],
            [random.uniform(cols_rot - distort_w, cols_rot-1), random.uniform(0, distort_h)],
            [random.uniform(0, distort_w), random.uniform(rows_rot - distort_h, rows_rot-1)]
        ])
        M_affine = cv2.getAffineTransform(src_points, dst_points)
        transformed_image = cv2.warpAffine(rotated, M_affine, (cols_rot, rows_rot))

        #4. 随机模糊
        transformed_image = self.add_random_blur(transformed_image)

        return transformed_image
    
    
def main():
    
    logger = setup_logging(logging.INFO)
    config = ParseConfig('config.yaml').config
    
    blur_probability = config.get('BLUR_PROBABILITY')
    kernel_range = config.get('KERNEL_RANGE', (5, 45))
    img_size = config.get('IMG_SIZE', (640, 640))
    scale_min = config.get('SCALE_MIN', 0.3)
    rotation_angle_range = config.get('ROTATION_ANGLE_RANGE', (-60, 60))
    affine_distortion_range = config.get('AFFINE_DISTORTION_RANGE', 0.1)

    config = {
        'BLUR_PROBABILITY': blur_probability,
        'KERNEL_RANGE': kernel_range,
        'IMG_SIZE': img_size,
        'SCALE_MIN': scale_min,
        'ROTATION_ANGLE_RANGE': rotation_angle_range,
        'AFFINE_DISTORTION_RANGE': affine_distortion_range
    }
    
    image_file = './test/test.jpg'
    
    image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
    processor = ImageProcessor(image, logger, **config)
    transformed_image = processor.random_transform()
    
    cv2.imshow('Transformed Image', transformed_image)
    cv2.waitKey(0)
    logger.info("Transformed image displayed.")
    
if __name__ == "__main__":
    main()