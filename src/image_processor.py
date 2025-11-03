from parse_config import ParseConfig
import cv2
import random
import numpy as np
import logging
import time
import os
from pathlib import Path

"""
TODO:
- noise, exposure, contrast functions
- 不把配置参数写在init中, 而是在使用random_transform时传入, 解耦合
"""

def setup_logging(level):
    logger = logging.getLogger()
    logger.setLevel(level)
    return logger

class ImageProcessor:

    def __init__(self, image, logger=None, **config):
        self.logger = logger if logger else setup_logging(logging.INFO) # 确保logger有效
        if image is None:
            self.logger.error("输入图像为 None。")
            raise ValueError("输入图像为 None, 请检查图像路径。")
        self.image = image
        
        self.blur_prob = config['BLUR_PROBABILITY']
        self.kernel_range = config['KERNEL_RANGE']
        self.img_size = config['IMG_SIZE']
        self.scale_min = config['SCALE_MIN']
        self.rotation_angle_range = config['ROTATION_ANGLE_RANGE']
        # self.affine_distortion_range = config['AFFINE_DISTORTION_RANGE']
        self.perspective_range = config['PERSPECTIVE_RANGE'] # 新增：透视变换扰动范围
        

    # 1.随机模糊
    def add_random_blur(self, transformed_image):
        """对图像应用随机模糊"""
        # 1. 按概率决定是否应用模糊
        if random.random() > self.blur_prob:
            return transformed_image

        # 2. 处理模糊核大小
        min_kernel, max_kernel = self.kernel_range
        
        if min_kernel % 2 == 0: min_kernel += 1
        if max_kernel % 2 == 0: max_kernel -= 1
            
        if min_kernel > max_kernel: min_kernel, max_kernel = max_kernel, min_kernel
        if min_kernel < 3: min_kernel = 3  
        
        # 3. 随机选择核大小
        try:
            kernel_sizes = list(range(min_kernel, max_kernel + 1, 2))
            if not kernel_sizes:
                kernel_size = min_kernel # 备用
            else:
                kernel_size = random.choice(kernel_sizes)
        except ValueError as e:
            self.logger.warning(f"计算模糊核大小时出错: {e}。使用默认值 3。")
            kernel_size = 3

        # 4. 应用高斯模糊
        # 确保图像不是 BGRA，如果是，先转成 BGR
        input_image_for_blur = transformed_image
        
        # 检查通道数前确保图像是 numpy 数组
        if isinstance(transformed_image, np.ndarray):
            has_alpha = transformed_image.shape[2] == 4 if len(transformed_image.shape) > 2 else False
            if has_alpha:
                input_image_for_blur = cv2.cvtColor(transformed_image, cv2.COLOR_BGRA2BGR)
        else:
             self.logger.warning("输入图像非Numpy数组，跳过模糊。")
             return transformed_image

        blurred_image = cv2.GaussianBlur(input_image_for_blur, (kernel_size, kernel_size), 0)
        
        # 5. 若原图有alpha通道
        if has_alpha:
            alpha = transformed_image[:, :, 3]
            blurred_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2BGRA)
            blurred_image[:, :, 3] = alpha
            
        return blurred_image


    # 2. 随机改变曝光
    def random_adjust_exposure(self, image, exposure_range=(0.5, 1.5)):
        pass
    
    # 3. 随机改变对比度
    def random_adjust_contrast(self, image, contrast_range=(0.5, 1.5)):
        pass
    
    def random_add_noise(self, imag, noise_level=10):
        pass
    
    def transform_yolo_coords(self, yolo_coords, M, output_shape):
        
        """
        对YOLO坐标进行变换（支持 2x3 仿射 和 3x3 透视 矩阵）
        
        参数:
            yolo_coords: list of tuples, (class_id, center_x, center_y, width, height)
            M: 2x3 或 3x3 的复合变换矩阵
            output_shape: 输出图像的尺寸 (height, width)
        
        返回:
            transformed_coords: list of tuples
        """
        if not yolo_coords:
            return []
        
        output_h, output_w = output_shape
        if output_w == 0 or output_h == 0:
            return []
            
        transformed_coords = []
        
        # 获取原始图像尺寸
        orig_h, orig_w = self.image.shape[:2]
        
        for coord in yolo_coords:
            class_id = coord[0]
            center_x_norm, center_y_norm, width_norm, height_norm = coord[1:5]
            
            # 1. 将YOLO坐标转换为原始图像的像素坐标角点
            center_x_px = center_x_norm * orig_w
            center_y_px = center_y_norm * orig_h
            width_px = width_norm * orig_w
            height_px = height_norm * orig_h
            
            x1 = center_x_px - width_px / 2
            y1 = center_y_px - height_px / 2
            x2 = center_x_px + width_px / 2
            y2 = center_y_px + height_px / 2
            
            corners = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            
            # 2. 将角点转换为齐次坐标 (N, 3)
            corners_hom = np.hstack([corners, np.ones((corners.shape[0], 1))])
            
            # 3. 应用复合变换矩阵 M (根据 M 的形状)
            if M.shape == (2, 3):
                # --- 仿射变换 (原逻辑) ---
                # (N, 3) @ (3, 2) -> (N, 2)
                transformed_corners = corners_hom @ M.T
            
            elif M.shape == (3, 3):
                # --- 透视变换 (新逻辑) ---
                # (N, 3) @ (3, 3) -> (N, 3)  结果是 [x', y', w']
                transformed_hom = corners_hom @ M.T
                
                # 透视除法: (x, y) = (x'/w', y'/w')
                w_prime = transformed_hom[:, 2]
                
                # 防止除以零
                w_prime[np.abs(w_prime) < 1e-6] = 1e-6
                
                # (N, 1)
                w_prime_inv = 1.0 / w_prime
                
                # (N, 2)
                transformed_corners_x = transformed_hom[:, 0] * w_prime_inv
                transformed_corners_y = transformed_hom[:, 1] * w_prime_inv
                transformed_corners = np.vstack([transformed_corners_x, transformed_corners_y]).T
            else:
                self.logger.error(f"不支持的变换矩阵形状: {M.shape}")
                continue

            # 4. 计算变换后的边界框
            x_coords = transformed_corners[:, 0]
            y_coords = transformed_corners[:, 1]
            
            new_x1 = np.min(x_coords)
            new_y1 = np.min(y_coords)
            new_x2 = np.max(x_coords)
            new_y2 = np.max(y_coords)
            
            # 5. 计算新的中心和宽高（像素坐标）
            new_center_x_px = (new_x1 + new_x2) / 2
            new_center_y_px = (new_y1 + new_y2) / 2
            new_width_px = new_x2 - new_x1
            new_height_px = new_y2 - new_y1
            
            # 6. 转换回YOLO格式（归一化）并裁剪到[0, 1]范围
            new_center_x_norm = np.clip(new_center_x_px / output_w, 0, 1)
            new_center_y_norm = np.clip(new_center_y_px / output_h, 0, 1)
            new_width_norm = np.clip(new_width_px / output_w, 0, 1)
            new_height_norm = np.clip(new_height_px / output_h, 0, 1)
            
            # 7. 过滤掉变换后过小的框
            if new_width_norm * output_w < 1 or new_height_norm * output_h < 1:
                continue
                
            transformed_coords.append((class_id, new_center_x_norm, new_center_y_norm, new_width_norm, new_height_norm))
        
        return transformed_coords

    def random_transform(self):
        """
        对图像进行随机变换（缩放、旋转、透视）。
        返回:
            transformed_image: 变换后的图像
            M_final: 从原始图像到最终图像的 3x3 复合变换矩阵
            final_shape: (height, width)
        """
        if len(self.image.shape) < 2:
            raise ValueError("图像尺寸异常，无法进行变换")
        rows, cols = self.image.shape[:2]
        max_w, max_h = self.img_size

        # 1. 随机缩放 
        scale_w = max_w / cols
        scale_h = max_h / rows
        max_allowed_scale = min(scale_w, scale_h)
        scale = random.uniform(self.scale_min, max_allowed_scale)
        M_scale = np.float32([[scale, 0, 0], [0, scale, 0]]) # 缩放矩阵 (2x3)
        scaled_cols = int(cols * scale)
        scaled_rows = int(rows * scale)
        
        # 2. 随机旋转
        angle = random.uniform(*self.rotation_angle_range)
        M_rot = cv2.getRotationMatrix2D((scaled_cols/2, scaled_rows/2), angle, 1) # (2x3)
        
        # 计算旋转后的边界框大小
        cos_val = abs(M_rot[0, 0])
        sin_val = abs(M_rot[0, 1])
        new_cols = int((scaled_rows * sin_val) + (scaled_cols * cos_val))
        new_rows = int((scaled_rows * cos_val) + (scaled_cols * sin_val))
        
        # 处理旋转后可能超出的情况
        if new_cols > max_w or new_rows > max_h:
            scale_after_rot = min(max_w / new_cols, max_h / new_rows)
            new_cols = int(new_cols * scale_after_rot)
            new_rows = int(new_rows * scale_after_rot)
            # 将这个额外的缩放应用到旋转矩阵中
            M_rot[0, 0] *= scale_after_rot
            M_rot[0, 1] *= scale_after_rot
            M_rot[1, 0] *= scale_after_rot
            M_rot[1, 1] *= scale_after_rot
            
        # 调整旋转矩阵的平移分量，使图像在新画布中居中
        M_rot[0, 2] += (new_cols / 2) - (scaled_cols / 2)
        M_rot[1, 2] += (new_rows / 2) - (scaled_rows / 2)
        
        # --- 3. 随机透视变换  ---
        rows_final, cols_final = new_rows, new_cols 
        
        # 4个源点（画布的四个角）
        src_points = np.float32([
            [0, 0], 
            [cols_final - 1, 0], 
            [cols_final - 1, rows_final - 1], 
            [0, rows_final - 1]
        ])
        
        # 4个目标点（在源点基础上随机扰动）
        distort_w = cols_final * self.perspective_range
        distort_h = rows_final * self.perspective_range
        
        dst_points = np.float32([
            [random.uniform(0, distort_w), random.uniform(0, distort_h)],
            [random.uniform(cols_final - 1 - distort_w, cols_final - 1), random.uniform(0, distort_h)],
            [random.uniform(cols_final - 1 - distort_w, cols_final - 1), random.uniform(rows_final - 1 - distort_h, rows_final - 1)],
            [random.uniform(0, distort_w), random.uniform(rows_final - 1 - distort_h, rows_final - 1)]
        ])

        # 使用 4 个点对计算透视变换矩阵 (3x3)
        M_perspective = cv2.getPerspectiveTransform(src_points, dst_points) 
        
        # 4. 组合所有变换矩阵
        # M_final = M_perspective * M_rot * M_scale
        M1_aug = np.vstack([M_scale, [0, 0, 1]])  # (3, 3)
        M2_aug = np.vstack([M_rot, [0, 0, 1]])    # (3, 3)
        M3_aug = M_perspective                   # (3, 3)
        
        M_final_aug = M3_aug @ M2_aug @ M1_aug 
        
        M_final = M_final_aug # (3, 3)
        
        # 5. 复合变换
        final_w, final_h = cols_final, rows_final
        transformed_image = cv2.warpPerspective(self.image, M_final, (final_w, final_h))

        # 6. 随机模糊
        transformed_image = self.add_random_blur(transformed_image)

        final_shape = (final_h, final_w)
        
        return transformed_image, M_final, final_shape

    
    def load_yolo_labels(self, label_path):
        """从 .txt 文件加载 YOLO 标签"""
        if not os.path.exists(label_path):
            self.logger.warning(f"标签文件未找到: {label_path}")
            return []
            
        coords = []
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # (class_id, x, y, w, h)
                        coords.append(tuple(map(float, parts)))
        except Exception as e:
            self.logger.error(f"加载标签文件时出错: {e}")
        return coords

    def draw_boxes(self, image, yolo_coords, color=(0, 255, 0), thickness=2):
        """在图像上绘制YOLO格式的边界框"""
        draw_img = image.copy()
        
        if len(draw_img.shape) == 2: # 灰度图
            draw_img = cv2.cvtColor(draw_img, cv2.COLOR_GRAY2BGR)
        elif draw_img.shape[2] == 4: # BGRA
            draw_img = cv2.cvtColor(draw_img, cv2.COLOR_BGRA2BGR)
        
        img_h, img_w = draw_img.shape[:2]
        
        for coord in yolo_coords:
            class_id, cx, cy, w, h = coord
            
            # 从归一化坐标反算
            cx_px = cx * img_w
            cy_px = cy * img_h
            w_px = w * img_w
            h_px = h * img_h
            
            # 计算左上角和右下角
            x1 = int(cx_px - w_px / 2)
            y1 = int(cy_px - h_px / 2)
            x2 = int(cx_px + w_px / 2)
            y2 = int(cy_px + h_px / 2)
            
            cv2.rectangle(draw_img, (x1, y1), (x2, y2), color, thickness)
            
        return draw_img

    def valid_result(self, transformed_image, M_final, output_shape, label_file_path):
        """
        验证变换结果
        """
        self.logger.info("开始验证变换结果...")
        
        # 1. 加载原始标签
        original_coords = self.load_yolo_labels(label_file_path)
        if not original_coords:
            self.logger.warning("未加载到原始标签，无法绘制。")
            
        # 2. 转换标签 
        transformed_coords = self.transform_yolo_coords(original_coords, M_final, output_shape)
        
        # 3. 绘制原始框 (绿色)
        original_with_boxes = self.draw_boxes(self.image, original_coords, color=(0, 255, 0))
        
        # 4. 绘制变换后的框 (蓝色)
        transformed_with_boxes = self.draw_boxes(transformed_image, transformed_coords, color=(255, 0, 0))
        
        # 5. 拼接图像
        h1, w1 = original_with_boxes.shape[:2]
        h2, w2 = transformed_with_boxes.shape[:2]
        
        max_h = max(h1, h2)
        total_w = w1 + w2
        # 确保画布是 3 通道
        canvas = np.zeros((max_h, total_w, 3), dtype=np.uint8)
        canvas[:h1, :w1] = original_with_boxes
        canvas[:h2, w1:w1+w2] = transformed_with_boxes
        
        cv2.imshow('Validation: Original (Green) vs Transformed (Blue)', canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    
def main():
    
    logger = setup_logging(logging.INFO)
    
    path = Path(__file__).parent.parent.resolve()

    config_path = os.path.join(path, 'config', 'config.yaml')
    config = ParseConfig(config_path).config

    blur_probability = config.get('BLUR_PROBABILITY', 0.5) 
    kernel_range = config.get('KERNEL_RANGE', (5, 45))
    img_size = config.get('IMG_SIZE', (640, 640))
    scale_min = config.get('SCALE_MIN', 0.3)
    rotation_angle_range = config.get('ROTATION_ANGLE_RANGE', (-60, 60))
    perspective_range = config.get('PERSPECTIVE_RANGE', 0.1) 

    # 传递给 Processor 的配置字典
    config_params = {
        'BLUR_PROBABILITY': blur_probability,
        'KERNEL_RANGE': kernel_range,
        'IMG_SIZE': img_size,
        'SCALE_MIN': scale_min,
        'ROTATION_ANGLE_RANGE': rotation_angle_range,
        'PERSPECTIVE_RANGE': perspective_range # 使用新的key
    }
    image_file = os.path.join(path, 'test', 'ablue001.jpg')
    label_file = os.path.join(path, 'test', 'ablue001.txt')
    
    # 检查测试文件是否存在
    if not os.path.exists(image_file):
        logger.error(f"测试图像未找到: {image_file}")
        logger.error("请确保 'test/ablue001.jpg' 文件存在于项目根目录。")
        return
    if not os.path.exists(label_file):
        logger.warning(f"测试标签未找到: {label_file}")
    # ---------------------

    image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
    
    if image is None:
        logger.error(f"无法读取图像: {image_file}")
        return

    processor = ImageProcessor(image, logger, **config_params)
    
    # 1. 执行变换
    transformed_image, M_final, output_shape = processor.random_transform()
    
    logger.info(f"图像变换完成。最终尺寸: {output_shape}")
    logger.info(f"最终变换矩阵 (3x3):\n{M_final}")
    
    # 2. 执行验证
    processor.valid_result(transformed_image, M_final, output_shape, label_file)
    
    logger.info("验证图像已显示。")
    
if __name__ == "__main__":
    main()