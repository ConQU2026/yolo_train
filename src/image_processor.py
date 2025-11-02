from parse_config import ParseConfig
import cv2
import random
import numpy as np
import logging
import time
import os 
from pathlib import Path

def setup_logging(level):
    logger = logging.getLogger()
    logger.setLevel(level)
    # 为日志添加一个处理器，否则可能看不到输出
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


class ImageProcessor:

    def __init__(self, image, logger=None, **config):
        self.logger = logger if logger else setup_logging(logging.INFO) # 确保logger有效
        if image is None:
            self.logger.error("输入图像为 None。")
            raise ValueError("输入图像为 None，请检查图像路径。")
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
        has_alpha = transformed_image.shape[2] == 4
        if has_alpha:
            input_image_for_blur = cv2.cvtColor(transformed_image, cv2.COLOR_BGRA2BGR)

        blurred_image = cv2.GaussianBlur(input_image_for_blur, (kernel_size, kernel_size), 0)
        
        # 5. 若原图有alpha通道
        if has_alpha:
            alpha = transformed_image[:, :, 3]
            blurred_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2BGRA)
            blurred_image[:, :, 3] = alpha
            
        return blurred_image


    def transform_yolo_coords(self, yolo_coords, M, output_shape):
        """
        对YOLO格式的坐标进行变换
        
        参数:
            yolo_coords: list of tuples, (class_id, center_x, center_y, width, height)
            M: 单个 2x3 的复合仿射变换矩阵
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
            
            # 3. 应用复合变换矩阵 M (M.T 的形状是 3x2)
            # (N, 3) @ (3, 2) -> (N, 2)
            transformed_corners = corners_hom @ M.T
            
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
        对图像进行随机变换（缩放、旋转、仿射）。
        返回:
            transformed_image: 变换后的图像
            M_final: 从原始图像到最终图像的 2x3 复合变换矩阵
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
        M_scale = np.float32([[scale, 0, 0], [0, scale, 0]]) # 缩放矩阵
        scaled_cols = int(cols * scale)
        scaled_rows = int(rows * scale)
        
        # 2. 随机旋转
        angle = random.uniform(*self.rotation_angle_range)
        # 旋转矩阵（相对于缩放后的图像中心）
        M_rot = cv2.getRotationMatrix2D((scaled_cols/2, scaled_rows/2), angle, 1)
        
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
        
        # 3. 随机仿射变换
        rows_rot, cols_rot = new_rows, new_cols # 这是旋转后的尺寸
        src_points = np.float32([[0, 0], [cols_rot-1, 0], [0, rows_rot-1]])
        distort_w = cols_rot * self.affine_distortion_range
        distort_h = rows_rot * self.affine_distortion_range
        dst_points = np.float32([
            [random.uniform(0, distort_w), random.uniform(0, distort_h)],
            [random.uniform(cols_rot - distort_w, cols_rot-1), random.uniform(0, distort_h)],
            [random.uniform(0, distort_w), random.uniform(rows_rot - distort_h, rows_rot-1)]
        ])
        M_affine = cv2.getAffineTransform(src_points, dst_points)
        
        # 4. 组合所有变换矩阵
        # M_final = M_affine * M_rot * M_scale
        # 需要使用 3x3 增广矩阵来组合
        M1_aug = np.vstack([M_scale, [0, 0, 1]])  # (3, 3)
        M2_aug = np.vstack([M_rot, [0, 0, 1]])    # (3, 3)
        M3_aug = np.vstack([M_affine, [0, 0, 1]]) # (3, 3)
        
        M_final_aug = M3_aug @ M2_aug @ M1_aug # 矩阵乘法
        
        M_final = M_final_aug[:2, :] # (2, 3)
        
        # 5. 应用最终的复合变换
        # 最终输出尺寸由旋转和仿射的画布决定
        final_w, final_h = cols_rot, rows_rot
        transformed_image = cv2.warpAffine(self.image, M_final, (final_w, final_h))

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
        # 处理 4 通道 BGRA 图像
        if draw_img.shape[2] == 4:
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
        验证变换结果：
        左边显示原图和原始框，右边显示变换后的图和变换后的框。
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

    blur_probability = config.get('BLUR_PROBABILITY', 0.5) # 假设默认值
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

    image_file = os.path.join(path, 'test', 'ablue001.jpg')
    
    label_file = os.path.join(path, 'test', 'ablue001.txt')
    
    image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
    
    if image is None:
        logger.error(f"无法读取图像: {image_file}")
        return

    processor = ImageProcessor(image, logger, **config)
    
    # 1. 执行变换
    transformed_image, M_final, output_shape = processor.random_transform()
    
    logger.info(f"图像变换完成。最终尺寸: {output_shape}")
    
    # 2. 执行验证
    processor.valid_result(transformed_image, M_final, output_shape, label_file)
    
    logger.info("验证图像已显示。")
    
if __name__ == "__main__":
    main()