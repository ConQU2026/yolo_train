import os
import random
import sys
import logging
from pathlib import Path


def setup_logging(level=logging.INFO):
    logger = logging.getLogger()
    # 如果没有 handler，添加一个简单的 StreamHandler
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    logger.setLevel(level)
    return logger


class SeperateValid:

    def __init__(self, valid_ratio=0.2, seed=42, dataset_dir="dataset", logger=None):
        self.seed = seed
        self.logger = logger
        self.valid_ratio = valid_ratio
        # 1. 定义原始数据集目录
        self.original_dataset_dir = os.path.join(Path(__file__).parent,'..', dataset_dir)
        
        # 2. 验证原始数据集
        if not self.valid_database(self.original_dataset_dir, mode=False):
            raise RuntimeError("Original dataset validation failed. Please check the dataset integrity.")
        
        # 3. 定义新的、分离后的数据集目录
        idx = 0
        base_dir = self.original_dataset_dir + '_seperated'
        # 循环检查，避免覆盖
        while os.path.exists(base_dir):
            idx += 1
            base_dir = self.original_dataset_dir + f'_seperated_{idx}'
        self.seperated_dataset_dir = base_dir
        
        self.logger.info(f"Original dataset: {self.original_dataset_dir}")
        self.logger.info(f"New separated dataset will be created at: {self.seperated_dataset_dir}")


    def create_symlinks(self, src_path, dst_path):
        """
        辅助函数：在 dst_path 创建一个指向 src_path 的符号链接。
        """
        try:
            # 检查源文件是否存在
            if not os.path.exists(src_path):
                self.logger.warning(f"Warning: Source file not found, skipping link: {src_path}")
                return False

            # 检查目标链接是否已存在
            if os.path.exists(dst_path) or os.path.lexists(dst_path):
                self.logger.warning(f"Warning: Link already exists, skipping: {dst_path}")
                return False
                
            os.symlink(src_path, dst_path)
            return True
            
        except OSError as e:
            self.logger.error(f"Error creating symlink: {e}")
            if sys.platform == "win32":
                self.logger.error("Hint: On Windows, you might need to run this script as Administrator.")
            return False
        except NotImplementedError:
            self.logger.error("Error: Symlinks not supported on this platform/filesystem.")
            return False


    #======================================================================================
    def valid_database(self, base_dir: str, mode: bool) -> bool:
        """验证数据集image和label是否一一对应
        
        args:
            base_dir (str): 要验证的数据集根目录 (例如 'dataset' 或 'dataset_seperated')
            mode (bool): True 表示分割之后的数据集验证 (有train/valid), False 表示原始数据集验证(只有train)
        """
        image_dir = os.path.join(base_dir, 'images')
        label_dir = os.path.join(base_dir, 'labels')
        
        image_train_dir = os.path.join(image_dir, 'train')
        label_train_dir =  os.path.join(label_dir, 'train')
        
        image_valid_dir = os.path.join(image_dir, 'valid')
        label_valid_dir =  os.path.join(label_dir, 'valid')

        error_count = 0
        
        # 检查 train set 目录是否存在
        if not os.path.exists(image_train_dir) or not os.path.exists(label_train_dir):
            self.logger.error(f"Train directory not found in {base_dir}")
            return False

        # 交叉验证 train set
        self.logger.debug(f"Validating train set in {base_dir}...")
        for image_name in os.listdir(image_train_dir):
            label_name = os.path.splitext(image_name)[0] + '.txt'
            if not os.path.exists(os.path.join(label_train_dir, label_name)):
                # 允许没有标签的图片存在
                self.logger.warning(f"Label file not found for image (Train): {image_name}")

        for label_name in os.listdir(label_train_dir):
            # 假设图片是 .jpg, 实际可能需要更鲁棒的检查
            image_name_jpg = os.path.splitext(label_name)[0] + '.jpg'
            image_name_png = os.path.splitext(label_name)[0] + '.png'
            if not os.path.exists(os.path.join(image_train_dir, image_name_jpg)) and \
                not os.path.exists(os.path.join(image_train_dir, image_name_png)):
                self.logger.error(f"Image file not found for label (Train): {label_name}")
                error_count += 1
        
        if mode:
            # 检查 valid set 目录是否存在
            if not os.path.exists(image_valid_dir) or not os.path.exists(label_valid_dir):
                self.logger.error(f"Valid directory not found in {base_dir}")
                return False
                
            self.logger.debug(f"Validating valid set in {base_dir}...")
            # 交叉验证 valid set
            for image_name in os.listdir(image_valid_dir):
                label_name = os.path.splitext(image_name)[0] + '.txt'
                if not os.path.exists(os.path.join(label_valid_dir, label_name)):
                    self.logger.warning(f"Label file not found for image (Valid): {image_name}")

            for label_name in os.listdir(label_valid_dir):
                image_name_jpg = os.path.splitext(label_name)[0] + '.jpg'
                image_name_png = os.path.splitext(label_name)[0] + '.png'
                if not os.path.exists(os.path.join(image_valid_dir, image_name_jpg)) and \
                    not os.path.exists(os.path.join(image_valid_dir, image_name_png)):
                    self.logger.error(f"Image file not found for label (Valid): {label_name}")
                    error_count += 1
        
        if error_count == 0:
            self.logger.info(f"Validation successful for {base_dir} (Mode: {mode}).")
            return True
        else:
            self.logger.error(f"Found {error_count} critical mismatches in {base_dir}.")
            return False

    def start_seperate(self):
        random.seed(self.seed)

        # 1. 定义所有 *源* 和 *目标* 路径
        src_img_dir = os.path.join(self.original_dataset_dir, 'images', 'train')
        src_lbl_dir = os.path.join(self.original_dataset_dir, 'labels', 'train')

        dst_train_img_dir = os.path.join(self.seperated_dataset_dir, 'images', 'train')
        dst_valid_img_dir = os.path.join(self.seperated_dataset_dir, 'images', 'valid')
        dst_train_lbl_dir = os.path.join(self.seperated_dataset_dir, 'labels', 'train')
        dst_valid_lbl_dir = os.path.join(self.seperated_dataset_dir, 'labels', 'valid')

        # 2. 创建 *新的* 目标目录结构
        os.makedirs(dst_train_img_dir, exist_ok=True)
        os.makedirs(dst_valid_img_dir, exist_ok=True)
        os.makedirs(dst_train_lbl_dir, exist_ok=True)
        os.makedirs(dst_valid_lbl_dir, exist_ok=True)

        # 3. 从 *原始* 目录获取文件列表
        # 确保只获取图片文件，排除 .DS_Store 等
        all_image_files = [f for f in os.listdir(src_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not all_image_files:
            self.logger.error(f"No image files found in {src_img_dir}.")
            return

        self.logger.info(f"Found {len(all_image_files)} images in original train set.")
        
        random.shuffle(all_image_files)
        
        # 4. 分割列表
        valid_count = int(len(all_image_files) * self.valid_ratio)
        valid_files = all_image_files[:valid_count]
        train_files = all_image_files[valid_count:]
        
        self.logger.info(f"Splitting into {len(train_files)} train and {len(valid_files)} valid files (by link).")

        # 5. 为 *新训练集* 创建符号链接
        self.logger.info("Creating symlinks for NEW train set...")
        total_train_links = 0
        for img_name in train_files:
            lbl_name = os.path.splitext(img_name)[0] + '.txt'
            
            src_img_path = os.path.join(src_img_dir, img_name)
            dst_img_path = os.path.join(dst_train_img_dir, img_name)
            
            src_lbl_path = os.path.join(src_lbl_dir, lbl_name)
            dst_lbl_path = os.path.join(dst_train_lbl_dir, lbl_name)

            # 链接图片
            if self.create_symlinks(src_img_path, dst_img_path):
                total_train_links += 1
            # 链接标签 (如果存在)
            if os.path.exists(src_lbl_path):
                self.create_symlinks(src_lbl_path, dst_lbl_path)

        # 6. 为 *新验证集* 创建符号链接
        self.logger.info("Creating symlinks for NEW valid set...")
        total_valid_links = 0
        for img_name in valid_files:
            lbl_name = os.path.splitext(img_name)[0] + '.txt'

            src_img_path = os.path.join(src_img_dir, img_name)
            dst_img_path = os.path.join(dst_valid_img_dir, img_name)
            
            src_lbl_path = os.path.join(src_lbl_dir, lbl_name)
            dst_lbl_path = os.path.join(dst_valid_lbl_dir, lbl_name)
            
            # 链接图片
            if self.create_symlinks(src_img_path, dst_img_path):
                total_valid_links += 1
            # 链接标签 (如果存在)
            if os.path.exists(src_lbl_path):
                self.create_symlinks(src_lbl_path, dst_lbl_path)

        # 7. 最后验证 *新创建的* 数据集
        if not self.valid_database(self.seperated_dataset_dir, mode=True):
            self.logger.error('-----------------------------------')
            self.logger.error('Validation of NEW separated dataset FAILED.')
            self.logger.error('-----------------------------------')
        else:
            self.logger.info('-----------------------------------')
            self.logger.info('Separation completed successfully.')
            self.logger.info(f'Total train links created: {total_train_links}')
            self.logger.info(f'Total valid links created: {total_valid_links}')
            self.logger.info(f"Original dataset at '{self.original_dataset_dir}' remains UNCHANGED.")
            self.logger.info('-----------------------------------')


def main():
    # 1. 设置日志级别
    logger = setup_logging(logging.INFO)

    try:
        # 2. 将日志记录器传递给 SeperateValid
        seperater = SeperateValid(valid_ratio=0.2, seed=42, dataset_dir="dataset", logger=logger)
        seperater.start_seperate()
    except (RuntimeError, FileNotFoundError) as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)

    
if __name__ == '__main__':
    main()
