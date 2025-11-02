import os
import shutil
import random
import sys
import logging
import argparse


#tree
# - dataset
#   - images
#     - train
#         - class1.jpg
#         - class2.jpg
#     - valid
#         - class1.jpg
        
#   - labels
#     - train
#         - class1.txt
#         - class2.txt
#     - valid
#         - class1.txt

logging.basicConfig(
    #动态设置日志级别
    level=logging.DEBUG, 
    format='%(asctime)s - %(levelname)s - %(message)s' 
)

class SeperateValid:

    def __init__(self, valid_ratio=0.2, seed=42, dataset_dir="dataset"):
        self.valid_ratio = valid_ratio
        self.seed = seed

        self.dataset_dir = os.path.join(self.get_self_dir(), dataset_dir)

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        
        #先对原始数据集进行验证
        if not self.valid_database(mode=False):
            raise RuntimeError("Original dataset validation failed. Please check the dataset integrity.")

        if dataset_dir is not None:
            self.copy_dataset()
            self.logger.info(f"Dataset copied to: {self.dataset_dir}")
        self.logger.info(f"Using dataset directory: {self.dataset_dir}")

        if not os.path.exists(self.dataset_dir):
            raise FileNotFoundError(f"Dataset directory '{self.dataset_dir}' does not exist.")
            
    @staticmethod
    def create_symlinks(src_dir, dst_dir, file_list):
        """
        use symbol link to copy dataset folder
        辅助函数：在 dst_dir 中为 file_list 中的每个文件创建符号链接，
        这些链接指向 src_dir 中的原始文件。
        """
        os.makedirs(dst_dir, exist_ok=True)
        
        for file_name in file_list:
            src_path = os.path.join(src_dir, file_name)
            dst_path = os.path.join(dst_dir, file_name)
            # 检查源文件是否存在
            if not os.path.exists(src_path):
                # 这种情况很常见，比如图片有对应的label，但反过来不一定
                print(f"Warning: Source file not found, skipping link: {src_path}")
                continue 

            # 检查目标链接是否已存在，避免重复运行时出错
            if os.path.exists(dst_path) or os.path.lexists(dst_path):
                # os.lexists 可以检查到损坏的链接
                print(f"Warning: Link already exists, skipping: {dst_path}")
                continue

            try:
                os.symlink(src_path, dst_path)
            except OSError as e:
                print(f"Error creating symlink: {e}")
                if sys.platform == "win32":
                    print("Hint: On Windows, you might need to run this script as Administrator.")
                return False
            except NotImplementedError:
                print("Error: Symlinks not supported on this platform/filesystem.")
                return False
                
        return True
        
    
    def copy_dataset(self):
        """Copy the dataset to a new location for separation.

        Raises:
            RuntimeError: If the dataset copy fails.
            RuntimeError: If the dataset directory is not found.
        """
        idx = 0
        
        base_dir = self.dataset_dir + '_seperated'
        # prevent existing data from being modified,and error if the folder exists
        while os.path.exists(base_dir):
            idx += 1
            base_dir = self.dataset_dir + f'_seperated_{idx}'

        if not self.create_symlinks(os.path.join(self.dataset_dir, 'images'), os.path.join(base_dir, 'images'), os.listdir(os.path.join(self.dataset_dir, 'images'))):
            raise RuntimeError("Failed to create symlinks for images.")
        if not self.create_symlinks(os.path.join(self.dataset_dir, 'labels'), os.path.join(base_dir, 'labels'), os.listdir(os.path.join(self.dataset_dir, 'labels'))):
            raise RuntimeError("Failed to create symlinks for labels.")
        self.dataset_dir = base_dir
        
    
    def get_self_dir(self):
        return os.path.dirname(os.path.abspath(__file__))

#======================================================================================
    def valid_database(self, mode :bool)  -> bool:
        """验证数据集image和label是否一一对应,如果不对应则报错
        
        args:
            mode (bool): True 表示是分割之后的数据集验证,  False 表示原始数据集验证(只有train,没有valid)
        """
        image_dir = os.path.join(self.dataset_dir, 'images')
        label_dir = os.path.join(self.dataset_dir, 'labels')
        
        image_train_dir = os.path.join(image_dir, 'train')
        label_train_dir =  os.path.join(label_dir, 'train')
        
        image_valid_dir = os.path.join(image_dir, 'valid')
        label_valid_dir =  os.path.join(label_dir, 'valid')

        error_count = 0
        #交叉验证train set
        
        if not mode:
            for image_name in os.listdir(image_train_dir):
                label_name = os.path.splitext(image_name)[0] + '.txt'
                if not os.path.exists(os.path.join(label_train_dir, label_name)):
                    self.logger.error(f"Label file not found for image: {image_name}")
                    error_count += 1


            for label_name in os.listdir(label_train_dir):
                image_name = os.path.splitext(label_name)[0] + '.jpg'
                if not os.path.exists(os.path.join(image_train_dir, image_name)):
                    self.logger.error(f"Image file not found for label: {label_name}")
                    error_count += 1
            
            if mode:
                #交叉验证valid set
                for image_name in os.listdir(image_valid_dir):
                    label_name = os.path.splitext(image_name)[0] + '.txt'
                    if not os.path.exists(os.path.join(label_valid_dir, label_name)):
                        self.logger.error(f"Label file not found for image: {image_name}")
                        error_count += 1
                    
                for label_name in os.listdir(label_valid_dir):
                    image_name = os.path.splitext(label_name)[0] + '.jpg'
                    if not os.path.exists(os.path.join(image_valid_dir, image_name)):
                        self.logger.error(f"Image file not found for label: {label_name}")
                        error_count += 1
            
        if error_count == 0:
            self.logger.info("All images and labels are properly matched.")
            return True
        else:
            self.logger.error(f"Found {error_count} mismatches.")
            return False
        
        
        
        
        

    def create_train_dir(self):

        target_train_dir = os.path.join(self.dataset_dir,'images', 'train')
        os.makedirs(target_train_dir, exist_ok=True)
        
    def create_valid_dir(self):

        target_valid_dir = os.path.join(self.dataset_dir,'images', 'valid')
        os.makedirs(target_valid_dir, exist_ok=True)
        
    
    def start_seperate(self):
        total_moved = 0 
        
        random.seed(self.seed)
        self.create_train_dir()
        self.create_valid_dir()

        #先对全部文件shuffle
        train_dir = os.path.join(self.dataset_dir, 'images', 'train')
        all_files = os.listdir(train_dir)
        self.logger.info(f"Total files found: {len(all_files)}")
        
        random.shuffle(all_files)
        
        valid_count = int(len(all_files) * self.valid_ratio)
        self.logger.info(f"Valid count calculated: {valid_count}")
        #划分valid和train
        valid_files = all_files[:valid_count]
        train_files = all_files[valid_count:]
        
        #同时移动images和labels
        for file_name in valid_files:
            #移动images
            src_image_path = os.path.join(train_dir, file_name)
            dst_image_path = os.path.join(self.dataset_dir, 'images', 'valid', file_name)
            shutil.move(src_image_path, dst_image_path)
            
            #移动labels
            label_name = os.path.splitext(file_name)[0] + '.txt'
            src_label_path = os.path.join(self.dataset_dir, 'labels', 'train', label_name)
            dst_label_path = os.path.join(self.dataset_dir, 'labels', 'valid', label_name)
            if not os.path.exists(os.path.join(self.dataset_dir, 'labels', 'valid')):
                os.makedirs(os.path.join(self.dataset_dir, 'labels', 'valid'))
                self.logger.warning(f"Labels valid directory created: {os.path.join(self.dataset_dir, 'labels', 'valid')}")
            if os.path.exists(src_label_path):
                shutil.move(src_label_path, dst_label_path)
                
            total_moved += 1
            self.logger.debug(f'Moved {file_name} and {label_name} to valid set.')
            
        #最后验证数据集完整性
        if not self.valid_database(mode=True):
            raise RuntimeError("Dataset validation after separation failed. Please check the dataset integrity.")
            self.logger.info('-----------------------------------')
            self.logger.error('Dataset validation failed.')
            self.logger.info('-----------------------------------')

        self.logger.info('-----------------------------------')
        self.logger.info('Separation completed.')
        self.logger.info(f'Total moved files: {total_moved}')


def main():
    # 1. 使用 argparse 获取命令行参数
    parser = argparse.ArgumentParser(description='将数据集划分为训练集和验证集。')
    parser.add_argument(
        '-l', '--log-level', 
        type=str, 
        default='INFO', 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='设置日志输出级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)'
    )
    args = parser.parse_args()
    
    # 将字符串日志级别转换为 logging 模块的常量
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    
    # 2. 动态配置 logging 基础设置
    logging.basicConfig(
        level=log_level,  # 动态设置基础级别
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 3. 将动态级别传递给 SeperateValid
    seperater = SeperateValid(valid_ratio=0.2, seed=42, dataset_dir="dataset") 
    seperater.start_seperate()

    
if __name__ == '__main__':
    main()
