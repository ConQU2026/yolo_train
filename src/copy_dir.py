import os
from pathlib import Path
import shutil
import sys

class DirCopier:
    
    def __init__(self, src_path=None, dst_path=None, logger=None):
        self.src_path = src_path
        self.dst_path = dst_path
        self.logger = logger

    def create_symlinks(self):
        """
        辅助函数：在 dst_path 创建一个指向 src_path 的符号链接。
        """
        try:
            # 检查源文件是否存在
            if not os.path.exists(self.src_path):
                self.logger.warning(f"Warning: Source file not found, skipping link: {self.src_path}")
                return False

            # 检查目标链接是否已存在
            if os.path.exists(self.dst_path) or os.path.lexists(self.dst_path):
                self.logger.warning(f"Warning: Link already exists, skipping: {self.dst_path}")
                return False

            os.symlink(self.src_path, self.dst_path)
            return True
            
        except OSError as e:
            self.logger.error(f"Error creating symlink: {e}")
            if sys.platform == "win32":
                self.logger.error("Hint: On Windows, you might need to run this script as Administrator.")
            return False
        except NotImplementedError:
            self.logger.error("Error: Symlinks not supported on this platform/filesystem.")
            return False

    def copy_directory_with_symlinks(self):
        """
        复制整个目录结构，通过创建符号链接而非实际复制文件。
        """
        
        #递归遍历源目录
        for root, dirs, files in os.walk(self.src_path):
            # 计算相对路径
            rel_path = os.path.relpath(root, self.src_path)
            # 目标目录路径
            dst_dir = os.path.join(self.dst_path, rel_path)
            # 创建目标目录
            os.makedirs(dst_dir, exist_ok=True)
            
            for file in files:
                src_file_path = os.path.join(root, file)
                dst_file_path = os.path.join(dst_dir, file)
                
                # 创建符号链接
                self.src_path = src_file_path
                self.dst_path = dst_file_path
                self.create_symlinks()
                self.logger.debug(f"Linked {src_file_path} to {dst_file_path}")
        
        self.logger.info(f"Directory copied with symlinks from {self.src_path} to {self.dst_path}")    
        
    def copy_directory(self):
        """
        完全复制,不使用符号链接
        """
        
        #use shutil.copytree to copy entire directory
        shutil.copytree(self.src_path, self.dst_path, dirs_exist_ok=True)
        
        self.logger.info(f"Directory fully copied from {self.src_path} to {self.dst_path}")
            
            
def main():
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    path = Path(__file__).parent.resolve()

    src_directory = os.path.join(path, '..', 'source_dir')
    dst_directory = os.path.join(path, '..','linked_dir')

    if not os.path.exists(src_directory):
        logger.error(f"Source directory does not exist: {src_directory}")
        return
    
    copier = DirCopier(src_path=src_directory, dst_path=dst_directory, logger=logger)
    copier.copy_directory()

if __name__ == "__main__":
    main()