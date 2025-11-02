"""
KFS 检测模型训练脚本
基于 YOLOv8，学习 main.py 的数据增强方法
"""

import os
import yaml
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
import random
from tqdm import tqdm

# ====================== 配置参数 ======================
# 路径配置
WORKSPACE_DIR = Path(__file__).parent
DATASET_DIR = WORKSPACE_DIR / "dataset_seperated"
CLASSES_FILE = WORKSPACE_DIR / "classes.txt"
OUTPUT_DIR = WORKSPACE_DIR / "runs" / "train"

# 训练参数
EPOCHS = 100
BATCH_SIZE = 16
IMAGE_SIZE = 640
DEVICE = 0  # 0 表示使用第一块GPU，'cpu' 表示使用CPU
WORKERS = 8

# 数据增强参数（学习自 main.py）
ROTATION_ANGLE_RANGE = (-60, 60)   # 旋转角度范围
AFFINE_DISTORTION_RANGE = 0.1      # 仿射扭曲幅度
BLUR_PROBABILITY = 0.9             # 模糊概率
BLUR_KERNEL_RANGE = (5, 45)        # 模糊核大小范围
SCALE_RANGE = (0.5, 1.5)           # 缩放范围

# 模型配置
PRETRAINED_MODEL = "yolov8n.pt"    # 预训练模型 (n/s/m/l/x)
# ======================================================


def read_classes(classes_file=CLASSES_FILE):
    """读取类别文件"""
    try:
        with open(classes_file, 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f if line.strip()]
        
        if not classes:
            print(f"警告：{classes_file} 为空，将从标签文件中自动提取类别")
            classes = auto_detect_classes()
        
        print(f"✓ 加载了 {len(classes)} 个类别")
        return classes
    except FileNotFoundError:
        print(f"未找到 {classes_file}，将自动检测类别...")
        classes = auto_detect_classes()
        # 保存类别文件
        with open(classes_file, 'w', encoding='utf-8') as f:
            for cls in classes:
                f.write(f"{cls}\n")
        print(f"✓ 已保存类别到 {classes_file}")
        return classes


def auto_detect_classes():
    """从标签文件中自动检测类别"""
    labels_dir = DATASET_DIR / "labels" / "train"
    class_ids = set()
    
    if not labels_dir.exists():
        raise FileNotFoundError(f"未找到标签目录: {labels_dir}")
    
    for txt_file in labels_dir.glob("*.txt"):
        try:
            with open(txt_file, 'r') as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.split()[0])
                        class_ids.add(class_id)
        except Exception as e:
            print(f"读取 {txt_file} 时出错: {e}")
    
    # 生成类别名称
    max_id = max(class_ids) if class_ids else 0
    classes = [f"class_{i}" for i in range(max_id + 1)]
    
    return classes


def create_data_yaml(classes, output_path="data.yaml"):
    """创建 YOLO 数据配置文件"""
    data_config = {
        'path': str(DATASET_DIR.absolute()),
        'train': 'images/train',
        'val': 'images/valid',
        'test': '',  # 可选
        'nc': len(classes),
        'names': classes
    }
    
    yaml_path = WORKSPACE_DIR / output_path
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✓ 数据配置已保存到: {yaml_path}")
    return yaml_path


def check_dataset_structure():
    """检查数据集结构"""
    print("\n" + "="*60)
    print("检查数据集结构...")
    print("="*60)
    
    required_dirs = [
        DATASET_DIR / "images" / "train",
        DATASET_DIR / "images" / "valid",
        DATASET_DIR / "labels" / "train",
        DATASET_DIR / "labels" / "valid"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        exists = dir_path.exists()
        status = "✓" if exists else "✗"
        print(f"{status} {dir_path.relative_to(WORKSPACE_DIR)}")
        
        if exists:
            # 统计文件数量
            image_exts = ['.jpg', '.jpeg', '.png', '.bmp']
            if 'images' in str(dir_path):
                files = [f for f in dir_path.iterdir() if f.suffix.lower() in image_exts]
            else:
                files = list(dir_path.glob("*.txt"))
            print(f"  └─ 包含 {len(files)} 个文件")
        else:
            all_exist = False
    
    if not all_exist:
        raise FileNotFoundError("数据集结构不完整，请检查文件夹结构！")
    
    print("="*60 + "\n")


def apply_augmentations(image, enable_blur=True, enable_rotation=True, enable_affine=True):
    """
    应用数据增强（学习自 main.py）
    
    Args:
        image: 输入图像
        enable_blur: 是否启用模糊
        enable_rotation: 是否启用旋转
        enable_affine: 是否启用仿射变换
    """
    h, w = image.shape[:2]
    
    # 1. 随机旋转
    if enable_rotation and random.random() > 0.5:
        angle = random.uniform(*ROTATION_ANGLE_RANGE)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    # 2. 随机仿射变换
    if enable_affine and random.random() > 0.5:
        distort_w = w * AFFINE_DISTORTION_RANGE
        distort_h = h * AFFINE_DISTORTION_RANGE
        
        src_points = np.float32([[0, 0], [w-1, 0], [0, h-1]])
        dst_points = np.float32([
            [random.uniform(0, distort_w), random.uniform(0, distort_h)],
            [random.uniform(w - distort_w, w-1), random.uniform(0, distort_h)],
            [random.uniform(0, distort_w), random.uniform(h - distort_h, h-1)]
        ])
        
        M_affine = cv2.getAffineTransform(src_points, dst_points)
        image = cv2.warpAffine(image, M_affine, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    # 3. 随机模糊
    if enable_blur and random.random() < BLUR_PROBABILITY:
        min_kernel, max_kernel = BLUR_KERNEL_RANGE
        if min_kernel % 2 == 0:
            min_kernel += 1
        if max_kernel % 2 == 0:
            max_kernel -= 1
        
        kernel_sizes = list(range(min_kernel, max_kernel + 1, 2))
        kernel_size = random.choice(kernel_sizes)
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    return image


def train_model():
    """训练 YOLOv8 模型"""
    print("\n" + "="*60)
    print("KFS 检测模型训练")
    print("="*60)
    
    # 1. 检查数据集结构
    check_dataset_structure()
    
    # 2. 读取类别
    classes = read_classes()
    
    # 3. 创建数据配置文件
    data_yaml = create_data_yaml(classes)
    
    # 4. 初始化模型
    print(f"\n加载预训练模型: {PRETRAINED_MODEL}")
    model = YOLO(PRETRAINED_MODEL)
    
    # 5. 配置训练参数
    train_args = {
        'data': str(data_yaml),
        'epochs': EPOCHS,
        'batch': BATCH_SIZE,
        'imgsz': IMAGE_SIZE,
        'device': DEVICE,
        'workers': WORKERS,
        'project': str(OUTPUT_DIR.parent),
        'name': OUTPUT_DIR.name,
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'AdamW',
        'verbose': True,
        'seed': 42,
        'deterministic': False,
        'single_cls': False,
        'rect': False,
        'cos_lr': True,
        'close_mosaic': 10,
        'resume': False,
        'amp': True,
        'fraction': 1.0,
        'profile': False,
        'freeze': None,
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 12.0,
        'kobj': 1.0,
        'label_smoothing': 0.0,
        'nbs': 64,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': ROTATION_ANGLE_RANGE[1],  # 使用配置的旋转角度
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
        'auto_augment': 'randaugment',
        'erasing': 0.4,
        'crop_fraction': 1.0,
    }
    
    print("\n" + "="*60)
    print("训练配置:")
    print("="*60)
    print(f"训练轮数: {EPOCHS}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"图像尺寸: {IMAGE_SIZE}×{IMAGE_SIZE}")
    print(f"设备: {'GPU ' + str(DEVICE) if isinstance(DEVICE, int) else DEVICE}")
    print(f"类别数量: {len(classes)}")
    print(f"数据路径: {DATASET_DIR}")
    print(f"旋转角度: {ROTATION_ANGLE_RANGE[0]}° ~ {ROTATION_ANGLE_RANGE[1]}°")
    print(f"模糊概率: {BLUR_PROBABILITY*100}%")
    print("="*60 + "\n")
    
    # 6. 开始训练
    try:
        results = model.train(**train_args)
        
        print("\n" + "="*60)
        print("✓ 训练完成！")
        print("="*60)
        print(f"模型保存路径: {OUTPUT_DIR}")
        print(f"最佳权重: {OUTPUT_DIR / 'weights' / 'best.pt'}")
        print(f"最后权重: {OUTPUT_DIR / 'weights' / 'last.pt'}")
        print("="*60 + "\n")
        
        return model, results
    
    except Exception as e:
        print(f"\n✗ 训练出错: {str(e)}")
        raise


def validate_model(model_path=None):
    """验证模型性能"""
    if model_path is None:
        model_path = OUTPUT_DIR / 'weights' / 'best.pt'
    
    if not Path(model_path).exists():
        print(f"未找到模型文件: {model_path}")
        return
    
    print("\n" + "="*60)
    print("验证模型性能")
    print("="*60)
    
    model = YOLO(str(model_path))
    data_yaml = WORKSPACE_DIR / "data.yaml"
    
    results = model.val(
        data=str(data_yaml),
        split='val',
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        device=DEVICE,
        workers=WORKERS,
        verbose=True
    )
    
    print("\n" + "="*60)
    print("验证结果:")
    print("="*60)
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    print("="*60 + "\n")
    
    return results


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="KFS 检测模型训练")
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'val', 'both'],
                       help='运行模式: train(训练), val(验证), both(训练+验证)')
    parser.add_argument('--model', type=str, default=None,
                       help='验证模式下的模型路径')
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数（覆盖默认配置）')
    parser.add_argument('--batch', type=int, default=None,
                       help='批次大小（覆盖默认配置）')
    parser.add_argument('--device', default=None,
                       help='设备（覆盖默认配置）')
    
    args = parser.parse_args()
    
    # 更新配置
    global EPOCHS, BATCH_SIZE, DEVICE
    if args.epochs:
        EPOCHS = args.epochs
    if args.batch:
        BATCH_SIZE = args.batch
    if args.device is not None:
        DEVICE = args.device if args.device == 'cpu' else int(args.device)
    
    # 执行对应模式
    if args.mode == 'train' or args.mode == 'both':
        model, results = train_model()
        
        if args.mode == 'both':
            print("\n开始验证模型...")
            validate_model()
    
    elif args.mode == 'val':
        validate_model(args.model)


if __name__ == "__main__":
    main()
