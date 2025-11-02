import cv2
import numpy as np
import os
import random
from pathlib import Path

# ---------------------- 可配置参数（用户可根据需求修改） ----------------------
# 1. 路径与基础配置
CLASSES_FILE = "classes.txt"       # 类别文件路径
KFS_FOLDER = "KFS"                 # KFS主文件夹（包含B和R子文件夹）
BACKGROUND_FOLDER = "background"   # 背景文件夹路径
OUTPUT_DIR = "dataset"             # 输出文件夹
BACKGROUND_SIZE = (640, 640)       # 固定背景尺寸（宽×高）
DATA_NUMBERS = 200                 # 通用样本数量（供多个类别复用）

# 2. 类别与样本数量配置
classes_count = {
    "R_R1": 100,
    "B_R1": 100,
    "T03": DATA_NUMBERS,
    "T04": DATA_NUMBERS,
    "T05": DATA_NUMBERS,
    "T06": DATA_NUMBERS,
    "T07": DATA_NUMBERS,
    "T08": DATA_NUMBERS,
    "T09": DATA_NUMBERS,
    "T10": DATA_NUMBERS,
    "T11": DATA_NUMBERS,
    "T12": DATA_NUMBERS,
    "T13": DATA_NUMBERS,
    "T14": DATA_NUMBERS,
    "T15": DATA_NUMBERS,
    "T16": DATA_NUMBERS,
    "T17": DATA_NUMBERS,
    "F18": DATA_NUMBERS,
    "F19": DATA_NUMBERS,
    "F20": DATA_NUMBERS,
    "F21": DATA_NUMBERS,
    "F22": DATA_NUMBERS,
    "F23": DATA_NUMBERS,
    "F24": DATA_NUMBERS,
    "F25": DATA_NUMBERS,
    "F26": DATA_NUMBERS,
    "F27": DATA_NUMBERS,
    "F28": DATA_NUMBERS,
    "F29": DATA_NUMBERS,
    "F30": DATA_NUMBERS,
    "F31": DATA_NUMBERS,
    "F32": DATA_NUMBERS
}

# 3. B/R文件夹样本比例
br_ratio = (5, 5)

# 4. 数据增强参数（含新增模糊配置）
ROTATION_ANGLE_RANGE = (-60, 60)   # 旋转角度范围（度）
AFFINE_DISTORTION_RANGE = 0.1      # 仿射扭曲幅度（0~0.5）
SCALE_MIN = 0.1                    # 最小缩放比例
# 新增！模糊参数（用户可控）
BLUR_PROBABILITY = 0.9             # 模糊触发概率（0~1，如0.5表示50%概率模糊）
BLUR_KERNEL_RANGE = (5, 45)        # 模糊核大小范围（必须为奇数，核越大模糊越强）
# --------------------------------------------------------------------------


def read_classes(classes_file=CLASSES_FILE):
    """读取classes.txt，返回类别列表"""
    try:
        with open(classes_file, 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f if line.strip()]
        for key_cls in ["B_R1", "R_R1"]:
            if key_cls not in classes:
                print(f"警告：classes.txt中未找到 {key_cls}，请确认文件内容")
        return classes
    except FileNotFoundError:
        raise FileNotFoundError(f"未找到类别文件: {classes_file}")


def get_random_image(folder):
    """从指定文件夹随机选择一张图片"""
    if not os.path.exists(folder):
        raise FileNotFoundError(f"文件夹不存在: {folder}")
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    images = []
    try:
        for root, dirs, files in os.walk(folder):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    images.append(os.path.join(root, file))
    except PermissionError:
        print(f"无权限访问文件夹: {folder}，跳过")
    except Exception as e:
        print(f"遍历文件夹出错: {str(e)}")
    
    if not images:
        raise ValueError(f"文件夹中未找到图片: {folder}")
    return random.choice(images)


def extract_class_from_filename(filename, classes):
    """提取带下划线的类别（如B_R1/R_R1）"""
    base_name = os.path.splitext(filename)[0]
    matched_cls = None
    max_cls_len = 0
    for cls in classes:
        if base_name.startswith(cls) and (len(base_name) == len(cls) or base_name[len(cls)] == '_'):
            if len(cls) > max_cls_len:
                max_cls_len = len(cls)
                matched_cls = cls
    if matched_cls:
        return matched_cls
    else:
        raise ValueError(f"文件名 {filename} 无法匹配到classes中的任何类别（classes包含：{classes}）")


def add_random_blur(image, blur_prob, kernel_range):
    """新增！对图像应用随机高斯模糊（基于用户配置）"""
    # 1. 按概率决定是否应用模糊
    if random.random() > blur_prob:
        return image  # 不应用模糊，直接返回原图
    
    # 2. 处理模糊核大小（确保为奇数，且在用户指定范围内）
    min_kernel, max_kernel = kernel_range
    # 若用户输入非奇数，自动调整为最近的奇数
    if min_kernel % 2 == 0:
        min_kernel += 1
    if max_kernel % 2 == 0:
        max_kernel -= 1
    # 确保核大小范围有效
    if min_kernel > max_kernel:
        min_kernel, max_kernel = max_kernel, min_kernel
    if min_kernel < 3:
        min_kernel = 3  # 最小核大小为3（避免过度模糊）
    
    # 3. 随机选择核大小（从奇数列表中选择）
    kernel_sizes = list(range(min_kernel, max_kernel + 1, 2))
    kernel_size = random.choice(kernel_sizes)
    
    # 4. 应用高斯模糊（sigmaX自动计算，确保模糊效果自然）
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    # 5. 若原图有alpha通道（透明背景），保留alpha通道
    if image.shape[2] == 4:
        # 提取原图alpha通道，合并到模糊后的图像
        alpha = image[:, :, 3]
        blurred_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2BGRA)
        blurred_image[:, :, 3] = alpha
    
    return blurred_image


def random_transform(image, max_size=BACKGROUND_SIZE):
    """对图像进行随机变换（含新增的模糊处理）"""
    if len(image.shape) < 2:
        raise ValueError("图像尺寸异常，无法进行变换")
    rows, cols = image.shape[:2]
    max_w, max_h = max_size
    
    # 1. 随机缩放
    scale_w = max_w / cols
    scale_h = max_h / rows
    max_allowed_scale = min(scale_w, scale_h)
    scale = random.uniform(SCALE_MIN, max_allowed_scale)
    scaled_cols = int(cols * scale)
    scaled_rows = int(rows * scale)
    scaled = cv2.resize(image, (scaled_cols, scaled_rows), interpolation=cv2.INTER_AREA)
    
    # 2. 随机旋转
    angle = random.uniform(*ROTATION_ANGLE_RANGE)
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
    distort_w = cols_rot * AFFINE_DISTORTION_RANGE
    distort_h = rows_rot * AFFINE_DISTORTION_RANGE
    dst_points = np.float32([
        [random.uniform(0, distort_w), random.uniform(0, distort_h)],
        [random.uniform(cols_rot - distort_w, cols_rot-1), random.uniform(0, distort_h)],
        [random.uniform(0, distort_w), random.uniform(rows_rot - distort_h, rows_rot-1)]
    ])
    M_affine = cv2.getAffineTransform(src_points, dst_points)
    transformed = cv2.warpAffine(rotated, M_affine, (cols_rot, rows_rot))
    
    # 新增！4. 随机模糊（应用用户配置的模糊参数）
    transformed = add_random_blur(transformed, BLUR_PROBABILITY, BLUR_KERNEL_RANGE)
    
    return transformed


def overlay_image(background, foreground, x, y):
    """将前景叠加到背景指定位置"""
    if foreground.shape[2] < 4:
        foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2BGRA)
    
    h, w = foreground.shape[:2]
    bg_h, bg_w = background.shape[:2]
    
    if x < 0:
        foreground = foreground[:, -x:]
        w = foreground.shape[1]
        x = 0
    if y < 0:
        foreground = foreground[-y:, :]
        h = foreground.shape[0]
        y = 0
    if x + w > bg_w:
        w = bg_w - x
        foreground = foreground[:, :w]
    if y + h > bg_h:
        h = bg_h - y
        foreground = foreground[:h, :]
    
    alpha = foreground[:, :, 3] / 255.0
    alpha = np.stack([alpha, alpha, alpha], axis=2)
    foreground_rgb = foreground[:, :, :3]
    background_roi = background[y:y+h, x:x+w]
    blended = (1 - alpha) * background_roi + alpha * foreground_rgb
    background[y:y+h, x:x+w] = blended.astype(np.uint8)
    
    return background, (x, y, w, h)


def generate_yolo_label(classes, class_name, bbox, background_shape):
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
    classes = read_classes()
    for cls in classes_count:
        if cls not in classes:
            raise ValueError(f"classes_count中的类别 {cls} 不在classes.txt中，请检查配置")
    
    total_samples = sum(classes_count.values())
    br_sum = sum(br_ratio)
    if br_sum == 0:
        raise ValueError("B/R比例不能全为0，请调整br_ratio参数")
    b_total = (total_samples * br_ratio[0]) // br_sum
    r_total = total_samples - b_total
    
    # 打印任务信息（含模糊配置）
    print(f"=== 任务配置 ===")
    print(f"总样本数: {total_samples} | B文件夹: {b_total}个 | R文件夹: {r_total}个")
    print(f"背景尺寸: {BACKGROUND_SIZE[0]}×{BACKGROUND_SIZE[1]} | 旋转角度: {ROTATION_ANGLE_RANGE}度")
    print(f"仿射扭曲幅度: {AFFINE_DISTORTION_RANGE} | 最小缩放比例: {SCALE_MIN}")
    print(f"模糊配置: 触发概率{BLUR_PROBABILITY*100}% | 核大小范围{BLUR_KERNEL_RANGE}（奇数）")
    print(f"输出路径: {os.path.abspath(OUTPUT_DIR)}")
    print(f"===============\n")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    generated = {cls: 0 for cls in classes_count}
    b_generated = 0
    r_generated = 0
    sample_idx = 0
    
    while sum(generated.values()) < total_samples:
        try:
            
            # 选择B/R文件夹
            if (b_generated < b_total) and (r_generated >= r_total or random.random() < br_ratio[0]/br_sum):
                selected_folder = os.path.join(KFS_FOLDER, "B")
                folder_type = "B"
            else:
                selected_folder = os.path.join(KFS_FOLDER, "R")
                folder_type = "R"
            
            # 选择前景图
            foreground_path = get_random_image(selected_folder)
            foreground = cv2.imread(foreground_path, cv2.IMREAD_UNCHANGED)
            if foreground is None or foreground.size == 0:
                print(f"跳过无效图片: {os.path.basename(foreground_path)}")
                continue
            
            # 提取类别
            filename = os.path.basename(foreground_path)
            class_name = extract_class_from_filename(filename, classes)
            if generated[class_name] >= classes_count[class_name]:
                continue
            
            # 图像变换（含模糊）
            try:
                transformed_fg = random_transform(foreground)
            except Exception as e:
                print(f"图像变换失败 {filename}: {str(e)}，跳过")
                continue
            
            # 选择并调整背景
            background_path = get_random_image(BACKGROUND_FOLDER)
            background = cv2.imread(background_path)
            if background is None or background.size == 0:
                print(f"跳过无效背景: {os.path.basename(background_path)}")
                continue
            background = cv2.resize(background, BACKGROUND_SIZE, interpolation=cv2.INTER_AREA)
            bg_h, bg_w = background.shape[:2]
            
            # 放置前景
            fg_h, fg_w = transformed_fg.shape[:2]
            x = random.randint(0, bg_w - fg_w)
            y = random.randint(0, bg_h - fg_h)
            
            # 叠加前景
            result, bbox = overlay_image(background.copy(), transformed_fg, x, y)
            if bbox is None:
                print(f"前景 {filename} 无法放置，重新尝试")
                continue
            
            # 保存结果
            output_img = os.path.join(OUTPUT_DIR, f"sample_{sample_idx}.jpg")
            output_txt = os.path.join(OUTPUT_DIR, f"sample_{sample_idx}.txt")
            cv2.imwrite(output_img, result)
            yolo_label = generate_yolo_label(classes, class_name, bbox, background.shape)
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
            print(f"进度: {progress:.1f}% | 已生成 {sample_idx}/{total_samples} 个 | 类别: {class_name} | 来源: {folder_type}", end='\r')
        
        except Exception as e:
            print(f"\n处理出错: {str(e)}，继续尝试下一个样本...")
    
    print(f"\n\n所有样本生成完成！共生成 {sample_idx} 个样本，保存于: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()