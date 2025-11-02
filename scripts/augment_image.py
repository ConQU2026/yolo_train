import cv2
import numpy as np
from PIL import Image, ImageOps
import random, os

input_path = r"E:\图像处理\image_augment_project\input\1.png"
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

def perspective_transform(img):
    """应用随机透视投影效果"""
    h, w = img.shape[:2]
    # 原始四个角点
    src = np.float32([[0,0], [w,0], [w,h], [0,h]])

    # 随机偏移目标角点，模拟透视效果
    delta = w * random.uniform(0.1, 0.4)
    dst = np.float32([
        [random.uniform(0, delta), random.uniform(0, delta)],             # 左上
        [w - random.uniform(0, delta), random.uniform(0, delta)],         # 右上
        [w - random.uniform(0, delta/2), h - random.uniform(0, delta)],   # 右下
        [random.uniform(0, delta/2), h - random.uniform(0, delta)]        # 左下
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    transformed = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return transformed

for i in range(200):
    pil_img = Image.open(input_path).convert("RGBA")

    # 随机缩放
    scale = random.uniform(0.6, 1.4)
    new_size = (int(pil_img.width * scale), int(pil_img.height * scale))
    pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)

    # 随机旋转
    angle = random.uniform(0, 180)
    pil_img = pil_img.rotate(angle, expand=True)

    # 随机翻转
    if random.choice([True, False]):
        pil_img = ImageOps.mirror(pil_img)
    if random.choice([True, False]):
        pil_img = ImageOps.flip(pil_img)

    # 转为OpenCV格式
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGRA)

    # 加透视变换
    cv_img = perspective_transform(cv_img)

    # 转回PIL保存
    pil_result = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGBA))
    pil_result.save(os.path.join(output_dir, f"augmented_perspective_{i+1}.png"))

print("✅ 已生成带透视投影的图片，保存在 output_images 文件夹中。")
