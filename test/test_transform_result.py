import cv2
import sys
import os
from pathlib import Path
import random

path = Path(__file__).parent.resolve()
sys.path.append(os.path.join(path.parent, 'src'))

def read_yolo_labels(label_path, W, H):
    """读取 YOLO 格式的标签并转换为像素坐标的矩形框 (x1, y1, x2, y2)"""
    boxes = []
    if not os.path.exists(label_path):
        print(f"Label file not found: {label_path}")
        return boxes
    
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            
            # class_id, x_center, y_center, w_norm, h_norm
            class_id = int(parts[0])
            xc, yc, w_norm, h_norm = map(float, parts[1:])
            
            # 坐标转换：归一化 -> 像素
            center_x = int(xc * W)
            center_y = int(yc * H)
            box_w = int(w_norm * W)
            box_h = int(h_norm * H)
            
            # 矩形框坐标 (x1, y1, x2, y2)
            x1 = center_x - box_w // 2
            y1 = center_y - box_h // 2
            x2 = center_x + box_w // 2
            y2 = center_y + box_h // 2
            
            boxes.append((class_id, x1, y1, x2, y2))
            
    return boxes

def main():
    path = Path(__file__).parent.resolve()
    
    target_dataset_path = os.path.join(path.parent, 'transformed_dataset')
    
    #随机从变换后的数据集中选择一张图片进行测试
    images_dir = os.path.join(target_dataset_path, 'images', 'train')
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print("No images found in the transformed dataset.")
        return
    random_image_file = random.choice(image_files)
    test_image_path = os.path.join(images_dir, random_image_file)
    
    # label path
    test_label_path = os.path.join(target_dataset_path, 'labels', 'train', os.path.splitext(random_image_file)[0] + '.txt')    
    #不使用ImageProcessor
    image = cv2.imread(test_image_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        print(f"Failed to load image: {test_image_path}")
        return
    
    #绘制识别框和标签
    H, W = image.shape[:2]
    boxes = read_yolo_labels(test_label_path, W, H)
    for class_id, x1, y1, x2, y2 in boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, str(class_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        
    cv2.imshow("Transformed Image with Labels", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    


if __name__ == "__main__":
    main()
