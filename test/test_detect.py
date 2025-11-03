import cv2
from ultralytics import YOLO
from pathlib import Path
import os


def detect_with_image(img_path=None, confidence_thres = 0.25):
    
    path = Path(__file__).parent.parent.resolve()
    model_path = os.path.join(path, 'result','best.pt')
    
    #设置置信度
    model = YOLO(model_path)
    
    # 进行目标检测
    results = model(img_path, conf = confidence_thres)
    
    # 显示检测结果
    annotated_img = results[0].plot()
    cv2.imshow('result', annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 保存结果
    output_path = os.path.join(path,'test', 'output')
    os.makedirs(output_path, exist_ok=True)
    output_img_path = os.path.join(output_path, 'detected_image.jpg')
    cv2.imwrite(output_img_path, annotated_img)
    
        
    
    
    
def detect_with_camera():
    path = Path(__file__).parent.parent.resolve()

    model_path = os.path.join(path, 'result','best.pt')
    model = YOLO(model_path)
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 进行目标检测
        results = model(frame)
        
        # 显示检测结果
        annotated_frame = results[0].plot()
        cv2.imshow('YOLOv8 Detection', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
def detect_with_video(video_path):
    path = Path(__file__).parent.parent.resolve()
    # 加载预训练的 YOLOv8 模型
    model_path = os.path.join(path, 'result','best.pt')
    model = YOLO(model_path)
    
    output_path = os.path.join(path,'test', 'output')

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20.0, 
                        (int(cap.get(3)), int(cap.get(4))))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 进行目标检测
        results = model(frame)
        
        # 显示检测结果
        annotated_frame = results[0].plot()
        out.write(annotated_frame)
        # cv2.imshow('YOLOv8 Detection', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    
    
def main():
    path = Path(__file__).parent.resolve()
    video_path = os.path.join(path, 'video', 'test.mp4')
    img_path = os.path.join(path, 'image', 'test3.jpg')
    
    
    # 使用图像进行检测
    detect_with_image(img_path=img_path, confidence_thres=0.25)
    
    # 使用摄像头进行检测
    # detect_with_camera()
    
    # 使用视频文件进行检测
    # detect_with_video(video_path=video_path)
    
if __name__ == "__main__":
    main()