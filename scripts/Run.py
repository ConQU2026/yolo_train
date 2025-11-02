import cv2
from ultralytics import YOLO

def yolov11_camera_inference(model_path="F:/ultralytics-main/models/exp6.pt", camera_id=0, conf_threshold=0.60):
    """
    使用YOLOv11从摄像头实时检测目标
    
    Args:
        model_path: YOLOv11模型权重路径（默认使用nano版预训练模型）
        camera_id: 摄像头设备ID（默认0为内置摄像头）
        conf_threshold: 置信度阈值（过滤低置信度检测结果）
    """
    # 1. 加载YOLOv11模型
    model = YOLO(model_path)
    print(f"已加载模型: {model_path}")
    print(f"置信度阈值: {conf_threshold}")

    # 2. 打开摄像头
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"无法打开摄像头 (ID: {camera_id})，请检查设备是否连接正常")
        return

    # 3. 实时读取并处理视频流
    print("开始实时检测... (按 'q' 键退出)")
    while True:
        # 读取一帧图像
        ret, frame = cap.read()
        if not ret:
            print("无法获取视频帧，可能摄像头已断开")
            break

        # 4. 执行推理（仅检测，不保存）
        results = model(frame, conf=conf_threshold, save=False)

        # 5. 在帧上绘制检测结果（边界框、类别、置信度）
        annotated_frame = results[0].plot()  # 内置绘制函数

        # 6. 显示处理后的帧
        cv2.imshow("YOLOv11 Camera Inference", annotated_frame)

        # 按 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 7. 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("检测结束，已释放资源")

if __name__ == "__main__":
    # 可根据需要修改参数：
    # - 模型可选：yolov11n.pt（轻量）、yolov11s.pt、yolov11m.pt、yolov11l.pt、yolov11x.pt（高精度）
    # - camera_id：多摄像头时可改为1、2等
    # - conf_threshold：提高阈值（如0.5）可减少误检，降低阈值（如0.1）可提高检出率
    yolov11_camera_inference(
        model_path="F:/ultralytics-main/models/exp6.pt",
        camera_id=0,
        conf_threshold=0.60
    )