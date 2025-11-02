import cv2
import numpy as np
import os
import random
from pathlib import Path
import logging
import re

from parse_config import ParseConfig
from image_processor import ImageProcessor

from ultralytics import YOLO

def setup_logging(level):
    logger = logging.getLogger()
    logger.setLevel(level)
    return logger

def train():
    path = Path(__file__).parent.parent.resolve()
    model_path = os.path.join(path, 'models', 'yolov8s.pt')
    data_path = os.path.join(path, 'config', 'data.yaml')
    model = YOLO(model_path)
    model.train(data=data_path, epochs=200, imgsz=640, batch=16, name="kfs_detect_model", workers=8, device=0)

def main():
    path = Path(__file__).parent.resolve()
    logger = setup_logging(logging.INFO)

    train()

if __name__ == "__main__":
    main()