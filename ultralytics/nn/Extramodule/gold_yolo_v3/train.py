# 导入yolo模型
from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    model = YOLO('yolov8n_gold_yolo_neck_v3.yaml')  # load a pretrained model (recommended for training)
    # Train the model
    results = model.train(data='coco128.yaml', epochs=20, imgsz=640, deterministic=False)