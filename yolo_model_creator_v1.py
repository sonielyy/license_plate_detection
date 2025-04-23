from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("runs/detect/train/weights/best.pt")
    results = model.train(data="config.yaml", epochs=100, imgsz=640, batch=8, device="0")