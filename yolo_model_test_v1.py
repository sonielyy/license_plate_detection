from ultralytics import YOLO
import os
import numpy as np
import pandas as pd
import cv2

model_path = "runs/detect/train2/weights/best.pt"
model = YOLO(model_path)

image_path = "test_files/test_images/test6.jpeg"
img = cv2.imread(image_path)

results = model.predict(source=image_path, save=False, conf = 0.3)

boxes = results[0].boxes

for box in boxes:
    xyxy = box.xyxy[0].cpu().numpy().astype(int)
    x1, y1, x2, y2 = xyxy
    conf = float(box.conf.cpu().numpy()[0])
    cls_idx = int(box.cls.cpu().numpy()[0])
    class_label = model.names[cls_idx]
    label_text = f"{class_label} {conf:.2f}"
    print(label_text)

    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 8, cv2.LINE_AA)

# Görseli yeniden boyutlandır (örnek olarak %50 küçültüyoruz)
scale_percent = 20  # %50'ye indir
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

# Yeniden boyutlandırılmış görseli göster
cv2.imshow("Tahmin Sonucu (Küçültülmüş)", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
