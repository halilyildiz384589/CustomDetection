# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 12:04:28 2023

@author: halil

"""
import os
import time
import cv2
import numpy as np
from ultralytics import YOLO

videos_dir = os.path.join('', 'C:\\Users\\halil\\Desktop\\OpenCV_python\\customdetection2\\videos') #video dosyaları nerede

video_path = os.path.join(videos_dir, 'building_workers.mp4') #giriş videosu adı
video_path_out = '{}_out.mp4'.format(video_path) #çıkış videosu adını oluşturduk

cap = cv2.VideoCapture(video_path)  #video dosyasını aldım
ret, frame = cap.read() #dosyanın çerçevelerini belirledim
H, W, _ = frame.shape #çerçeve genişliğini belriledik 
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H)) #çıkış videosunu yazdırdık

model_path = os.path.join('.', 'C:\\Users\\halil\\Desktop\\OpenCV_python\\customdetection2\\yolov8m.pt') #YOLO modelinin olduğu dosya yolunu belirttik

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.1 #☺0.1 eşik değerinin üzerindekileri tara ve yazdır

while ret:

    results = model(frame)[0] #0 ile yoloya tek çıktı ver diyoruz

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA) #dörtgen çiz etiketi bas

    out.write(frame) #çerçeveye yaz
    ret, frame = cap.read() #kenar ve köşeleri oku

#her şeyi kapat
cap.release()
out.release()
cv2.destroyAllWindows()