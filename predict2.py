import cv2
from ultralytics import YOLO

model_path = 'C:\\Users\\halil\\Desktop\\OpenCV_python\\customdetection2\\yolov8m.pt'
model = YOLO(model_path)

threshold = 0.1

cap = cv2.VideoCapture(0)  # Bilgisayar kamerasını başlat

while True:
    ret, frame = cap.read()  # Görüntü yakalama

    if not ret:
        break

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 3, cv2.LINE_AA)
           
            #Merkez koordinatlarını hesapla
            center_x = int((x1+x2) / 2)
            center_y = int((y1+y2) / 2)
            
            #Kırmızı nokta çizelim
            cv2.circle(frame, (center_x, center_y), 5, (0,0,255), -1)
            
            
            #Koordinatları ekrana bastır
            print(f"Nesne: {results.names[int(class_id)]}, Koordinatlar: x1={x1}, y1={y1}, x2={x2}, y2={y2}, Merkez: ({center_x}, {center_y})")
    cv2.imshow('Real-time Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


