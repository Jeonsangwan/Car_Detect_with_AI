from ultralytics import YOLO
import cv2

model = YOLO("/Users/sangwanjeon/Documents/GitHub/Car_Detect_with_AI/runs/detect/car-only3/weights/best.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.1)  # 일단 다 잡아
    r = results[0]

    for box, cls_id, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
        cls_id = int(cls_id)
        label = r.names[cls_id]

        # 👇 여기서 car만 남김
        if label != "car":
            continue

        x1, y1, x2, y2 = box.int().tolist()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("car only", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()