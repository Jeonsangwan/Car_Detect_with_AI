from ultralytics import YOLO
import cv2, time

model = YOLO("/Users/sangwanjeon/Documents/GitHub/Car_Detect_with_AI/runs/detect/car-only/weights/best.pt")

cap = cv2.VideoCapture(0)

REAL_DIST_M = 1.0
LINE_TOP = 200
LINE_BOTTOM = 240
last_cross_time = {}
last_speed = None

def check_and_measure(car_id, cy):
    now = time.time()

    # 윗선 근처 들어왔는지 찍어보자
    # print(f"car {car_id} cy={cy}")  # 필요하면 켜기

    if LINE_TOP < cy < LINE_BOTTOM:
        if car_id not in last_cross_time:
            last_cross_time[car_id] = now
            print(f"[{car_id}] first line at {now:.3f}")

    if cy >= LINE_BOTTOM and car_id in last_cross_time:
        t1 = last_cross_time.pop(car_id)
        dt = now - t1
        if dt > 0:
            speed_kmh = (REAL_DIST_M / dt) * 3.6
            return speed_kmh
    return None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    results = model(frame)
    boxes = results[0].boxes.xyxy
    classes = results[0].boxes.cls

    for i, box in enumerate(boxes):
        cls_id = int(classes[i])
        if cls_id == 0:  # car만
            x1, y1, x2, y2 = box
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # 디버그: 지금 이 차의 y위치가 몇인지 찍기
            # 이게 200~240 사이로 안 오면 속도 안 나옴
            print(f"detected car {i} at cy={cy}")

            speed = check_and_measure(i, cy)
            if speed is not None:
                last_speed = speed
                print(f"🚗 speed = {speed:.2f} km/h")  # ← 네가 보고 싶은 줄

            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    # 선 그리기
    cv2.line(frame, (0, LINE_TOP), (w, LINE_TOP), (255, 0, 255), 2)
    cv2.line(frame, (0, LINE_BOTTOM), (w, LINE_BOTTOM), (255, 0, 255), 2)

    # 화면에도 속도 표시
    if last_speed is not None:
        cv2.putText(frame, f"speed: {last_speed:.1f} km/h", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "speed: -- km/h", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Speed Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()