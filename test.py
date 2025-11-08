from ultralytics import YOLO
import cv2, time

# --- YOLO 모델 불러오기 ---
model = YOLO("/Users/sangwanjeon/Documents/GitHub/Car_Detect_with_AI/runs/detect/car-only/weights/best.pt")

# --- 카메라 열기 ---
cap = cv2.VideoCapture(0)

# --- 실제 거리 (선 사이 10m) ---
REAL_DIST_M = 1.0
LINE_TOP = 500      # 프레임에서 100m 위치로 네가 잡은 y좌표
LINE_BOTTOM = 550   # 바로 아래쪽, 실제로 10m 차이
last_cross_time = {}  # 차량별 통과 시간 저장용

# 화면에 항상 보여줄 마지막 속도
last_speed = None

# --- 차량 속도 계산 함수 ---
def check_and_measure(car_id, cy):
    global last_cross_time
    now = time.time()

    # 첫 번째 선을 지날 때 시간 저장
    if LINE_TOP < cy < LINE_BOTTOM:
        if car_id not in last_cross_time:
            last_cross_time[car_id] = now

    # 두 번째 선을 통과하면 속도 계산
    if cy >= LINE_BOTTOM and car_id in last_cross_time:
        t1 = last_cross_time.pop(car_id)
        dt = now - t1
        if dt > 0:
            speed_kmh = (REAL_DIST_M / dt) * 3.6
            return speed_kmh
    return None

# --- 바닥 신호 제어 ---
def decide_signal(ped_signal, speed):
    if ped_signal != "GREEN":
        return "OFF"
    if speed is None:
        return "GREEN"
    if speed <= 30:
        return "GREEN"
    elif speed <= 50:
        return "YELLOW_BLINK_FAST"
    else:
        return "RED_BLINK_FAST"

# --- 시각화용 함수 ---
def draw_floor_signal(frame, mode):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = 10, h - 70, 200, h - 10

    color = (0, 255, 0)
    text = "GREEN"

    if mode == "YELLOW_BLINK_FAST":
        color, text = (0, 255, 255), "YELLOW BLINK"
    elif mode == "RED_BLINK_FAST":
        color, text = (0, 0, 255), "RED BLINK"
    elif mode == "OFF":
        color, text = (50, 50, 50), "OFF"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    cv2.putText(frame, text, (x1 + 10, y1 + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

# --- 메인 루프 ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    results = model(frame)
    boxes = results[0].boxes.xyxy
    classes = results[0].boxes.cls

    current_speed = None

    for i, box in enumerate(boxes):
        cls_id = int(classes[i])
        if cls_id == 0:  # car 클래스만
            x1, y1, x2, y2 = box
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            speed = check_and_measure(i, cy)  # 차량 ID(i) 기준 속도 측정

            # 이번 프레임에 속도가 계산됐으면 저장
            if speed is not None:
                current_speed = speed
                last_speed = speed  # 화면에 계속 보여줄 값 갱신
                print(f"[car {i}] speed = {speed:.2f} km/h")

            # 차 위에도 찍고 싶으면 이거 유지
            if last_speed is not None:
                cv2.putText(frame, f"{last_speed:.1f} km/h", (cx, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # 보행자 신호는 일단 항상 GREEN 가정
    mode = decide_signal("GREEN", last_speed)
    draw_floor_signal(frame, mode)

    # 기준선 시각화 (프레임 너비 w 사용)
    cv2.line(frame, (0, LINE_TOP), (w, LINE_TOP), (255, 0, 255), 2)
    cv2.line(frame, (0, LINE_BOTTOM), (w, LINE_BOTTOM), (255, 0, 255), 2)

    # 화면 왼쪽 위에 항상 속도 표시
    if last_speed is not None:
        txt = f"speed: {last_speed:.1f} km/h"
    else:
        txt = "speed: -- km/h"
    cv2.putText(frame, txt, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Speed Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

