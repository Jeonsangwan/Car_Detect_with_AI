from ultralytics import YOLO

# 1) 사전학습 모델 불러오기
model = YOLO("yolov8n.pt")

# 2) 내 데이터로 학습
model.train(
    data="/Users/sangwanjeon/Documents/GitHub/Car_Detect_with_AI/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="car-only",
    # --- 여기부터 증강 관련 ---
    flipud=0.0,      # 위아래 뒤집기는 보통 0 (도로는 위아래 바뀌면 이상함)
    fliplr=0.5,      # 좌우 대칭 50% 확률
    hsv_h=0.015,     # 색상 살짝 변형
    hsv_s=0.7,       # 채도
    hsv_v=0.4,       # 밝기
    scale=0.5,       # 크기 랜덤
    translate=0.1,   # 위치 살짝 이동
    mosaic=1.0,      # 4장 섞는 증강 (데이터 적으면 켜두는 게 좋음)
)