from ultralytics import YOLO

# 1. 기본 사전학습 모델 불러오기
model = YOLO("yolov8n.pt")   # 이미 방금 다운로드 된 그거

# 2. 우리 데이터로 학습시키기
model.train(
    data="/Users/sangwanjeon/Documents/GitHub/Car_Detect_with_AI/data.yaml",
    epochs=80,
    imgsz=640,
    batch=16,
    # 아래부터 증강 쪽
    fliplr=0.5,     # 좌우반전 50%
    scale=0.5,      # 크기 랜덤 스케일
    hsv_h=0.015,    # 색조 살짝
    hsv_s=0.7,      # 채도
    hsv_v=0.4,      # 밝기
    mosaic=1.0,     # 모자이크 켜둠 (사진 적으면 도움 됨)
    mixup=0.1       # 이미지 두 장 섞기 (너무 크면 이상해지니 0.1 정도)
)
