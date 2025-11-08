from ultralytics import YOLO
import cv2

# 1️⃣ 방금 학습이 끝난 모델 불러오기
model = YOLO("/Users/sangwanjeon/Documents/GitHub/Car_Detect_with_AI/runs/detect/car-only3/weights/best.pt")

# 2️⃣ 테스트할 이미지 읽기
img = cv2.imread("test.jpg")  # 실제 자동차가 있는 이미지 경로

# 3️⃣ 모델로 예측 실행
results = model(img)

# 4️⃣ 결과 시각화 (자동차 주변에 박스 그려서 새 창으로 띄움)
results[0].show()
