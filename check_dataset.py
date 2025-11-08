import os

# 🔹 여기에 확인할 폴더 경로 입력
folder_path = "/Users/sangwanjeon/Documents/GitHub/Car_Detect_with_AI/bottom_signal/dataset/images/train"

# 🔹 이미지 확장자 목록 (필요시 더 추가 가능)
img_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

# 🔹 폴더 안에서 이미지 파일만 세기
count = sum(1 for f in os.listdir(folder_path) if f.lower().endswith(img_ext))

print(f"📸 이미지 파일 개수: {count}개")
