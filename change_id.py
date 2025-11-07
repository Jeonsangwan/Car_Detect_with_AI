import glob, os

label_dir = "/Users/sangwanjeon/Documents/GitHub/Car_Detect_with_AI/bottom_signal/dataset/labels/train"

for path in glob.glob(os.path.join(label_dir, "*.txt")):
    with open(path) as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if parts:
            parts[0] = "0"  # car → class 0
            new_lines.append(" ".join(parts) + "\n")
    with open(path, "w") as f:
        f.writelines(new_lines)

print("✅ 모든 라벨을 class_id = 0 (car)으로 변경했습니다.")
