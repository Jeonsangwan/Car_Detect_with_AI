import os
import glob
import shutil

dataset_root = "/Users/sangwanjeon/Documents/GitHub/Car_Detect_with_AI/bottom_signal/dataset"
splits = ["train", "val"]  # 학습/검증 둘 다 확인

for split in splits:
    print(f"\n=== {split.upper()} 폴더 정리 중 ===")

    img_dir = os.path.join(dataset_root, "images", split)
    lbl_dir = os.path.join(dataset_root, "labels", split)

    unlabeled_out = os.path.join(img_dir, "unlabeled")  # 이미지만 있는 애들
    orphan_out = os.path.join(lbl_dir, "orphans")       # 라벨만 있는 애들
    os.makedirs(unlabeled_out, exist_ok=True)
    os.makedirs(orphan_out, exist_ok=True)

    img_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    imgs = []
    for ext in img_exts:
        imgs.extend(glob.glob(os.path.join(img_dir, f"*{ext}")))

    # 확장자 관계없이 이름만 모으기
    img_names = {os.path.splitext(os.path.basename(i))[0].lower(): i for i in imgs}
    lbl_names = {os.path.splitext(os.path.basename(l))[0].lower(): l
                 for l in glob.glob(os.path.join(lbl_dir, "*.txt"))}

    moved_img = moved_lbl = 0

    # 이미지 있는데 라벨 없는 파일
    for name, img_path in img_names.items():
        if name not in lbl_names:
            shutil.move(img_path, os.path.join(unlabeled_out, os.path.basename(img_path)))
            moved_img += 1

    # 라벨 있는데 이미지 없는 파일
    for name, lbl_path in lbl_names.items():
        if name not in img_names:
            shutil.move(lbl_path, os.path.join(orphan_out, os.path.basename(lbl_path)))
            moved_lbl += 1

    print(f"📸 라벨 없는 이미지: {moved_img}개 -> {unlabeled_out}")
    print(f"📝 이미지 없는 라벨: {moved_lbl}개 -> {orphan_out}")

print("\n✅ 데이터셋 정리 완료! 이제 check_dataset.py로 다시 검증해봐.")