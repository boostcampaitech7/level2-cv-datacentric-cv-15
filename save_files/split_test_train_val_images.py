import os
import shutil

# 이미지가 혼합된 폴더 경로
source_folder = '/data/ephemeral/cord-v2/eng_receipt'  # 이미지가 있는 폴더 경로로 변경하세요

# 목적지 폴더 경로 설정
dest_folders = {
    "test": os.path.join(source_folder, "test"),
    "train": os.path.join(source_folder, "train"),
    "validation": os.path.join(source_folder, "validation")
}

# 목적지 폴더가 없으면 생성
for folder in dest_folders.values():
    os.makedirs(folder, exist_ok=True)

# 파일 이동 작업
for filename in os.listdir(source_folder):
    # 파일이 이미지인지 확인
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # 파일 이름에 따라 목적지 폴더 선택
        if filename.startswith("test_"):
            dest_path = dest_folders["test"]
        elif filename.startswith("train_"):
            dest_path = dest_folders["train"]
        elif filename.startswith("validation_"):
            dest_path = dest_folders["validation"]
        else:
            print(f"[WARNING] {filename}은(는) 알 수 없는 접두사를 가지고 있어 스킵되었습니다.")
            continue
        
        # 파일 이동
        src_path = os.path.join(source_folder, filename)
        shutil.move(src_path, dest_path)
        print(f"[INFO] {filename}을(를) {dest_path}로 이동했습니다.")

print("이미지 분류 및 이동 작업이 완료되었습니다.")
