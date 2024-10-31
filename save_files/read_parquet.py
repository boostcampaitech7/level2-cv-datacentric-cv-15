from datasets import load_dataset

# Parquet 파일 경로 지정
parquet_file_path = '/data/ephemeral/cord-v2/data_local/train-00000-of-00004-b4aaeceff1d90ecb.parquet'

# Parquet 파일을 dataset으로 불러오기
dataset = load_dataset('parquet', data_files=parquet_file_path)

# 데이터 확인
print(dataset)  # 전체 데이터셋 정보 출력
print(dataset['train'][0])  # 첫 번째 레코드 확인
