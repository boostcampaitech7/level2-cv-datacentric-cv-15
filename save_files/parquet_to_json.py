import pandas as pd
import json

# Parquet 파일 경로
parquet_file_path = '/data/ephemeral/cord-v2/data_local/validation-00000-of-00001-cc3c5779fe22e8ca.parquet'

# Parquet 파일 읽기
df = pd.read_parquet(parquet_file_path)

# 필터링할 필드 리스트 정의
fields_to_keep = ['ground_truth', 'meta', 'valid_line']

# 필요한 필드만 포함된 데이터를 저장할 리스트 생성
filtered_data = []
for record in df.to_dict(orient='records'):
    filtered_record = {key: record[key] for key in fields_to_keep if key in record}
    filtered_data.append(filtered_record)

# JSON 파일 경로 설정
json_file_path = '/data/ephemeral/cord-v2/data_local/validation-00000-of-00001-cc3c5779fe22e8ca.json'

# JSON 파일로 저장
with open(json_file_path, 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=4)

print(f"Filtered JSON file has been saved to {json_file_path}.")
