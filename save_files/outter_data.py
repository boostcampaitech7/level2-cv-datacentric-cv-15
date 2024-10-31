from datasets import load_dataset

ds = load_dataset("naver-clova-ix/cord-v2")

# or load the separate splits if the dataset has train/validation/test splits
train_dataset = load_dataset("jinhyoung0127/receipt_data", split="train")
valid_dataset = load_dataset("jinhyoung0127/receipt_data", split="validation")
test_dataset  = load_dataset("jinhyoung0127/receipt_data", split="test")

receipt_data.push_to_hub("jinhyoung0127/receipt_data")