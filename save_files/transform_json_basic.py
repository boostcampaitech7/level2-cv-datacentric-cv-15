import json
import os

# Paths to input and output JSON files
input_file = '/data/ephemeral/cord-v2/data_local/validation-00000-of-00001-cc3c5779fe22e8ca.json'
output_file = '/data/ephemeral/cord-v2/eng_receipt/ufo/validation4.json'

# Ensure output directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Load input data
with open(input_file, 'r', encoding='utf-8') as infile:
    input_data = json.load(infile)

# Initialize output data structure
converted_data = {"images": {}}

# Iterate over each entry in input data
for item in input_data:
    # Parse 'ground_truth' and extract information if 'valid_line' key exists
    ground_truth = json.loads(item['ground_truth'])
    image_id = ground_truth['meta']['image_id']
    valid_lines = ground_truth.get('valid_line', [])
    
    # Skip images with no 'valid_line' data
    if not valid_lines:
        print(f"[WARNING] Image {image_id} has no valid_line entries.")
        continue
    
    # Prepare words data for each image
    words_data = {}
    word_idx = 0
    for line in valid_lines:
        for word in line['words']:
            # Extract text and bounding box points for each individual word
            text = word['text']
            points = [
                [word['quad']['x1'], word['quad']['y1']],
                [word['quad']['x2'], word['quad']['y2']],
                [word['quad']['x3'], word['quad']['y3']],
                [word['quad']['x4'], word['quad']['y4']]
            ]
            
            # Add each individual word with its points to words_data
            words_data[f"{word_idx:04d}"] = {
                "transcription": text,
                "points": points
            }
            word_idx += 1
    
    # Add to images dictionary
    converted_data["images"][f"validation_{image_id}.jpg"] = {
        "paragraphs": {},  # Empty as no 'paragraph' info provided in the input
        "words": words_data
    }
    print(f"[INFO] Processed image {image_id} with {len(words_data)} words.")

# Save converted data to output JSON file
with open(output_file, 'w', encoding='utf-8') as outfile:
    json.dump(converted_data, outfile, ensure_ascii=False, indent=4)

print(f"File transformation completed. Output saved to {output_file}.")
