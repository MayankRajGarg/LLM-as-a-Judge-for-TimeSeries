import json

def merge_datasets(synthetic_file, temp_file, output_file):
    # Step 1: Load the base series data into a dictionary for fast lookup
    print(f"Loading base data from {synthetic_file}...")
    series_data = {}
    with open(synthetic_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line.strip())
            # We map the ID to the actual series array
            series_data[data['id']] = data.get('series', [])
            
    print(f"Loaded {len(series_data)} unique time series.")

    # Step 2: Read the results, inject the series, and write to a new file
    print(f"Merging with {temp_file} and writing to {output_file}...")
    merged_count = 0
    missing_count = 0
    
    with open(temp_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            if not line.strip(): continue
            result_data = json.loads(line.strip())
            record_id = result_data.get('id')
            
            # Create a new dictionary to control the order of keys (id, model, series, question, explanation)
            merged_record = {
                "id": record_id,
                "model": result_data.get("model", "unknown")
            }
            
            # Inject the series data
            if record_id in series_data:
                merged_record["series"] = series_data[record_id]
                merged_count += 1
            else:
                merged_record["series"] = []
                print(f"Warning: ID '{record_id}' not found in {synthetic_file}")
                missing_count += 1
                
            # Add the question and explanation at the end
            merged_record["question"] = result_data.get("question", "")
            merged_record["explanation"] = result_data.get("explanation", "")
            
            # Write the merged JSON object as a new line
            f_out.write(json.dumps(merged_record) + '\n')

    print("\n--- Merge Complete ---")
    print(f"Successfully merged: {merged_count} records")
    if missing_count > 0:
        print(f"Failed to find series for: {missing_count} records")

if __name__ == "__main__":
    BASE_DATA_FILE = 'Data/synthetic.jsonl'
    RESULTS_FILE = 'Results/Exp1_collated.jsonl'
    OUTPUT_FILE = 'Results/Exp1_merged.jsonl'
    
    merge_datasets(BASE_DATA_FILE, RESULTS_FILE, OUTPUT_FILE)
