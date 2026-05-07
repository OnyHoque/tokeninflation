import json

file_path = "dataset/general_15K.json"


out_path = f"dataset/general_15K_adv.json"

with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

new_data = []
for item in data:
    if item["ground_truth"] > 400 and item["ground_truth"] < 600:
        item["ground_truth"] = item["ground_truth"]*5
    new_data.append(item)
        


with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(new_data, f, indent=4, ensure_ascii=False)

print(f" variant done, length of dataset: {len(new_data)}")
