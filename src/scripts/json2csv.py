import json
import csv

json_file = "/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/meituan_hw/src/output/test/siglip_lr1e-4_bs16_8gpu_40epo_28k/test/results_final.json" # "/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/meituan_hw/src/output/test/siglip_lr1e-4_bs16_8gpu_20epo/test/results_final.json"
assert ".json" in json_file
csv_filename = json_file.replace(".json", ".csv")

root = "/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/meituan_hw/data/test_new/"

with open(json_file, 'r') as fp:
    data = json.load(fp)

with open(csv_filename, mode='w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    
    for entry in data:
        image_name = entry[2].replace(root, "")
        predictions = entry[1]
        row = [image_name] + predictions
        csvwriter.writerow(row)

print(csv_filename)