import os
import json

root_dir = "task_histories"
trajectory_output_file = "trajectory_mapping.json"
result = {}

# 遍历每个子文件夹
for folder_name in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            json_path = os.path.join(folder_path, file_name)
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    trajectory_type = data.get("trajectory_type", "Unknown")
                    result[folder_name] = trajectory_type
            except Exception as e:
                print(f"Error reading {json_path}: {e}")
            break  # 每个文件夹只取第一个 json 文件

# 保存为 JSON 文件
with open(trajectory_output_file, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=4)

print(f"✅ 结果已保存为 {trajectory_output_file}")
