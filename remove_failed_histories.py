import os
import shutil

# 根目录
root_dir = "task_histories_receipt_delete/task_histories"

# 要删除的文件夹名称列表
folders_to_delete = [
    "683",
    "1239",
    "1472",
    "1974",
    "2239",
    "2342",
    "2395",
    "2808",
    "2908",
    "2959",
    "3061",
    "3062",
    "3316",
    "3411",
    "3614",
    "3663",
    "3764",
    "4123",
    "4220",
    "4271",
    "4322",
    "4472",
    "4522",
    "4671",
    "4768",
    "4915",
    "4963",
    "5060",
    "5254",
    "5306"
]

for folder_name in folders_to_delete:
    folder_path = os.path.join(root_dir, folder_name)
    if os.path.isdir(folder_path):
        try:
            shutil.rmtree(folder_path)
            print(f"已删除：{folder_path}")
        except Exception as e:
            print(f"删除失败：{folder_path}，错误：{e}")
    else:
        print(f"未找到文件夹：{folder_path}")
