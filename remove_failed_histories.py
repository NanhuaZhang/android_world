import os
import shutil

# 根目录
root_dir = "task_histories"

# task_result
task_result = {
    "2821": 0,
    "1711": 0,
    "3228": 2,
    "1768": 0,
    "1934": 0,
    "1599": 0,
    "3127": 2,
    "3777": 0,
    "1486": 0,
    "2670": 0,
    "1431": 0,
    "2043": 0,
    "951": 0,
    "278": 0,
    "2515": 0,
    "1134": 0,
    "1309": 0
}

folders_to_delete = [key for key, value in task_result.items() if value == 2]

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
