import csv
import os
import json
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_csv(file_path):
    data = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data[row['id']] = row['instruction']
        logging.info(f"成功读取CSV文件: {file_path}")
    except Exception as e:
        logging.error(f"读取CSV文件时出错: {e}")
    return data

def update_json_files(csv_data, task_folder):
    success_count = 0
    for root, dirs, files in os.walk(task_folder):
        for dir_name in dirs:
            json_file_path = os.path.join(root, dir_name, 'task.json')
            if os.path.exists(json_file_path):
                try:
                    with open(json_file_path, 'r', encoding='utf-8') as jsonfile:
                        data = json.load(jsonfile)
                    
                    if dir_name in csv_data:
                        data['instruction'] = csv_data[dir_name]
                        
                        with open(json_file_path, 'w', encoding='utf-8') as jsonfile:
                            json.dump(data, jsonfile, ensure_ascii=False, indent=2)
                        
                        logging.info(f"成功更新文件: {json_file_path}")
                        success_count += 1
                    else:
                        logging.warning(f"CSV中未找到ID {dir_name} 的对应指令")
                except Exception as e:
                    logging.error(f"处理文件 {json_file_path} 时出错: {e}")
    
    return success_count

def main():
    csv_file_path = './output.csv'
    task_folder = './task_histories'

    csv_data = read_csv(csv_file_path)
    success_count = update_json_files(csv_data, task_folder)

    logging.info(f"处理完成。成功修改了 {success_count} 个文件。")

if __name__ == "__main__":
    main()
