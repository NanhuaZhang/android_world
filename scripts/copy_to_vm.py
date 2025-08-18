# pyinstaller --onefile --add-data "source:source" copy_to_vm.py

import tkinter as tk
from tkinter import messagebox, filedialog
import subprocess
import csv
import os
from tkinter import ttk
from copy_to_vm_utils import extract_file_delete, extract_file_move, get_id_key, check_device_connection, resource_path, extract_retro_playlist_create, extract_retro_play, scan_music_directory, create_mp3_file, COMMON_GIVEN_NAMES, clear_internal_storage, delete_files
import random

def select_csv_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        csv_path_var.set(file_path)
        csv_path_label.config(text=f"已选择: {file_path}")

def process_task(task, instruction, result_text):
    if task == 'FilesDeleteFile':
        filenames, folder_name = extract_file_delete(instruction)
        process_files(filenames, folder_name, task, result_text)
    elif task == 'FilesMoveFile':
        filenames, folder_name, _ = extract_file_move(instruction)
        process_files(filenames, folder_name, task, result_text)
    elif task == 'RetroCreatePlaylist':
        _, files = extract_retro_playlist_create(instruction)
        files_with_mp3 = [file + ".mp3" for file in files]
        process_files(files_with_mp3, 'Music', task, result_text)
        scan_music_directory()
    elif task == 'RetroPlayingQueue':
        files = extract_retro_play(instruction)
        files_with_mp3 = [file + ".mp3" for file in files]
        process_files(files_with_mp3, 'Music', task, result_text)
        scan_music_directory()
    else:
        result_text.insert(tk.END, f"暂不支持当前类型 {task}")
        return
    
def process_files(filenames, folder_name, task, result_text):
    if not filenames or not folder_name:
        result_text.insert(tk.END, f"无法解析指令")
        return

    # 确保 filenames 是一个列表
    if isinstance(filenames, str):
        filenames = [filenames]

    dest_paths = []
    success_paths = []
    failed_paths = []
    clear_internal_storage()

    print(f"即将处理文件: {filenames}，目标文件夹: {folder_name}")
    for filename in filenames:
        # Determine source file
        file_extension = os.path.splitext(filename)[1].lower()
        if file_extension in ['.mp3']:
            source_file = create_mp3_file(filename, artist=random.choice(COMMON_GIVEN_NAMES), duration_milliseconds=random.randint(3 * 60 * 1000, 5 * 60 * 1000))
        elif file_extension in ['.mp4', '.pdf', '.txt', '.jpg', '.png', '.wav']:
            source_file = resource_path(f'source/sample{file_extension}')
        else:
            source_file = resource_path('source/sample.txt')

        # Prepare destination path
        dest_path = f'/sdcard/{folder_name}/{filename}'
        dest_paths.append(dest_path)

        # Execute adb command
        command = f'adb push "{source_file}" "{dest_path}"'
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        if result.returncode == 0:
            success_paths.append(dest_path)
        else:
            failed_paths.append((dest_path, result.stderr))
        
        delete_files([filename])

    # 输出结果
    if success_paths:
        result_text.insert(tk.END, f"以下文件已成功创建：\n" + "\n".join(success_paths) + "\n")

    if failed_paths:
        result_text.insert(tk.END, f"以下文件操作失败：\n")
        for path, error in failed_paths:
            result_text.insert(tk.END, f"{path}: {error}\n")

    if not success_paths and not failed_paths:
        result_text.insert(tk.END, f"没有文件被处理")

def copy_file():
    result_text.config(state=tk.NORMAL)
    result_text.delete(1.0, tk.END)

    id_value = id_entry.get()
    
     # 检查设备连接
    if not check_device_connection():
        result_text.insert(tk.END, "错误：没有检测到已连接的设备。请确保设备已正确连接。")
        result_text.config(state=tk.DISABLED)
        return
    
    # Read CSV file
    csv_path = csv_path_var.get()

    if not csv_path:
        result_text.insert(tk.END, "错误：请先选择CSV文件。")
        result_text.config(state=tk.DISABLED)
        return
    
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        task_data = None
        for row in reader:
            id_key = get_id_key(row)
            if row[id_key] == id_value:
                task_data = row
                break
    
    if not task_data:
        messagebox.showerror("错误", f"未找到ID为 {id_value} 的任务")
        return

    task = task_data['task']
    instruction = task_data['instruction']

    process_task(task, instruction, result_text)

# Create main window
root = tk.Tk()
root.title("文件复制到虚拟机")
root.geometry("500x300")  # 设置窗口大小

# 创建主框架并添加内边距
main_frame = ttk.Frame(root, padding="20 20 20 20")
main_frame.pack(fill=tk.BOTH, expand=True)

csv_path_var = tk.StringVar()

# CSV file selection
tk.Label(main_frame, text="选择CSV文件:").grid(row=0, column=0, sticky="w", pady=10)
select_csv_button = tk.Button(main_frame, text="浏览", command=select_csv_file)
select_csv_button.grid(row=0, column=1, sticky="w", pady=10)
csv_path_label = tk.Label(main_frame, text="未选择文件", wraplength=400)
csv_path_label.grid(row=1, column=0, columnspan=2, sticky="w", pady=5)

# ID input
tk.Label(main_frame, text="输入ID:").grid(row=2, column=0, sticky="w", pady=10)
id_entry = tk.Entry(main_frame)
id_entry.grid(row=2, column=1, sticky="w", pady=10)
id_entry.insert(0, "1")  # Default value

# Copy button
copy_button = tk.Button(main_frame, text="复制文件到虚拟机", command=copy_file)
copy_button.grid(row=3, column=0, columnspan=2, pady=20)

# 结果文本区域
result_text = tk.Text(main_frame, wrap=tk.WORD, height=3)
result_text.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(10, 0))
result_text.config(state=tk.DISABLED)

# 配置列的权重，使其能够扩展
main_frame.columnconfigure(1, weight=1)

root.mainloop()
