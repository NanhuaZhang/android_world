import re
import subprocess
import os
import sys
import pydub

def get_id_key(row):
    return '\ufeffid' if '\ufeffid' in row else 'id'

def check_device_connection():
    command = 'adb devices'
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    devices = result.stdout.strip().split('\n')[1:]
    return len(devices) > 0 and not all(device.endswith('offline') for device in devices)


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def extract_file_delete(instruction):
    pattern = r"Delete the file ([\w.-]+?\.[a-zA-Z0-9]+).*?located in the ([\w\s]+?) folder"
    match = re.search(pattern, instruction)

    if match:
        filename = match.group(1)
        folder_name = match.group(2)
        return filename, folder_name
    else:
        return None, None
    
def extract_file_move(instruction):
    # print(f"extract_file_move: ", instruction)
    pattern = r"Move the file ([\w.-]+?\.[\w]+) from ([\w\s]+?) .*? to the ([\w\s]+?) within"
    match = re.search(pattern, instruction)

    if match:
        file_name = match.group(1)
        source_folder = match.group(2)
        destination_folder = match.group(3)
        return file_name, source_folder, destination_folder
    else:
        return None, None, None
    
def extract_retro_playlist_create(instruction):
    # 提取播放列表名称
    playlist_name_match = re.search(r'Create a playlist in Retro Music titled "([^"]+)"', instruction)
    playlist_name = playlist_name_match.group(1) if playlist_name_match else None

    # 提取文件列表
    files_match = re.search(r'in order: (.+)', instruction)
    files = files_match.group(1).split(', ') if files_match else []

    return playlist_name, files

def extract_retro_play(instruction):
    # 提取文件列表
    instruction = instruction.replace(' to my playing queue in Retro music.', '')
    files_match = re.search(r'Add the following songs, in order, (.+)', instruction)
    files = files_match.group(1).split(', ') if files_match else []

    return files

def scan_music_directory():
    """扫描音乐目录并更新媒体存储"""
    
    # 发送广播通知扫描音乐文件
    broadcast_command = [
        'adb', 'shell', 'am', 'broadcast',
        '-a', 'android.intent.action.MEDIA_SCANNER_SCAN_FILE',
        '-d', 'file:///storage/emulated/0/Music'
    ]
    subprocess.run(broadcast_command, check=True)
    
    # 关闭Retro Music应用
    close_command = [
        'adb', 'shell', 'am', 'force-stop', 'code.name.monkey.retromusic'
    ]
    subprocess.run(close_command, check=True)

    print("音乐目录已扫描,Retro Music已关闭")

COMMON_GIVEN_NAMES = [
    # keep-sorted start
    "Abdullah",
    "Adam",
    "Ahmed",
    "Alejandro",
    "Ali",
    "Alice",
    "Amelia",
    "Amina",
    "Amir",
    "Ana",
    "Anna",
    "Aria",
    "Arthur",
    "Ava",
    "Camila",
    "Carlos",
    "Charlie",
    "Charlotte",
    "Daniel",
    "David",
    "Elias",
    "Ella",
    "Ema",
    "Emil",
    "Emilia",
    "Emily",
    "Emma",
    "Eva",
    "Fatima",
    "Freya",
    "Gabriel",
    "George",
    "Grace",
    "Hana",
    "Hannah",
    "Henry",
    "Hugo",
    "Ian",
    "Ibrahim",
    "Isabella",
    "Isla",
    "Ivan",
    "Jack",
    "James",
    "Jose",
    "Juan",
    "Laura",
    "Leo",
    "Leon",
    "Leonardo",
    "Liam",
    "Lily",
    "Lina",
    "Louis",
    "Luca",
    "Lucas",
    "Luis",
    "Luka",
    "Maria",
    "Mariam",
    "Mark",
    "Martin",
    "Martina",
    "Maryam",
    "Mateo",
    "Matteo",
    "Maya",
    "Mia",
    "Miguel",
    "Mila",
    "Mohammad",
    "Muhammad",
    "Nikola",
    "Noa",
    "Noah",
    "Nora",
    "Oliver",
    "Olivia",
    "Omar",
    "Oscar",
    "Petar",
    "Samuel",
    "Santiago",
    "Sara",
    "Sarah",
    "Sofia",
    "Sofija",
    "Sophia",
    "Sophie",
    "Theo",
    "Theodore",
    "Thiago",
    "Thomas",
    "Valentina",
    "Victoria",
    "William",
    "Willow",
    # keep-sorted end
]

def create_mp3_file(filename, artist, duration_milliseconds):
    """
    创建一个指定文件名、艺术家、标题和时长的 MP3 文件。

    :param filename: 要创建的 MP3 文件名
    :param artist: 艺术家名称
    :param title: 歌曲标题
    :param duration_milliseconds: 音频时长（毫秒）
    :return: 创建的文件的完整路径
    """
    # 确保文件名以 .mp3 结尾
    if not filename.lower().endswith('.mp3'):
        filename += '.mp3'

    title = os.path.splitext(filename)[0]

    # 获取当前运行程序的目录
    current_dir = os.getcwd()

    # 构建完整的文件路径
    file_path = os.path.join(current_dir, filename)

    # 创建指定时长的静音音频段
    tone = pydub.AudioSegment.silent(duration=duration_milliseconds)

    # 导出 MP3 文件，并设置元数据
    tone.export(
        file_path,
        format="mp3",
        tags={"artist": artist, "title": title}
    )

    return file_path


def clear_internal_storage():
    """
    Deletes all files from internal storage (sdcard), leaving directory structure intact.
    Ignores permission errors and continues with deletion.
    """
    try:
        adb_command = [
            "adb",
            "shell",
            "find",
            "/storage/emulated/0/",
            "-mindepth",
            "1",
            "-type",
            "f",
            "-delete",
            "2>/dev/null"  # Redirect stderr to /dev/null
        ]
        
        result = subprocess.run(adb_command, check=False, capture_output=True, text=True)
        
        if "Permission denied" in result.stderr:
            print("Some files could not be deleted due to permission issues, but the operation continued.")
        elif result.returncode != 0:
            print(f"Warning: Command completed with non-zero exit status. Some files may not have been deleted.")
        
        print("Operation completed. Most files should have been deleted from internal storage.")
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def delete_files(files):
    """
    Delete files from the current directory based on the provided list of filenames.

    :param files: A list of filenames to be deleted
    :return: A dictionary with the status of each file deletion
    """
    current_dir = os.getcwd()
    results = {}

    for file in files:
        file_path = os.path.join(current_dir, file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                results[file] = "Successfully deleted"
            else:
                results[file] = "File not found"
        except Exception as e:
            results[file] = f"Error deleting file: {str(e)}"

    return results