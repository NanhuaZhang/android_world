# 假设你的 JSON 文件名为 data.json
import json

from minimal_tasks_runner import parse_recipes, extract_from_template, extract_expense_add_multiple


def getJson():
    with open('trajectory_mapping.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def getKeys():
    data = getJson()

    # 获取所有值为 2 的 key
    keys_with_value_2 = [key for key, value in data.items() if value == 2]

    keys_with_value_2 = list(map(int, keys_with_value_2))

    keys_with_value_2.sort()
    print(keys_with_value_2)
    return keys_with_value_2


if __name__ == '__main__':
#     print(extract_from_template(
#     'Create a playlist titled "{playlist_name1}" with the following files in VLC (located in Internal Memory/VLCVideos), in order: {files1}. And then, create a playlist titled "{playlist_name2}" with the following files in VLC, in order: {files2}.',
# 'Create a playlist titled ""Travel Guide Essentials"" with the following files in VLC (located in Internal Memory/VLCVideos), in order: recording_37_HD_VxEM.mp4, clip_92__2023_04_23.mp4, scene_87_4K_wlpU.mp4, episode_97_export_2023_01_27.mp4. And then, create a playlist titled ""Gaming Sessions Essentials"" with the following files in VLC, in order: moment_94_export_copy.mp4, scene_22_HD_final.mp4, moment_21__copy.mp4.'
#             ))
    getKeys()