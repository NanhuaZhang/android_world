# 假设你的 JSON 文件名为 data.json
import json
import os
import re


def getKeys():
    with open('trajectory_mapping.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 获取所有值为 2 的 key
    keys_with_value_2 = [key for key, value in data.items() if value == 2]

    print(keys_with_value_2)
    return list(map(int, keys_with_value_2))


if __name__ == '__main__':
    print(re.search(r"\.md$", '2023_02_09_busy_pig.txt'))