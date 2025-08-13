# 假设你的 JSON 文件名为 data.json
import json

from minimal_tasks_runner import parse_recipes


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
#     print(parse_recipes("""Add the following recipes into the Broccoli app:
# title|description|servings|preparationTime|ingredients|directions
# Chickpea Vegetable Soup|An ideal recipe for experimenting with different flavors and ingredients.|3-4 servings|4 hrs|flexible ingredients|Saut?© onions, carrots, and celery, add broth, canned tomatoes, and chickpeas. Simmer with spinach and seasonings. Feel free to substitute with ingredients you have on hand."
# """))
    getKeys()