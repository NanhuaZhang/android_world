# Copyright 2025 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runs a single task.

The minimal_run.py module is used to run a single task, it is a minimal version
of the run.py module. A task can be specified, otherwise a random task is
selected.
"""
import base64
import dataclasses
import datetime
import io
import json
from collections.abc import Sequence
import os
import random
import re
from typing import Type

from android_world.agents.humans.draw_pro_agent import DrawPro
from android_world.agents.humans.change_markor import ChangeMarkorContent, MarkorAddNoteHeader, MarkorMergeNotes
from android_world.task_evals.single import vlc
from android_world.utils import datetime_utils
import pandas as pd
from PIL import Image
from absl import app
from absl import flags
from absl import logging
from android_world import registry
from android_world.agents import infer, m3a_utils
from android_world.agents import t3a
from android_world.agents import doubao_agent
from android_world.agents.doubao_agent import Doubao
from android_world.env import env_launcher, device_constants
from android_world.task_evals import task_eval
import xml.etree.ElementTree as ET
import subprocess

from android_world.task_evals.single.calendar import calendar_utils
from android_world.task_evals.single.calendar.calendar import generate_noise_events, _REPEAT_INTERVALS
from android_world.task_evals.single.retro_music import _SONGS, _generate_playlist_name
from android_world.task_evals.utils import receipt_generator, sqlite_schema_utils
from android_world.utils.datetime_utils import create_random_october_2023_unix_ts, _create_unix_ts
from android_world.task_evals.single.vlc import generate_file_name
from android_world.task_evals.utils import sqlite_schema_utils, user_data_generation

from android_world.task_evals.common_validators import sqlite_validators
from android_world.task_evals.single.expense import _get_random_timestamp, _generate_expense
from android_world.task_evals.single.markor import _NOTE_TITLES, generate_random_sentence
from android_world.task_evals.single.recipe import _generate_random_recipe
from android_world.task_evals.single.calendar import events_generator
from parse import parse

logging.set_verbosity(logging.WARNING)

os.environ['GRPC_VERBOSITY'] = 'ERROR'  # Only show errors
os.environ['GRPC_TRACE'] = 'none'  # Disable tracing

RETRY_KEYWORDS = [
    'not visible',
    'i will try to',
    'from a different angle',
    'attempt to',
    'further actions may be needed',
    'but they remain inaccessible',
    'failed',
    'wrong',
    'incorrect',
]

_NOTES = [
    'Paid by card',
    'Urgent',
    'Monthly recurring',
    'Want to have',
    'A need',
    'Remember to transfer funds',
    'I may repeat this',
]


def _find_adb_directory() -> str:
    """Returns the directory where adb is located."""
    potential_paths = [
        os.path.expanduser('~/Library/Android/sdk/platform-tools/adb'),
        os.path.expanduser('~/Android/Sdk/platform-tools/adb'),
        os.path.expanduser('~/Android/platform-tools/adb'),
        os.path.expanduser('C:\\Users\\Appen\\AppData\\Local\\Android\\Sdk\\platform-tools\\adb.exe'),
    ]
    for path in potential_paths:
        if os.path.isfile(path):
            return path
    raise EnvironmentError(
        'adb not found in the common Android SDK paths. Please install Android'
        " SDK and ensure adb is in one of the expected directories. If it's"
        ' already installed, point to the installed location.'
    )


_ADB_PATH = flags.DEFINE_string(
    'adb_path',
    _find_adb_directory(),
    'Path to adb. Set if not installed through SDK.',
)
_EMULATOR_SETUP = flags.DEFINE_boolean(
    'perform_emulator_setup',
    True,
    'Whether to perform emulator setup. This must be done once and only once'
    ' before running Android World. After an emulator is setup, this flag'
    ' should always be False.',
)
_DEVICE_CONSOLE_PORT = flags.DEFINE_integer(
    'console_port',
    5554,
    'The console port of the running Android device. This can usually be'
    ' retrieved by looking at the output of `adb devices`. In general, the'
    ' first connected device is port 5554, the second is 5556, and'
    ' so on.',
)

_AGENT_TYPE = flags.DEFINE_string(
    'agent_type',
    'doubao',
    'Agent to use. Options: openai, doubao',
)

_MAX_STEP_COUNT = flags.DEFINE_integer(
    'max_step_count',
    180,
    'The max step count of the agent.',
)


# 定义正则表达式提取函数
def extract_name_and_number(instruction):
    # 提取电话号码（+ 和数字）
    number_match = re.search(r'\+?\d{10,}', instruction)
    number = number_match.group() if number_match else None

    # 提取姓名（寻找 “contact for xxx.” 的格式）
    name_match = re.search(r'contact for ((?:[A-Z][a-z]+(?:\s[A-Z][a-z]+)*)|(?:[A-Z][a-z]+))', instruction)
    name = name_match.group(1) if name_match else None

    return name, number


def extract_sms_reply(instruction):
    number_match = re.search(r'\+?1?\d{10}', instruction)
    message_match = re.search(r'message:\s*(.*)\s+in', instruction, re.IGNORECASE)

    number = number_match.group() if number_match else None
    message = message_match.group(1).strip() if message_match else None

    return number, message


def extract_sms_info(instruction):
    # 提取电话号码（支持+或无前缀）
    number_match = re.search(r'\+?(\d{10,})', instruction)
    number = number_match.group(1) if number_match else None

    # 提取短信内容（在 message: 后）
    message_match = re.search(r'message:\s*(.+)', instruction)
    message = message_match.group(1).strip() if message_match else None

    return number, message


def extract_sms_number(instruction):
    # 提取电话号码（+ 和数字）
    number_match = re.search(r'\+?\d{10,}', instruction)
    number = number_match.group() if number_match else None

    return number


def extract_sms_message(instruction):
    # 提取短信内容（在 message: 后）
    message_match = re.search(r'message:\s*(.+)', instruction)
    message = message_match.group(1).strip() if message_match else None

    return message


def extract_contact_details(instruction):
    first = re.search(r'First Name:\s*([A-Za-z]+)', instruction)
    last = re.search(r'Last Name:\s*([A-Za-z]+)', instruction)
    phone = re.search(r'Phone:\s*([\d\-]+)', instruction)
    label = re.search(r'Phone Label:\s*([A-Za-z]+)', instruction)

    return {
        'first_name': first.group(1) if first else None,
        'last_name': last.group(1) if last else None,
        'phone': phone.group(1) if phone else None,
        'phone_label': label.group(1) if label else None,
    }


def extract_calendar_event_info(instruction):
    # 日期和时间
    date_match = re.search(r'on (\d{4})-(\d{2})-(\d{2}) at (\d+)h', instruction)
    year, month, day, hour = date_match.groups() if date_match else (None, None, None, None)

    # 标题
    title_match = re.search(r"title ['\"](.+?)['\"]", instruction)
    title = title_match.group(1) if title_match else None

    # 描述
    desc_match = re.search(r"description ['\"](.+?)['\"]", instruction)
    description = desc_match.group(1) if desc_match else None

    # 持续时间（分钟）
    duration_match = re.search(r'last for (\d+) mins?', instruction)
    duration = int(duration_match.group(1)) if duration_match else None

    return int(year), int(month), int(day), int(hour), title, description, duration


def extract_repeat_event_info(text: str):
    # 提取标题
    title_match = re.search(r"titled\s+'([^']+)'", text)
    title = title_match.group(1) if title_match else ""

    # 提取时间
    date_time_match = re.search(r"starting on (\d{4})-(\d{1,2})-(\d{1,2}) at (\d{1,2})h", text)
    year, month, day, hour = map(int, date_time_match.groups()) if date_time_match else (0, 0, 0, 0)

    # 提取重复频率
    recurrence_match = re.search(r"recurs\s+(daily|weekly)", text, re.IGNORECASE)
    recurrence = recurrence_match.group(1).lower() if recurrence_match else ""

    # 提取时长
    duration_match = re.search(r"lasts for (\d+)\s*minutes?", text)
    duration = int(duration_match.group(1)) if duration_match else 0

    # 提取描述
    desc_match = re.search(r"description should be\s+'([^']+)'", text)
    description = desc_match.group(1) if desc_match else ""

    return title, year, month, day, hour, recurrence, duration, description


def extract_event_info(text):
    # 提取时间（小时）
    time_match = re.search(r'(\d{1,2})\s*(?:h|:00)', text)
    hour = int(time_match.group(1)) if time_match else None

    # 提取标题（单引号或双引号包裹）
    title_match = re.search(r"(?:title\s+['\"])(.*?)(?:['\"])", text, re.IGNORECASE)
    title = title_match.group(1) if title_match else None

    # 提取描述（description后跟引号包裹的内容）
    description_match = re.search(r"(?:description\s+['\"])(.*?)(?:['\"])", text, re.IGNORECASE)
    description = description_match.group(1) if description_match else None

    # 提取持续时间（分钟）
    duration_match = re.search(r'(\d+)\s*min', text, re.IGNORECASE)
    duration = int(duration_match.group(1)) if duration_match else None

    return hour, title, description, duration


def extract_event_details(text):
    weekday_match = re.search(r'\b(on|for)?\s*(this\s+)?(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b',
                              text, re.IGNORECASE)
    hour_match = re.search(r'\b(?:at\s+)?(\d{1,2})h\b', text)
    title_match = re.search(r"title\s+'([^']+)'", text)
    description_match = re.search(r"description\s+'([^']+)'", text)
    duration_match = re.search(r'last(?:s)? for (\d+) mins', text)

    weekday = weekday_match.group(3).capitalize() if weekday_match else None
    hour = int(hour_match.group(1)) if hour_match else None
    title = title_match.group(1) if title_match else None
    description = description_match.group(1) if description_match else None
    duration = int(duration_match.group(1)) if duration_match else None

    return weekday, hour, title, description, duration


def extract_event_tuple(text: str):
    try:
        hour = int(re.search(r'at (\d{1,2})h', text).group(1))
        title = re.search(r"title\s+'([^']+)'", text).group(1)
        description = re.search(r"description\s+'([^']+)'", text).group(1)
        duration = int(re.search(r'last for (\d+)\s*min', text).group(1))

        return hour, title, description, duration
    except AttributeError:
        return None


def weekday_to_number(weekday_str):
    weekday_map = {
        'Monday': 1,
        'Tuesday': 2,
        'Wednesday': 3,
        'Thursday': 4,
        'Friday': 5,
        'Saturday': 6,
        'Sunday': 7,
    }
    return weekday_map.get(weekday_str.capitalize(), None)


def extract_file_delete(instruction):
    pattern = r"Delete the file ([\w.-]+?\.[a-zA-Z0-9]+).*?located in the ([\w\s]+?) folder"
    match = re.search(pattern, instruction)

    if match:
        filename = match.group(1)  # 输出: final_smart_lion.mp3
        folder_name = match.group(2)  # 输出: Notifications
        return filename, folder_name
    else:
        return None, None


def extract_file_move(instruction):
    pattern = r"Move the file ([\w.-]+?\.[\w]+) from ([\w\s]+?) .*? to the ([\w\s]+?) within"
    match = re.search(pattern, instruction)

    if match:
        file_name = match.group(1)
        source_folder = match.group(2)
        destination_folder = match.group(3)
        return file_name, source_folder, destination_folder
    else:
        return None, None, None


def extract_vlc_playlist_create(instruction):
    # 提取播放列表名称
    playlist_name_match = re.search(r'Create a playlist titled "([^"]+)"', instruction)
    playlist_name = playlist_name_match.group(1) if playlist_name_match else None

    # 提取文件列表
    files_match = re.search(r'in order: (.+)', instruction)
    files = files_match.group(1).split(', ') if files_match else []

    return playlist_name, files


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
    files_match = re.search(r'Add the following songs, in order, (.+)', instruction)
    files = files_match.group(1).split(', ') if files_match else []

    return files


def extract_expense_add_multiple(instruction):
    # 定义正则表达式
    pattern_a = r'Expense:\s*(.+?)\s*amount_dollars:\s*\$?(\d+(?:\.\d+)?)\s*category_name:\s*(.+?)\s*note:\s*(.+?)(?=\r\n|$)'
    pattern_b = r'(.+?)\|(\$?\d+(?:\.\d+)?)\|(.+?)\|(.+)'

    results = []
    type = None

    # 尝试匹配格式 a
    expenses_a = re.findall(pattern_a, instruction, re.DOTALL)
    if expenses_a:
        for name, amount, category, note in expenses_a:
            results.append({
                'name': name.strip(),
                'amount': amount.strip('$'),
                'category': category.strip(),
                'note': note.strip()
            })
        type = "text_block"
    else:
        # 尝试匹配格式 b
        expenses_b = re.findall(pattern_b, instruction)
        if expenses_b:
            for name, amount, category, note in expenses_b:
                results.append({
                    'name': name.strip(),
                    'amount': amount.strip('$'),  # 去掉美元符号
                    'category': category.strip(),
                    'note': note.strip()
                })
            type = "csv"
    return type, results


def detect_recipe_type(text: str) -> str | None:
    stripped = text.strip()

    # 如果第一行是表头，且包含竖线分隔
    first_line = stripped.splitlines()[0]
    if "|" in first_line and first_line.lower().startswith("title|description"):
        return "csv"

    # 如果文本中有明显的 Recipe: 开头
    if "Recipe:" in stripped.splitlines()[0] or any(line.startswith("Recipe:") for line in stripped.splitlines()):
        return "text_block"

    return None


def parse_recipes(text: str):
    recipes = []
    recipe_type = None

    # 先尝试查找格式2的 Recipe: 开头的多条记录
    pattern2 = re.compile(
        r"Recipe:\s*(.*?)\n\s*description:\s*(.*?)\n\s*servings:\s*(.*?)\n\s*preparationTime:\s*(.*?)\n\s*ingredients:\s*(.*?)\n\s*directions:\s*(.*?)(?=\nRecipe:|\Z)",
        re.DOTALL
    )
    for match in pattern2.finditer(text):
        recipe = {
            "title": match.group(1).strip(),
            "description": match.group(2).strip(),
            "servings": match.group(3).strip(),
            "preparationTime": match.group(4).strip(),
            "ingredients": match.group(5).strip(),
            "directions": match.group(6).strip(),
        }
        recipes.append(recipe)
        recipe_type = 'text_block'

    # 再尝试查找格式1的表格形式
    if "title|description|servings|preparationTime|ingredients|directions" in text:
        lines = text.strip().splitlines()
        try:
            header_index = lines.index("title|description|servings|preparationTime|ingredients|directions")
            for line in lines[header_index + 1:]:
                if not line.strip():
                    continue
                parts = line.split("|")
                if len(parts) == 6:
                    recipe = {
                        "title": parts[0].strip(),
                        "description": parts[1].strip(),
                        "servings": parts[2].strip(),
                        "preparationTime": parts[3].strip(),
                        "ingredients": parts[4].strip(),
                        "directions": parts[5].strip(),
                    }
                    recipes.append(recipe)
                    recipe_type = 'csv'
        except ValueError:
            pass  # 找不到表头，忽略

    return recipe_type, recipes


def extract_expense_delete_details(instruction):
    # 找到冒号后的部分
    if ':' in instruction:
        after_colon = instruction.split(':', 1)[1]
        # 去除句号并按逗号分隔
        items = [item.strip().rstrip('.') for item in after_colon.split(',')]
        # 去除空项并返回
        return [item for item in items if item]
    return []


def extract_from_template(template: str, text: str) -> tuple:
    result = parse(template, text)
    return result.named.values() if result else None


def get_day_of_week(day_of_week):
    # 创建日期对象，假设 a 是当前日期的日
    current_date = device_constants.DT

    # 将星期几转换为数字（0=Monday, 6=Sunday）
    days_of_week = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6
    }

    # 获取目标星期几的数字表示
    target_weekday = days_of_week.get(day_of_week)

    if target_weekday is None:
        raise ValueError("Invalid day of the week")

    # 计算目标日期
    current_weekday = current_date.weekday()
    days_difference = (target_weekday - current_weekday) % 7
    target_date = current_date + datetime.timedelta(days=days_difference)

    return {
        "year": target_date.year,
        "month": target_date.month,
        "day": target_date.day,
    }


from typing import List, Dict, Any, Optional, Union


def count_key_values(data: List[Dict[str, Any]], key: str, value: Optional[Any] = None) -> Union[int, dict]:
    """
    统计数组对象中某个 key 的值出现次数（不用 Counter）。

    :param data: 列表，每个元素是字典
    :param key: 要统计的 key
    :param value: 可选，只统计某个具体值
    :return: 如果 value 为空，返回 dict；否则返回指定值的次数
    """
    counts = {}
    for item in data:
        val = item.get(key)
        counts[val] = counts.get(val, 0) + 1

    if value is not None:
        return counts.get(value, 0)
    return counts


def read_csv():
    # 读取 CSV 文件
    df = pd.read_csv('output.csv')  # 替换为你的文件
    return df

retry_task = [
576,
708,
1262,
1551,
2157,
2209,
2469,
2880,
3433,
3533,
3634,
3735,
3939,
4093,
4395,
4741,
5032,
5129]

complete_task = []
def _main() -> None:
    """Runs a single task."""
    env = env_launcher.load_and_setup_env(
        console_port=_DEVICE_CONSOLE_PORT.value,
        emulator_setup=_EMULATOR_SETUP.value,
        adb_path=_ADB_PATH.value,
    )
    env.reset(go_home=True)
    task_registry = registry.TaskRegistry()
    aw_registry = task_registry.get_registry(task_registry.ANDROID_WORLD_FAMILY)

    file_result = read_csv()
    for idx, row in file_result.iterrows():
        task_value = row['task']
        if task_value not in aw_registry:
            raise ValueError('Task {} not found in registry.'.format(task_value))
        task_type: Type[task_eval.TaskEval] = aw_registry[task_value]
        params = None

        # if os.path.exists(os.path.join("./task_histories", str(row['id']))):
        #     print(f"文件已存在 {row['id']}")
        #     continue

        if len(retry_task) > 0 and row['id'] not in retry_task:
            continue

        # if task_value == 'MarkorTranscribeVideo':
        #     video_name, file_name= extract_from_template(
        #         "Transcribe the contents of video {video_name} by watching it in VLC player (located in Download) and writing the sequence of strings shown on each frame to the text file {file_name} in Markor as a comma separated list. For example, if the first frame shows the text \"edna\" and the second frame shows the text \"pineapple\", then the text file should contain only the following text: \"edna, pineapple\".",
        #         row['instruction'])
        #     if video_name is not None and file_name is not None:
        #         print(f"currentTask metadata: video_name: {video_name}, file_name: {file_name} ")
        #         messages = list(
        #             random.sample(
        #                 user_data_generation.COMMON_GIVEN_NAMES, random.randint(2, 4)
        #             )
        #         )
        #         params = {
        #             "file_name": file_name,
        #             "text": ",".join(messages),
        #             # Video specific.
        #             "messages": messages,
        #             "video_name": video_name,
        #             "noise_files": [
        #                 vlc.generate_file_name() for _ in range(random.randint(3, 10))
        #             ],
        #         }

        # if task_value == 'TasksDueNextWeek':
        #     extract_from_template(
        #         "How many tasks do I have due next week in Tasks app? Assume the week starts from Monday. Express your answer as a single integer.",
        #         row['instruction']
        #     )
        #     params = {}

        # if task_value == 'TasksHighPriorityTasks':
        #     extract_from_template(
        #         "What are my high priority tasks in Tasks app? Answer with the titles only. If there are multiples titles, format your answer in a comma separated list.",
        #         row['instruction']
        #     )
        #     params = {
        #       "task_title": random.choice([
        #         "Complete project proposal",
        #         "Review code changes",
        #         "Schedule team meeting",
        #         "Submit expense report",
        #         "Update website content",
        #         "Review quarterly goals",
        #         "Organize files and folders",
        #         "Draft marketing email",
        #         "Attend networking event",
        #         "Prepare presentation for meeting",
        #         "Call client for follow-up",
        #         "Research market trends"
        #       ]),
        #       "notes": random.choice([
        #         "Remember to complete this task.",
        #         "This task is important.",
        #         "Don't forget to follow up on this task.",
        #         "Double-check calculations.",
        #         "Confirm meeting location.",
        #         "Follow up with team members.",
        #         "Prepare presentation slides.",
        #         "Update project status.",
        #         "Check email for updates.",
        #         "Review feedback from client.",
        #         "Complete paperwork.",
        #         "Schedule follow-up meeting."
        #       ]),
        #       "due_date": random.choice([
        #         "October 16 2023",
        #         "October 17 2023",
        #         "October 18 2023",
        #         "October 19 2023",
        #         "October 20 2023",
        #         "October 21 2023",
        #         "October 22 2023"
        #       ]),
        #       "time": random.choice([
        #         "11:00am",
        #         "1:30pm",
        #         "9:45am",
        #         "21:45",
        #         "10:00am",
        #         "3:15pm",
        #         "12pm",
        #         "6:30am",
        #         "8:00pm",
        #         "12am",
        #         "5:20pm"
        #       ]),
        #       "hide_until_date": random.choice([
        #         "October 10 2023",
        #         "October 11 2023",
        #         "October 12 2023",
        #         "October 13 2023",
        #         "October 14 2023",
        #         "October 15 2023"
        #       ])
        #     }

        # if task_value == 'TasksHighPriorityTasksDueOnDate':
        #     date= extract_from_template(
        #         "Which tasks with high priority are due {date} in the Tasks app? Answer with the title only. If there are multiples titles, format your answer in a comma separated list.",
        #         row['instruction'])
        #     date = list(date)[0]
        #     if date is not None:
        #         print(f"date: {date} ")
        #         params = {
        #             'title': 'Team Sync-Up Meeting',
        #             'date': date,
        #             'time': '5:00pm',
        #         }

        # if task_value == 'TasksDueOnDate':
        #     date= extract_from_template(
        #         "What tasks do I have due {date} in Tasks app? Answer with the titles only. If there are multiples titles, format your answer in a comma separated list.",
        #         row['instruction'])
        #     date = list(date)[0]
        #     if date is not None:
        #         print(f"date: {date} ")
        #         params = {
        #             'title': 'Attend networking event',
        #             'date': date,
        #             'notes': 'Complete paperwork.',
        #             'importance': '0'
        #         }

        # if task_value == 'TasksIncompleteTasksOnDate':
        #     date= extract_from_template(
        #         "What incomplete tasks do I have still have to do by {date} in Tasks app? Answer with the titles only. If there are multiples titles, format your answer in a comma separated list.",
        #         row['instruction'])
        #     date = list(date)[0]
        #     if date is not None:
        #         print(f"date: {date} ")
        #         params = {
        #             'title': 'Schedule team meeting',
        #             'date': date,
        #             'notes': 'Remember to review ahead of time.',
        #             'time': '9:45am',
        #             'importance': '2'
        #         }

        # if task_value == 'TasksCompletedTasksForDate':
        #     date= extract_from_template(
        #         "Which tasks have I completed for {date} in Tasks app? Answer with the titles only. If there are multiples titles, format your answer in a comma separated list.",
        #         row['instruction'])
        #     date = list(date)[0]
        #     if date is not None:
        #         print(f"date: {date} ")
        #         params = {
        #           "title": random.choice([
        #             "Complete project proposal",
        #             "Review code changes",
        #             "Schedule team meeting",
        #             "Submit expense report",
        #             "Update website content",
        #             "Review quarterly goals",
        #             "Organize files and folders",
        #             "Draft marketing email",
        #             "Attend networking event",
        #             "Prepare presentation for meeting",
        #             "Call client for follow-up",
        #             "Research market trends"
        #           ]),
        #           "notes": random.choice([
        #             "Remember to complete this task.",
        #             "This task is important.",
        #             "Don't forget to follow up on this task.",
        #             "Double-check calculations.",
        #             "Confirm meeting location.",
        #             "Follow up with team members.",
        #             "Prepare presentation slides.",
        #             "Update project status.",
        #             "Check email for updates.",
        #             "Review feedback from client.",
        #             "Complete paperwork.",
        #             "Schedule follow-up meeting."
        #           ]),
        #           "date": date,
        #           "time": random.choice([
        #             "11:00am",
        #             "1:30pm",
        #             "9:45am",
        #             "21:45",
        #             "10:00am",
        #             "3:15pm",
        #             "12pm",
        #             "6:30am",
        #             "8:00pm",
        #             "12am",
        #             "5:20pm"
        #           ]),
        #           "completed_date": random.choice([
        #             "October 10 2023",
        #             "October 11 2023",
        #             "October 12 2023",
        #             "October 13 2023",
        #             "October 14 2023",
        #             "October 15 2023"
        #           ])
        #         }


        # if task_value == 'SportsTrackerActivitiesCountForWeek':
        #     category= extract_from_template(
        #         "How many {category} activities did I do this week in the OpenTracks app? Assume the week starts from Monday. Express your answer as a single integer.",
        #         row['instruction'])
        #     category = list(category)[0]
        #     if category is not None:
        #         print(f"category: {category} ")
        #         params = {
        #             'start_date': 'October 10 2023',
        #             'category': category,
        #             'duration': '30',
        #             'distance': '300',
        #             'start_time': '11:00am',
        #             'elevation': '100',
        #             'activity_name': category,
        #             'activity_description': 'Wandered off the beaten path.'
        #         }
        #
        # if task_value == 'SportsTrackerActivitiesOnDate':
        #     date= extract_from_template(
        #         "What activities did I do {date} in the OpenTracks app? Answer with the activity type only. If there are multiple types, format your answer in a comma separated list.",
        #     row['instruction'])
        #     date = list(date)[0]
        #     if date is not None:
        #         print(f"date: {date} ")
        #         params = {
        #             'category': 'cycling',
        #             'date': date,
        #             'duration': '30',
        #             'distance': '300',
        #             'start_time': '11:00am',
        #             'elevation': '100',
        #             'activity_name': 'cycling',
        #             'activity_description': 'Shared laughs and made memories with friends.'
        #         }
        # if task_value == 'SportsTrackerActivityDuration':
        #     category,date= extract_from_template(
        #         "How long was my {category} activity {date} in the OpenTracks app? Express your answer in minutes as a single integer.",
        #     row['instruction'])
        #     if date is not None and category is not None:
        #         print(f"date: {date} ")
        #         params = {
        #             'category': category,
        #             'date': date,
        #             'duration': '30',
        #             'distance': '300',
        #             'start_time': '11:00am',
        #             'elevation': '100',
        #             'activity_name': category,
        #             'activity_description': 'Shared laughs and made memories with friends.'
        #         }
        #
        # if task_value == 'SportsTrackerLongestDistanceActivity':
        #     category= extract_from_template(
        #         "What was the longest distance covered in a {category} activity in the OpenTracks app this week? Assume the week starts from Monday. Express your answer as a single number in meters rounded to the nearest integer.",            row['instruction'])
        #     category = list(category)[0]
        #     if category is not None:
        #         print(f"category: {category} ")
        #         params = {
        #             'category': category,
        #             'start_date': "October 14 2023",
        #             'duration': '30',
        #             'distance': '300',
        #             'start_time': '11:00am',
        #             'elevation': '100',
        #             'activity_name': category,
        #             'activity_description': 'Shared laughs and made memories with friends.'
        #         }

        # if task_value == 'SportsTrackerTotalDurationForCategoryThisWeek':
        #     category= extract_from_template(
        #         "What was the total duration of {category} activities in the OpenTracks app this week? Assume the week starts from Monday. Express your answer in minutes as a single integer.",
        #         row['instruction']
        #     )
        #     category = list(category)[0]
        #     if category is not None:
        #         print(f"category: {category} ")
        #         params = {
        #             'category': category,
        #             'start_date': random.choice([
        #                 "October 9 2023",
        #                 "October 10 2023",
        #                 "October 11 2023",
        #                 "October 12 2023",
        #                 "October 13 2023",
        #                 "October 14 2023",
        #                 "October 15 2023"
        #             ]),
        #             'duration': random.choice([
        #                 "15",
        #                 "30",
        #                 "45",
        #                 "60",
        #                 "90",
        #                 "120",
        #                 "40",
        #                 "50",
        #                 "75",
        #                 "80",
        #                 "20",
        #                 "105",
        #                 "135"
        #             ]),
        #             'distance': random.choice([
        #                 "100",
        #                 "300",
        #                 "500",
        #                 "800",
        #                 "1000",
        #                 "1200",
        #                 "1500",
        #                 "2000",
        #                 "2500",
        #                 "3000"
        #             ]),
        #             'start_time': random.choice([
        #                 "8:00am",
        #                 "10:30am",
        #                 "5:00pm",
        #                 "6:30am",
        #                 "9:45am",
        #                 "2:15pm",
        #                 "7:00am",
        #                 "11:00am",
        #                 "4:00pm",
        #                 "1:30pm"
        #             ]),
        #             'elevation': random.choice([
        #                 "50",
        #                 "100",
        #                 "250",
        #                 "150",
        #                 "75",
        #                 "300",
        #                 "200"
        #             ]),
        #             'activity_name': random.choice([
        #                 "More tired than usual today",
        #                 "Need more strength and conditioning",
        #                 "Slow day",
        #                 "Laps around the lake",
        #                 "Trying and failing to keep up with John",
        #                 "Quick outing",
        #                 "Recovery day",
        #                 "Active Rest Day",
        #                 "Skill work"
        #             ]),
        #             'activity_description': random.choice([
        #                 "Shared laughs and made memories with friends.",
        #                 "Enjoyed a fun outing with good company.",
        #                 "Had a blast with my favorite people.",
        #                 "Created lasting memories that I'll cherish.",
        #                 "Experienced something unforgettable.",
        #                 "Captured moments that will bring a smile to my face.",
        #                 "Wandered off the beaten path.",
        #                 "Ventured into uncharted territory.",
        #                 "Stepped outside my comfort zone.",
        #                 "Pushed my boundaries and tried something different.",
        #                 "Tested my limits and grew as a person."
        #             ]),
        #         }
        # if task_value == 'SportsTrackerTotalDistanceForCategoryOverInterval':
        #     category,start_date,end_date= extract_from_template(
        #         "What was the total distance covered for {category} activities in the OpenTracks app from {start_date} to {end_date}? Express your answer as a single number in meters rounded to the nearest integer.",
        #     row['instruction'])
        #     if category is not None and start_date is not None and end_date is not None:
        #         print(f"category: {category} start_date: {start_date} end_date: {end_date} ")
        #         params = {
        #             'category': category,
        #             'start_date': start_date,
        #             'end_date': end_date,
        #             'duration': '30',
        #             'distance': '300',
        #             'start_time': '11:00am',
        #             'elevation': '100',
        #             'activity_name': 'Skill work',
        #             'activity_description': 'Shared laughs and made memories with friends.'
        #         }

        if task_value == 'RecipeDeleteMultipleRecipesWithConstraint':
            ingredient = extract_from_template('Delete the recipes from Broccoli app that use {ingredient} in the'
        ' directions.',row['instruction'])
            ingredient = list(ingredient)[0]
            if ingredient is not None:
                noise = sqlite_schema_utils.get_random_items(
                    6,
                    _generate_random_recipe,
                    replacement=False,
                    filter_fn=lambda r: ingredient not in r.directions.lower(),
                )
                n_rows = 3
                targets = []
                while n_rows > 0:
                    try:
                        targets = sqlite_schema_utils.get_random_items(
                            n_rows,
                            _generate_random_recipe,
                            replacement=False,
                            filter_fn=lambda r: ingredient in r.directions.lower(),
                        )
                        break
                    except ValueError:
                        n_rows -= 1
                params = {
                    sqlite_validators.ROW_OBJECTS: targets,
                    sqlite_validators.NOISE_ROW_OBJECTS: [],
                    'ingredient': ingredient,
                }
        # if task_value == 'VlcCreatePlaylist':
        #   playlist_name,files = extract_from_template(
        #       'Create a playlist titled "{playlist_name}" with the following files'
        #       ' in VLC (located in Internal Memory/VLCVideos), in order: {files}'
        #       ,
        #     row['instruction']
        #   )
        #
        #   if playlist_name is not None and files is not None:
        #       files = files.split(', ')
        #       params = {
        #           'playlist_name': playlist_name,
        #           'files': files,
        #           'noise_files': [generate_file_name() for _ in range(2)],
        #         }
        #
        #
        # if task_value == 'VlcCreateTwoPlaylists':
        #     playlist_name1, files1,playlist_name2, files2= extract_from_template(
        #         'Create a playlist titled "{playlist_name1}" with the following files in VLC (located in Internal Memory/VLCVideos), in order: {files1}. And then, create a playlist titled "{playlist_name2}" with the following files in VLC, in order: {files2}.',
        #         row['instruction']
        #     )
        #
        #     if playlist_name1 is not None and files1 is not None and playlist_name2 is not None and files2 is not None:
        #         files1 = files1.split(', ')
        #         files2 = files2.split(', ')
        #         params = {
        #             'playlist_name1': playlist_name1,
        #             'files1': files1,
        #             'playlist_name2': playlist_name2,
        #             'files2': files2,
        #             'noise_files1': [generate_file_name() for _ in range(2)],
        #             'noise_files2': [generate_file_name() for _ in range(2)],
        #         }
        #
        # if task_value == 'MarkorMergeNotes':
        #   options = extract_from_template(
        #     (
        #         "Merge the contents of Markor notes {file1_name}, {file2_name} and"
        #         " {file3_name} (in the same order) into a new Markor note named"
        #         " {new_file_name} and save it. Add a new line between the content of each"
        #         " note."
        #     ),
        #     row['instruction']
        #   )
        #   if options is not None:
        #     file1_name, file2_name, file3_name, new_file_name = options
        #     params = {
        #       'file1_name': file1_name,
        #       'file2_name': file2_name,
        #       'file3_name': file3_name,
        #       'new_file_name': new_file_name,
        #       "file1_content": user_data_generation.generate_random_string(20),
        #       "file2_content": user_data_generation.generate_random_string(20),
        #       "file3_content": user_data_generation.generate_random_string(20),
        #     }
        #
        # if task_value == 'SimpleCalendarDeleteOneEvent':
        #     year, month, day, hour, event_title = extract_from_template(
        #         (
        #             "In Simple Calendar Pro, delete the calendar event on"
        #             " {year}-{month}-{day} at {hour}h with the title '{event_title}'"
        #         ),
        #         row['instruction']
        #     )
        #     if year is not None and month is not None and day is not None and hour is not None and event_title is not None:
        #         year = int(year)
        #         month = int(month)
        #         day = int(day)
        #         hour = int(hour)
        #         # 创建一个 datetime 对象
        #         dt = datetime.datetime(year, month, day, hour)
        #         # 将 datetime 对象转换为 Unix 时间戳
        #         unix_timestamp = int(dt.timestamp()) + (8*60*60)
        #         event: sqlite_schema_utils.CalendarEvent = events_generator.generate_event(
        #             unix_timestamp,
        #             event_title
        #         )
        #         noise_events = generate_noise_events(
        #             [event],
        #             5,
        #             filter_fn=(
        #                 lambda candidate: (candidate.start_datetime != event.start_datetime)
        #                 and (candidate.title != event.title)
        #             ),
        #         )
        #         params = {
        #             'year': year,
        #             'month': month,
        #             'day': day,
        #             'hour': hour,
        #             'duration_mins': event.duration_mins,
        #             'event_title': event.title,
        #             'event_description': event.description,
        #             sqlite_validators.ROW_OBJECTS: [event],
        #             sqlite_validators.NOISE_ROW_OBJECTS: noise_events,
        #         }
        #
        # if task_value == 'ExpenseAddMultipleFromMarkor':
        #   params = task_type.generate_random_params()

        if params is None:
            continue

        env.reset(go_home=True)

        task = task_type(params)
        task.initialize_task(env)

        agent = Doubao(env, infer.DoubaoWrapper('doubao-1-5-ui-tars-250428'))
        if _AGENT_TYPE.value == 'openai':
            agent = t3a.T3A(env, infer.Gpt4Wrapper('gpt-4o-mini-2024-07-18'))
        if _AGENT_TYPE.value == 'openai4o':
            agent = t3a.T3A(env, infer.Gpt4Wrapper('gpt-4o-2024-11-20'))
        # file_count = 3
        # titles = [file for file in _NOTE_TITLES if file < params["original_name"]]
        # if len(titles) < 2:
        #     file_count = 1
        # agent = ChangeMarkorContent(env,infer.DoubaoWrapper('doubao-1-5-ui-tars-250428'))
        # agent.steps(file_count, params['original_name'], params['updated_content'], params['new_name'])

        # agent = MarkorAddNoteHeader(env,infer.DoubaoWrapper('doubao-1-5-ui-tars-250428'))
        # agent.steps(params['original_name'], params['header'], params['new_name'])
        
        agent = MarkorMergeNotes(env,infer.DoubaoWrapper('doubao-1-5-ui-tars-250428'))
        agent.steps(
          params['file1_name'],
          params['file2_name'],
          params['file3_name'],
          params['new_file_name']
        )
        
        agent_successful = True

        print('Goal: ' + str(task.goal))
        is_done = False
        for _ in range(min(int(task.complexity * 10), _MAX_STEP_COUNT.value)):
            response = agent.step(task.goal)
            if count_key_values(agent.history, 'action', 'wait') >= 2:
                break

            if response.done:
                is_done = True
                break
        agent_successful = is_done and task.is_successful(env) == 1

        # 任务跑完后，保存执行历史
        save_task_history(agent, str(row['id']), task.goal, agent_successful)

        print(
            f'{"Task Successful ✅" if agent_successful else "Task Failed ❌"};'
            f' {task.goal}'
        )

    env.close()


def ui_elements_to_xml(ui_elements: list, filename: str):
    """
    将 UIElement 列表保存为 XML 文件。
    """
    root = ET.Element("UIElements")
    for idx, element in enumerate(ui_elements):
        el = ET.SubElement(root, "Element", index=str(idx))
        for k, v in element.__dict__.items():
            if isinstance(v, (str, int, float, bool)):
                ET.SubElement(el, k).text = str(v)
    tree = ET.ElementTree(root)
    tree.write(filename, encoding="utf-8", xml_declaration=True)


def get_emulator_screen_size():
    try:
        # Execute adb command to get the display metrics
        output = subprocess.check_output(["adb", "shell", "wm", "size"])
        # Decode the output from bytes to string
        output = output.decode("utf-8").strip()
        # Extract the screen size from the output
        screen_size = output.split(": ")[1]
        width, height = map(int, screen_size.split('x'))
        return [width, height]
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        return None


def clean_element_number_text(text):
    # 替换 "UI element {number}" 为 "this element"
    cleaned_text = re.sub(r'UI element \d+', 'this element', text)
    # 删除 "(index {number})"
    cleaned_text = re.sub(r'\(index \d+\)', '', cleaned_text)
    return cleaned_text.strip()


def save_task_history(agent, task_id: str, task_goal: str, success: bool, output_dir: str = "./task_histories"):
    """
    将 agent.history 中每一步的 action、reason、summary 及截图
    编码为 base64，输出到 JSON 文件。
    """
    # 为每个 task 单独建一个文件夹
    task_dir = os.path.join(output_dir, task_id)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(task_dir, exist_ok=True)
    trajectory = []
    prompt_tokens = []
    completion_tokens = []

    screen_size = get_emulator_screen_size()
    if screen_size is None:
        screen_size = [1080, 2480]

    export = {
        'os': 'Android 13.0',
        "episode_id": task_id,
        "screen_resolution": screen_size,
        "instruction": task_goal,
        "trajectory": trajectory,
        "trajectory_type": 2
    }

    exist_retry_step = False

    for idx, step in enumerate(agent.history, start=1):
        reason, action = m3a_utils.parse_reason_action_output(step.get("action_output"))
        prompt_tokens.append(step.get('prompt_tokens'))
        completion_tokens.append(step.get('completion_tokens'))

        rec = {
            "step_id": idx,
            "action": step.get("action"),
            "think": clean_element_number_text(reason),
            # "think": reason,
            # "summary": step.get("summary"),
            "action_inputs": {
                "start_coords": step.get("start_coords"),
                "end_coords": step.get("end_coords"),
                "direction": step.get("direction"),
                "keycode": step.get("keycode"),
                "content": step.get("content"),
                'status': step.get("status"),
                'app_name': step.get("app_name")
            }
        }

        # 将 reason 转换为小写
        reason_lower = reason.lower()
        # 检查 reason_lower 是否包含 RETRY_KEYWORDS 中的任意字符
        if any(keyword in reason_lower for keyword in RETRY_KEYWORDS):
            exist_retry_step = True

        if step.get("action") == "status":
            if success:
                # if exist_retry_step:
                #     export['trajectory_type'] = 1
                # else:
                    export['trajectory_type'] = 0

        # 两张截图：before/after
        for key in ("before_screenshot", 'before_screenshot_mark'):
            img_array = step.get(key)
            if img_array is not None:
                filename = f"{idx}.png"  # e.g. before_screenshot_1.png
                if key == "before_screenshot_mark":
                    filename = f"{idx}-label.png"
                file_path = os.path.join(task_dir, filename)
                # 将 ndarray 转为 PIL 并保存
                Image.fromarray(img_array.astype("uint8")).save(file_path)
                if key == "before_screenshot_mark":
                    rec['observation_mark'] = filename
                else:
                    rec['observation'] = filename
        trajectory.append(rec)

        # 保存页面结构 XML
        for key in ("before_element_list", ''):
            elements = step.get(key)
            if elements is not None:
                filename = f"{idx}.xml"
                file_path = os.path.join(task_dir, filename)
                ui_elements_to_xml(elements, file_path)
                rec['xml'] = filename

    out_path = os.path.join(task_dir, "task.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)
    print(f"✅ Task history saved to {out_path}")
    # with open('cost_token', "w", encoding="utf-8") as f:
    #     json.dump({
    #         'prompt_tokens': sum(prompt_tokens),
    #         'prompt_tokens_avg': sum(prompt_tokens) / len(prompt_tokens),
    #         'completion_tokens': sum(completion_tokens),
    #         'completion_tokens_avg': sum(completion_tokens) / len(completion_tokens)
    #     }, f, ensure_ascii=False, indent=2)

    # print('消耗prompt_tokens Token：', sum(prompt_tokens))
    # print('消耗completion_tokens Token：', sum(completion_tokens))


def main(argv: Sequence[str]) -> None:
    del argv
    _main()


if __name__ == '__main__':
    app.run(main)

    # if task_value == 'ContactsAddContact':
    #     name, number = extract_name_and_number(row['instruction'])
    #     if name is not None and number is not None:
    #         print(f"Name: {name}, Number: {number}")
    #         params = {
    #             'name': name,
    #             'number': number
    #         }
    #
    # if task_value == 'ContactsNewContactDraft':
    #     result = extract_contact_details(row['instruction'])
    #     if name is not None and number is not None:
    #         print(f"Name: {result}")
    #         params = {
    #             "first": result['first_name'],
    #             "last": result['last_name'],
    #             "phone": result['phone'],
    #             "phone_label": result['phone_label'],
    #         }

    # if task_value == 'NotesIsTodo':
    #     title = extract_from_template(
    #         "Is the note titled '{title}' in the Joplin app marked as a todo item? Respond with either 'True' if it is a todo or 'False' if not.",
    #         row['instruction'])
    #     title = list(title)[0]
    #     if title is not None:
    #         print(f"title: {title},")
    #         params = {
    #             'title': title,
    #             'is_todo': "True",
    #             'body': 'Buy milk, eggs, bread, and cereal from the grocery store.',
    #             'folder': 'School'
    #         }
    #
    # if task_value == 'NotesMeetingAttendeeCount':
    #     title = extract_from_template(
    #         "How many attendees were present in the meeting titled '{title}' in the Joplin app? Express your answer as just a single number.",
    #         row['instruction'])
    #     title = list(title)[0]
    #     if title is not None:
    #         print(f"title: {title},")
    #         params = {
    #             'title': title,
    #             'attendee_count': '5',
    #             'is_todo': "True",
    #             'body': 'Meeting Notes:\n- Discussed project milestones\n- Assigned action items to team members\n- Reviewed budget allocation\n- Decided on next meeting date\n- Attended by {attendee_count} participants\n',
    #         }

    # if task_value == 'NotesRecipeIngredientCount':
    #     ingredient, title= extract_from_template(
    #         "What quantity of {ingredient} do I need for the recipe '{title}' in the Joplin app? Express your answer in the format <amount> <unit> where both the amount and unit exactly match the format in the recipe.",
    #         row['instruction'])
    #     if title is not None and ingredient is not None:
    #         print(f"title: {title},ingredient{ingredient} ")
    #         ingredient_quantity = '2 cups'
    #         params = {
    #             'title': title,
    #             'ingredient_quantity': ingredient_quantity,
    #             'ingredient': ingredient,
    #             'body': f'Ingredients:\n- 1 cup all-purpose flour\n- 1/2 cup granulated sugar\n- {ingredient_quantity} {ingredient}\n- 1 teaspoon baking powder\n- 2 tablespoons unsalted butter\n- 1/4 teaspoon salt\n\nInstructions:\n1. Preheat oven to 350°F (175°C).\n2. In a mixing bowl, combine all-purpose flour, granulated sugar, and baking powder.\n3. Add unsalted butter and salt, mixing until well combined.\n4. Grease a baking dish and pour the mixture into it.\n5. Bake in preheated oven for 25-30 minutes, or until golden brown.\n6. Let cool for a few minutes before serving.\n',
    #         }

    # if task_value == 'NotesTodoItemCount':
    #     folder= extract_from_template(
    #         "How many to-dos do I have in the '{folder}' folder in the Joplin app? Express your answer as just a single number.",
    #         row['instruction'])
    #     folder = list(folder)[0]
    #     if folder is not None:
    #         print(f"title: {folder} ")
    #         params = {
    #             'title': 'Personal Goals',
    #             'folder': folder,
    #             'body': 'Discuss project updates, assign tasks, and review deadlines.',
    #         }

    # if task_value == 'MarkorAddNoteHeader':
    #     result = extract_from_template(
    #         (
    #             "Update the Markor note {original_name} by adding the following text,"
    #             ' along with a new blank line before the existing content: "{header}",'
    #             " and rename it to {new_name}."
    #         ),
    #         row['instruction']
    #     )
    #     if result is not None:
    #       original_name, header, new_name = result
    #       if original_name is not None and new_name is not None and header is not None:
    #         original_content = generate_random_sentence()
    #         params = {
    #           "original_name": original_name,
    #           "original_content": original_content,
    #           "new_name": new_name,
    #           "header": header,
    #         }

    # if task_value == 'MarkorChangeNoteContent':
    #     result = extract_from_template(
    #         (
    #             'Update the content of {original_name} to "{updated_content}" in Markor'
    #             " and change its name to {new_name}."
    #         ),
    #         row['instruction']
    #     )
    #     if result is not None:
    #       original_name, updated_content, new_name = result
    #       if original_name is not None and new_name is not None and updated_content is not None:
    #         params = {
    #           "original_name": original_name,
    #           "updated_content": updated_content,
    #           "new_name": new_name,
    #         }

    # if task_value == 'MarkorCreateNote':
    #     file_name,text = extract_from_template(
    #         "Create a new note in Markor named {file_name} with the following text: {text}",
    #         row['instruction'])
    #     if file_name is not None and text is not None:
    #         print(f"text: {text},file_name:{file_name}")
    #         params = {
    #             'file_name': file_name,
    #             'text': text,
    #         }
    #
    # if task_value == 'MarkorCreateNoteAndSms':
    #     file_name,text,number = extract_from_template(
    #         "Create a new note in Markor named {file_name} with the following text:{text}. Share the entire content of the note with the phone number {number} via SMS using Simple SMS Messenger",
    #         row['instruction'])
    #     if file_name is not None and text is not None and number is not None:
    #         print(f"text: {text},file_name:{file_name},number:{number}")
    #         params = {
    #             'file_name': file_name,
    #             'text': text,
    #             'number': number
    #         }
    #
    # if task_value == 'MarkorCreateNoteFromClipboard':
    #     file_name= extract_from_template(
    #           "Create a note in Markor named {file_name}. Perform a paste operation in the note and save the note.",
    #         row['instruction'])
    #     file_name = list(file_name)[0]
    #     if file_name is not None :
    #         print(f"file_name:{file_name}")
    #         params = {
    #             'file_name': file_name,
    #             'file_content':user_data_generation.generate_random_string(10)
    #         }
    #
    # if task_value == 'SimpleCalendarAddOneEvent':
    #     year, month, day, hour, title, description, duration = extract_calendar_event_info(row['instruction'])
    #     if year is not None and month is not None and day is not None and hour is not None and title is not None and description is not None and duration is not None:
    #         print(f"year: {year}, month: {month}")
    #
    #         start_ts = _create_unix_ts(day=day,hour=hour)
    #         end_ts = start_ts + duration * 60
    #         event = sqlite_schema_utils.CalendarEvent(
    #             start_ts=start_ts,
    #             end_ts=end_ts,
    #             title=title,
    #             description=description
    #         )
    #         n_noise_events = random.randint(0, 20)
    #         params = {
    #             "year": year,
    #             'month': month,
    #             "day": day,
    #             'hour': hour,
    #             'duration_mins': duration,
    #             'event_title': title,
    #             'event_description': description,
    #             'row_objects': [event],
    #             'noise_row_objects': generate_noise_events(
    #                 [event], n_noise_events
    #             ),
    #         }
    # if task_value == 'ContactsNewContactDraft':
    #     result = extract_contact_details(row['instruction'])
    #     if result['first_name'] is not None and result['last_name'] is not None and result['phone'] is not None and result['phone_label'] is not None:
    #         print(f"Name: {result}")
    #         params = {
    #             "first": result['first_name'],
    #             "last": result['last_name'],
    #             "phone": result['phone'],
    #             "phone_label": result['phone_label'],
    #         }
    #
    # if task_value == 'SimpleCalendarAddOneEventInTwoWeeks':
    #     hour, title, description, duration = extract_event_info(row['instruction'])
    #     if hour is not None and title is not None and description is not None and duration is not None:
    #         print(f"hour: {hour}, title: {title}")
    #
    #         start_ts = _create_unix_ts(day= device_constants.DT.day + 14,hour=hour)
    #         end_ts = start_ts + duration * 60
    #         event = sqlite_schema_utils.CalendarEvent(
    #             start_ts=start_ts,
    #             end_ts=end_ts,
    #             title=title,
    #             description=description
    #         )
    #         n_noise_events = random.randint(0, 20)
    #         params = {
    #             "year": device_constants.DT.year,
    #             'month': device_constants.DT.month,
    #             "day": event.start_datetime.day,
    #             'hour': hour,
    #             'duration_mins': duration,
    #             'event_title': title,
    #             'event_description': description,
    #             'row_objects': [event],
    #             'noise_row_objects': generate_noise_events(
    #                 [event], n_noise_events
    #             ),
    #         }
    #
    # if task_value == 'SimpleCalendarAddOneEventTomorrow':
    #     hour, title, description, duration = extract_event_tuple(row['instruction'])
    #     if hour is not None and title is not None and description is not None and duration is not None:
    #         print(f"hour: {hour}, title: {title}, description: {description}")
    #
    #         start_ts = _create_unix_ts(day= device_constants.DT.day + 1,hour=hour)
    #         end_ts = start_ts + duration * 60
    #         event = sqlite_schema_utils.CalendarEvent(
    #             start_ts=start_ts,
    #             end_ts=end_ts,
    #             title=title,
    #             description=description
    #         )
    #         n_noise_events = random.randint(0, 20)
    #         params = {
    #             "year": device_constants.DT.year,
    #             'month': device_constants.DT.month,
    #             "day": event.start_datetime.day,
    #             'hour': hour,
    #             'duration_mins': duration,
    #             'event_title': title,
    #             'event_description': description,
    #             'row_objects': [event],
    #             'noise_row_objects': generate_noise_events(
    #                 [event], n_noise_events
    #             ),
    #         }
    #
    # if task_value == 'SimpleCalendarAddOneEventRelativeDay':
    #     week, hour, title, description, duration = extract_event_details(row['instruction'])
    #     if hour is not None and title is not None and description is not None and duration is not None and week is not None:
    #         print(f"hour: {hour}, title: {title}, week: {week}")
    #
    #         date_num = weekday_to_number(week)
    #         start_ts = _create_unix_ts(day= device_constants.DT.day + date_num,hour=hour)
    #         end_ts = start_ts + duration * 60
    #         event = sqlite_schema_utils.CalendarEvent(
    #             start_ts=start_ts,
    #             end_ts=end_ts,
    #             title=title,
    #             description=description
    #         )
    #         n_noise_events = random.randint(0, 20)
    #         params = {
    #             "year": device_constants.DT.year,
    #             'month': device_constants.DT.month,
    #             "day": event.start_datetime.day,
    #             'hour': hour,
    #             'duration_mins': duration,
    #             'event_title': title,
    #             'event_description': description,
    #             'row_objects': [event],
    #             'noise_row_objects': generate_noise_events(
    #                 [event], n_noise_events
    #             ),
    #         }
    #
    # if task_value == 'SimpleCalendarAddRepeatingEvent':
    #     title, year, month, day, hour, recurrence, duration, description = extract_repeat_event_info(row['instruction'])
    #     if year is not None and month is not None and day is not None and hour is not None and title is not None and description is not None and duration is not None and recurrence is not None:
    #         print(f"year: {year}, month: {month}")
    #
    #         start_ts = _create_unix_ts(day=day, hour=hour)
    #         end_ts = start_ts + duration * 60
    #         template = sqlite_schema_utils.CalendarEvent(
    #             start_ts=start_ts,
    #             end_ts=end_ts,
    #             title=title,
    #             description=description
    #         )
    #         if recurrence == "weekly":
    #             repeat_rule = calendar_utils.generate_simple_calendar_weekly_repeat_rule(
    #                 template.start_datetime.isoweekday()
    #             )
    #         else:
    #             repeat_rule = 0
    #
    #         event = dataclasses.replace(
    #             template,
    #             repeat_interval=_REPEAT_INTERVALS[recurrence],
    #             repeat_rule=repeat_rule,
    #         )
    #
    #         n_noise_events = random.randint(0, 20)
    #         params = {
    #             "year": year,
    #             'month': month,
    #             "day": day,
    #             'hour': hour,
    #             'duration_mins': duration,
    #             'event_title': title,
    #             'event_description': description,
    #             'row_objects': [event],
    #             'noise_row_objects': generate_noise_events(
    #                 [event], n_noise_events
    #             ),
    #             'repeat_rule': recurrence
    #         }
    #
    # if task_value == 'SimpleCalendarEventOnDateAtTime':
    #     date, time = extract_from_template(
    #         'What is on my schedule for {date} at {time} in Simple Calendar Pro? Answer with the titles only. If there are multiples titles, format your answer in a comma separated list.',
    #         row['instruction'])
    #     if date is not None and time is not None:
    #         print(f"date: {date}, time: {time}")
    #         params = {
    #             'date': date,
    #             'time': time,
    #             'duration': '30 m',
    #             'title': 'Team Meeting'
    #         }
    #
    # if task_value == 'SimpleCalendarAnyEventsOnDate':
    #     date = extract_from_template(
    #         "Do I have any events {date} in Simple Calendar Pro? Answer with the titles only. If there are multiples titles, format your answer in a comma separated list.",
    #         row['instruction'])
    #     if date is not None:
    #         print(f"date: {date}")
    #         params = {
    #             'date': date,
    #             'title': '1-on-1 with Manager',
    #             'time': '11:00am',
    #         }
    #
    # if task_value == 'SimpleCalendarEventsInNextWeek':
    #     extract_from_template(
    #         "What events do I have in the next week in Simple Calendar Pro? Answer with the titles only. If there are multiples titles, format your answer in a comma separated list.",
    #         row['instruction'])
    #     params = {
    #         'date': 'October 16 2023',
    #         'duration': '30 m',
    #         'title': 'Sports game',
    #         'person': 'Amanda',
    #         'time': '11:00am',
    #     }
    #
    # if task_value == 'SimpleCalendarEventsInTimeRange':
    #     start_time, date = extract_from_template(
    #         "Do I have any events between {start_time} and 8pm {date} in Simple Calendar Pro? Answer with the titles only. If there are multiples titles, format your answer in a comma separated list.",
    #         row['instruction'])
    #     if date is not None and start_time is not None:
    #         print(f"start_time: {start_time}, date: {date}")
    #         params = {
    #             'start_time': start_time,
    #             'date': date,
    #             'duration': '30 m',
    #             'title': 'Sports game',
    #         }
    #
    # if task_value == 'SimpleCalendarEventsOnDate':
    #     date = extract_from_template(
    #         "What events do I have {date} in Simple Calendar Pro? Answer with the titles only. If there are multiple titles, format your answer as a comma separated list.",
    #         row['instruction'])
    #     date = list(date)[0]
    #     if date is not None:
    #         print(f"date: {date}")
    #         params = {
    #             'date': date,
    #             'duration': '30 m',
    #             'title': 'Sports game',
    #             'time': '11:00am',
    #         }
    #
    # if task_value == 'SimpleCalendarFirstEventAfterStartTime':
    #     time, date = extract_from_template(
    #         "What is my first event after {time} {date} in Simple Calendar Pro? Answer with the titles only. If there are multiples titles, format your answer in a comma separated list.",
    #         row['instruction'])
    #     if date is not None and time is not None:
    #         print(f"date: {date},time :{time}")
    #         params = {
    #             'date': date,
    #             'time': time,
    #             'duration': '30 m',
    #             'title': 'Sports game',
    #         }
    #
    # if task_value == 'SimpleCalendarLocationOfEvent':
    #     title = extract_from_template(
    #         "What is the location of my {title} event in Simple Calendar Pro? Answer with the location only.",
    #         row['instruction'])
    #     title = list(title)[0]
    #     if title is not None:
    #         print(f"title: {title},")
    #         params = {
    #             'title': title,
    #             'location': 'Conference Room A',
    #             'date': 'October 16 2023',
    #             'time': '1:30pm'
    #         }
    #
    # if task_value == 'SimpleCalendarNextEvent':
    #     print(f"SimpleCalendarNextEvent,")
    #     params = {
    #         'time': '7:15pm',
    #         'duration': '30 m',
    #         'title': 'Sports game',
    #     }
    #
    # if task_value == 'SimpleCalendarNextMeetingWithPerson':
    #     person = extract_from_template(
    #         "When is my next meeting with {person} in Simple Calendar Pro? Express your answer in the format <month name> <day> <year> <hour in 24-hour format>:<minutes>.",
    #         row['instruction'])
    #     person = list(person)[0]
    #     if person is not None:
    #         print(f"person: {person},")
    #         params = {
    #             'person': person,
    #             'time': '7:15pm',
    #             'date': 'October 16 2023',
    #         }
    # if task_value == 'ExpenseDeleteMultiple' or task_value == 'ExpenseDeleteMultiple2' or task_value == 'ExpenseDeleteSingle':
    #     names = extract_expense_delete_details(row['instruction'])
    #     if len(names) > 0:
    #         target_rows = []
    #         for name in names:
    #             if name is not None:
    #                 category_id = random.choice(
    #                     list(sqlite_schema_utils.Expense.category_id_to_name.keys())
    #                 )
    #                 amount = random.randint(
    #                     1000, 50000
    #                 )  # Amount in cents (e.g., $10.00 - $500.00)
    #                 note = random.choice(_NOTES)
    #                 expense_unix_time_s = _get_random_timestamp()
    #                 expense_unix_time_ms = expense_unix_time_s * 1000
    #                 target_rows.append(sqlite_schema_utils.Expense(
    #                     name,
    #                     amount,
    #                     category_id,
    #                     note,
    #                     expense_unix_time_ms,
    #                     expense_unix_time_ms,
    #                 ))
    #                 if task_value == 'ExpenseDeleteMultiple2':
    #                   noise_rows = sqlite_schema_utils.get_random_items(
    #                       20,
    #                       _generate_expense,
    #                       replacement=False,
    #                       filter_fn=lambda r: all(r.name != t.name for t in target_rows),
    #                   )
    #                 elif task_value == 'ExpenseDeleteMultiple':
    #                   noise_rows: list[sqlite_schema_utils.Expense] = []
    #         if task_value == 'ExpenseDeleteSingle' and len(target_rows) > 0:
    #           params = {
    #             sqlite_validators.ROW_OBJECTS: target_rows,
    #           }
    #         elif len(target_rows) > 0:
    #           params = {
    #             sqlite_validators.ROW_OBJECTS: target_rows,
    #             sqlite_validators.NOISE_ROW_OBJECTS: noise_rows,
    #           }

    # if task_value == 'SimpleSmsReply':
    #     number, message = extract_sms_reply(row['instruction'])
    #     if message is not None and number is not None:
    #         print(f"message: {message}, Number: {number}")
    #         params = {
    #             'message': message,
    #             'number': number
    #         }

    # if task_value == 'SimpleSmsSend':
    #     number, message = extract_sms_info(row['instruction'])
    #     if message is not None and number is not None:
    #         print(f"message: {message}, Number: {number}")
    #         params = {
    #             'message': message,
    #             'number': number
    #         }

    # if task_value == 'SimpleSmsSendClipboardContent':
    #     number = extract_sms_number(row['instruction'])
    #     message = random.choice(user_data_generation.RANDOM_SENTENCES)
    #     if number is not None:
    #         print(f"Number: {number}")
    #         params = {
    #             'message': message,
    #             'number': number
    #         }

    # if task_value == 'SimpleSmsReplyMostRecent':
    #     message = extract_sms_message(row['instruction'])
    #     number = user_data_generation.generate_random_number()
    #     if message is not None:
    #         print(f"message: {message}")
    #         params = {
    #             'message': message,
    #             'number': number
    #         }

    # if task_value == 'FilesDeleteFile':
    #     file_name, subfolder = extract_file_delete(row['instruction'])
    #     if file_name is not None and subfolder is not None:
    #         print(f"file_name: {file_name}, subfolder: {subfolder}")
    #         noise_candidates = user_data_generation.EMULATOR_DIRECTORIES[subfolder]
    #         params = {
    #             "file_name": file_name,
    #             "subfolder": subfolder,
    #             "noise_candidates": noise_candidates,
    #         }

    # if task_value == 'FilesMoveFile':
    #     file_name, source_folder, destination_folder = extract_file_move(row['instruction'])
    #     if file_name is not None and source_folder is not None and destination_folder is not None:
    #         print(f"file_name: {file_name}, source_folder: {source_folder}, destination_folder: {destination_folder}")
    #         noise_candidates = user_data_generation.EMULATOR_DIRECTORIES[source_folder]
    #         params = {
    #             "file_name": file_name,
    #             "source_folder": source_folder,
    #             "destination_folder": destination_folder,
    #             "noise_candidates": noise_candidates,
    #         }

    # if task_value == 'RetroCreatePlaylist':
    #     playlist_name, names = extract_retro_playlist_create(row['instruction'])
    #     if playlist_name is not None and names is not None:
    #         print(f'playlist_name: {playlist_name}, names: {names}')
    #         files = [f'{name}.mp3' for name in names]
    #         params = {
    #             'playlist_name': playlist_name,
    #             'files': files,
    #             'noise_files': [],
    #         }

    # if task_value == 'RetroPlayingQueue':
    #     names = extract_retro_play(row['instruction'].replace(' to my playing queue in Retro music.', ''))
    #     if names is not None:
    #         print(f'names: {names}')
    #         playlist_name = _generate_playlist_name()
    #         files = [f'{name}.mp3' for name in names]
    #         params = {
    #             'playlist_name': playlist_name,
    #             'files': files,
    #             'noise_files': [],
    #         }

    # if task_value == 'RetroPlaylistDuration':
    #     playlist_name, = extract_from_template(
    #         'Create a playlist in Retro Music titled "{playlist_name}" with a duration between 45 and 50 minutes using the provided songs.',
    #         row['instruction'],
    #     )
    #     if playlist_name is not None:
    #         print(f'playlist_name: {playlist_name}')
    #         files = ['Beyond the Horizon.mp3', 'Bright Lights.mp3'] #固定
    #         random_files = [f'{name}.mp3' for name in random.sample(_SONGS, 15)]
    #         noise_files = []
    #         for name in random_files:
    #             if name not in files:
    #                 noise_files.append(name)
    #         print(f'noise files: {noise_files}')
    #         params = {
    #             'playlist_name': playlist_name,
    #             'files': files,
    #             'noise_files': noise_files,
    #         }
    #
    # if task_value == 'RetroSavePlaylist':
    #     playlist_name, names = extract_retro_playlist_create(row['instruction'].replace('. Then export the playlist to the Downloads directory on the device.', ''))
    #     if playlist_name is not None and names is not None:
    #         print(f'playlist_name: {playlist_name}, names: {names}')
    #         files = [f'{name}.mp3' for name in names]
    #         params = {
    #             'playlist_name': playlist_name,
    #             'files': files,
    #             'noise_files': [],
    #         }

    # if task_value == 'VlcCreatePlaylist':
    #     playlist_name, files = extract_vlc_playlist_create(row['instruction'])
    #     if playlist_name is not None and files:
    #         print(f"Playlist Name: {playlist_name}, Files: {files}")
    #         params = {
    #             'playlist_name': playlist_name,
    #             'files': files,
    #             'noise_files': [generate_file_name() for _ in range(len(files))]
    #         }

    # if task_value == 'MarkorDeleteNote':
    #     date = extract_from_template(
    #         "Delete the note in Markor named {file_name}.",
    #         row['instruction'])
    #     date = list(date)[0]
    #     if date is not None:
    #         print(f"date: {date}")
    #         params = {
    #             'file_name': date,
    #             "noise_candidates": _NOTE_TITLES
    #         }

    # if task_value == 'MarkorEditNote':
    #   edit_type = None
    #   param1 = None
    #   param2 = None
    #   if "the top" in row['instruction']:
    #     edit_type = "header"
    #     param1, param2 = extract_from_template(
    #         "Edit {file_name} in Markor. Add to the top of the note {header}",
    #         row['instruction']
    #     )
    #   elif "the bottom" in row['instruction']:
    #     edit_type = "footer"
    #     param1, param2 = extract_from_template(
    #         "Edit {file_name} in Markor. Add to the bottom of the note {footer}",
    #         row['instruction']
    #     )
    #   elif "replace the text" in row['instruction'].lower():
    #     edit_type = "replace"
    #     param1, param2 = extract_from_template(
    #         "Edit {file_name} in Markor. Replace the text with {replace_text}.",
    #         row['instruction']
    #     )
    #   print(f"edit_type: {edit_type}")
    #   print(f"param1: {param1}, param2: {param2}")
    #   if param1 is not None and param2 is not None:
    #     params = {
    #       'file_name': param1,
    #       'edit_type': edit_type,
    #     }

    #     if edit_type == "header":
    #       params["header"] = param2
    #     elif edit_type == "footer":
    #       params["footer"] = param2
    #     elif edit_type == "replace":
    #       params["replace_text"] = param2

    # if task_value == 'RecipeAddMultipleRecipes' or task_value == 'RecipeAddSingleRecipe':
    #     recipe_type, recipes = parse_recipes(row['instruction'])
    #     if recipe_type is not None and len(recipes) > 0:
    #         target_rows: list[sqlite_schema_utils.Recipe] = []
    #         for recipe in recipes:
    #             if recipe['title'] is not None and recipe['description'] is not None and recipe['servings'] is not None and recipe['preparationTime'] is not None and recipe['ingredients'] is not None and recipe['directions'] is not None:
    #                 target_rows.append(sqlite_schema_utils.Recipe(
    #                     recipe['title'],
    #                     recipe['description'],
    #                     recipe['servings'],
    #                     recipe['preparationTime'],
    #                     '',
    #                     recipe['ingredients'],
    #                     recipe['directions'],
    #                 ))
    #         if task_value == 'RecipeAddSingleRecipe':
    #           noise_rows = []
    #         else:
    #           noise_rows = sqlite_schema_utils.get_random_items(
    #             5,
    #             _generate_random_recipe,
    #             replacement=False,
    #             filter_fn=lambda r: any([r.title != t.title for t in target_rows]),
    #         )
    #         params = {
    #             sqlite_validators.ROW_OBJECTS: target_rows,
    #             sqlite_validators.NOISE_ROW_OBJECTS: noise_rows,
    #             "text_representation_type": recipe_type,
    #         }

    # if task_value == 'SimpleCalendarDeleteEvents':
    #     year, month, day = extract_from_template(
    #         (
    #             "In Simple Calendar Pro, delete all the calendar events on"
    #             " {year}-{month}-{day}"
    #         ),
    #         row['instruction']
    #     )
    #     if year is not None and month is not None and day is not None:
    #         year = int(year)
    #         month = int(month)
    #         day = int(day)
    #         events: list[sqlite_schema_utils.CalendarEvent] = [
    #             events_generator.generate_event(
    #                 datetime_utils.create_random_october_2023_unix_ts(
    #                     start_day=day, end_day=day
    #                 )
    #             )
    #             for _ in range(3)
    #         ]
    #         noise_events: list[sqlite_schema_utils.CalendarEvent] = generate_noise_events(
    #             events,
    #             5,
    #             filter_fn=lambda candidate: candidate.start_datetime.day
    #             not in (target.start_datetime.day for target in events),
    #         )
    #         params = {
    #             'year': year,
    #             'month': month,
    #             'day': day,
    #             sqlite_validators.ROW_OBJECTS: events,
    #             sqlite_validators.NOISE_ROW_OBJECTS: noise_events,
    #         }



    # if task_value == 'SimpleCalendarDeleteEventsOnRelativeDay':
    #     day_of_week = extract_from_template(
    #         (
    #             "In Simple Calendar Pro, delete all events scheduled for this"
    #             " {day_of_week}."
    #         ),
    #         row['instruction']
    #     )
    #     day_of_week = list(day_of_week)[0]
    #     if day_of_week is not None:
    #         target_date = get_day_of_week(day_of_week.lower())
    #         events: list[sqlite_schema_utils.CalendarEvent] = [
    #             events_generator.generate_event(
    #                 datetime_utils.create_random_october_2023_unix_ts(
    #                     start_day=target_date['day'], end_day=target_date['day']
    #                 )
    #             )
    #             for _ in range(2)
    #         ]
    #         noise_events: list[sqlite_schema_utils.CalendarEvent] = generate_noise_events(
    #             events,
    #             5,
    #             filter_fn=lambda candidate: candidate.start_datetime.day
    #             not in (target.start_datetime.day for target in events),
    #         )
    #         params = {
    #             'year': target_date['year'],
    #             'month': target_date['month'],
    #             'day': target_date['day'],
    #             'day_of_week': day_of_week,
    #             sqlite_validators.ROW_OBJECTS: events,
    #             sqlite_validators.NOISE_ROW_OBJECTS: noise_events,
    #         }

    # if task_value == 'MarkorMoveNote':
    #   options = extract_from_template(
    #     (
    #         "In Markor, move the note {file_name} from {source_folder} to"
    #         " {destination_folder}."
    #     ),
    #     row['instruction']
    #   )
    #   if options is not None:
    #     file_name, source_folder, destination_folder = options
    #     params = {
    #       'file_name': file_name,
    #       'source_folder': source_folder,
    #       'destination_folder': destination_folder,
    #       'noise_candidates': _NOTE_TITLES,
    #     }

    # if task_value == 'RecipeDeleteMultipleRecipesWithNoise':
    #     titles = extract_from_template('Delete the following recipes from Broccoli app: {titles}.', row['instruction'])
    #     titles = list(titles)[0]
    #     if titles is None:
    #         continue
    #
    #     titles = titles.split(',')
    #     if len(titles) > 0:
    #         target_rows: list[sqlite_schema_utils.Recipe] = []
    #         noise_rows: list[sqlite_schema_utils.Recipe] = []
    #         for title in titles:
    #             candidate = _generate_random_recipe()
    #             candidate = dataclasses.replace(candidate, title=title)
    #             target_rows.append(candidate)
    #
    #         while len(noise_rows) < 3:
    #             candidate = _generate_random_recipe()
    #             if not any([candidate.title == r.title for r in target_rows]) and not any(
    #                     [candidate.title == r.title for r in noise_rows]):
    #                 noise_rows.append(candidate)
    #
    #         params = {
    #             sqlite_validators.ROW_OBJECTS: target_rows,
    #             sqlite_validators.NOISE_ROW_OBJECTS: noise_rows,
    #         }
    #
    # if task_value == 'RecipeDeleteMultipleRecipes':
    #     titles = extract_from_template('Delete the following recipes from Broccoli app: {titles}.', row['instruction'])
    #     titles = list(titles)[0]
    #     if titles is None:
    #         continue
    #
    #     titles = titles.split(',')
    #     if len(titles) > 0:
    #         target_rows: list[sqlite_schema_utils.Recipe] = []
    #         for title in titles:
    #             candidate = _generate_random_recipe()
    #             candidate = dataclasses.replace(candidate, title=title)
    #             target_rows.append(candidate)
    #
    #         params = {
    #             sqlite_validators.ROW_OBJECTS: target_rows,
    #             # sqlite_validators.NOISE_ROW_OBJECTS: noise_rows,
    #         }
    #
    # if task_value == 'RecipeDeleteSingleRecipe':
    #     titles = extract_from_template('Delete the following recipes from Broccoli app: {titles}.', row['instruction'])
    #     titles = list(titles)[0]
    #     if titles is None:
    #         continue
    #
    #     titles = titles.split(',')
    #     if len(titles) > 0:
    #         target_rows: list[sqlite_schema_utils.Recipe] = []
    #         for title in titles:
    #             candidate = _generate_random_recipe()
    #             candidate = dataclasses.replace(candidate, title=title)
    #             target_rows.append(candidate)
    #
    #         params = {
    #             sqlite_validators.ROW_OBJECTS: target_rows,
    #             # sqlite_validators.NOISE_ROW_OBJECTS: noise_rows,
    #         }
    #
    # if task_value == 'RecipeDeleteSingleWithRecipeWithNoise':
    #     titles = extract_from_template('Delete the following recipes from Broccoli app: {titles}.', row['instruction'])
    #     titles = list(titles)[0]
    #     if titles is None:
    #         continue
    #
    #     titles = titles.split(',')
    #     if len(titles) > 0:
    #         target_rows: list[sqlite_schema_utils.Recipe] = []
    #         noise_rows: list[sqlite_schema_utils.Recipe] = []
    #         for title in titles:
    #             candidate = _generate_random_recipe()
    #             candidate = dataclasses.replace(candidate, title=title)
    #             target_rows.append(candidate)
    #
    #         while len(noise_rows) < 3:
    #             candidate = _generate_random_recipe()
    #             if not any([candidate.title == r.title for r in target_rows]) and not any(
    #                     [candidate.title == r.title for r in noise_rows]):
    #                 noise_rows.append(candidate)
    #
    #         params = {
    #             sqlite_validators.ROW_OBJECTS: target_rows,
    #             sqlite_validators.NOISE_ROW_OBJECTS: noise_rows,
    #         }
