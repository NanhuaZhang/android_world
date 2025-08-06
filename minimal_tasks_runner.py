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
import io
import json
from collections.abc import Sequence
import os
import random
import re
from typing import Type

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
from android_world.task_evals.utils import sqlite_schema_utils
from android_world.utils.datetime_utils import create_random_october_2023_unix_ts, _create_unix_ts
from android_world.task_evals.single.vlc import generate_file_name
from android_world.task_evals.utils import sqlite_schema_utils, user_data_generation

from android_world.task_evals.common_validators import sqlite_validators
from android_world.task_evals.single.expense import _get_random_timestamp, _generate_expense

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


def _find_adb_directory() -> str:
    """Returns the directory where adb is located."""
    potential_paths = [
        os.path.expanduser('~/Library/Android/sdk/platform-tools/adb'),
        os.path.expanduser('~/Android/Sdk/platform-tools/adb'),
        os.path.expanduser('~/Android/platform-tools/adb'),
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
    False,
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
    30,
    'The max step count of the agent.',
)


# 定义正则表达式提取函数
def extract_name_and_number(instruction):
    # 提取电话号码（+ 和数字）
    number_match = re.search(r'\+?\d{10,}', instruction)
    number = number_match.group() if number_match else None

    # 提取姓名（寻找 “contact for xxx.” 的格式）
    name_match = re.search(r'contact for ([A-Z][a-z]+\s[A-Z][a-z]+)', instruction)
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
    weekday_match = re.search(r'\b(on|for)?\s*(this\s+)?(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b', text, re.IGNORECASE)
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

def extract_expense_add_multiple(instruction):
    # 定义正则表达式
    pattern_a = r'Expense:\s*(.+?)\s*amount_dollars:\s*\$(\d+\.\d{2})\s*category_name:\s*(.+?)\s*note:\s*(.+?)(?=\n\n|$)'
    pattern_b = r'(.+?)\|(\$?\d+\.\d{2})\|(.+?)\|(.+)'
    pattern_c = r'Expense:\s*(.+?)\s*amount_dollars:\s*\$(\d+\.\d{2})\s*category_name:\s*(.+?)\s*note:\s*(.+?)(?=\n\n|$)'

    results = []

    # 尝试匹配格式 a
    expenses_a = re.findall(pattern_a, instruction, re.DOTALL)
    if expenses_a:
        for name, amount, category, note in expenses_a:
            results.append({
                'name': name.strip(),
                'amount': amount,
                'category': category.strip(),
                'note': note.strip()
            })
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
        else:
            # 尝试匹配格式 c
            expenses_c = re.findall(pattern_c, instruction, re.DOTALL)
            if expenses_c:
                for name, amount, category, note in expenses_c:
                    results.append({
                        'name': name.strip(),
                        'amount': amount,
                        'category': category.strip(),
                        'note': note.strip()
                    })
    
    return results

def read_csv():
    # 读取 CSV 文件
    df = pd.read_csv('output.csv')  # 替换为你的文件
    return df


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
        env.reset(go_home=True)

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

        if task_value == 'SimpleCalendarAddOneEvent':
            year, month, day, hour, title, description, duration = extract_calendar_event_info(row['instruction'])
            if year is not None and month is not None and day is not None and hour is not None and title is not None and description is not None and duration is not None:
                print(f"year: {year}, month: {month}")

                start_ts = _create_unix_ts(day=day,hour=hour)
                end_ts = start_ts + duration * 60
                event = sqlite_schema_utils.CalendarEvent(
                    start_ts=start_ts,
                    end_ts=end_ts,
                    title=title,
                    description=description
                )
                n_noise_events = random.randint(0, 20)
                params = {
                    "year": year,
                    'month': month,
                    "day": day,
                    'hour': hour,
                    'duration_mins': duration,
                    'event_title': title,
                    'event_description': description,
                    'row_objects': [event],
                    'noise_row_objects': generate_noise_events(
                        [event], n_noise_events
                    ),
                }
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

        if task_value == 'SimpleCalendarAddOneEventInTwoWeeks':
            hour, title, description, duration = extract_event_info(row['instruction'])
            if hour is not None and title is not None and description is not None and duration is not None:
                print(f"hour: {hour}, title: {title}")

                start_ts = _create_unix_ts(day= device_constants.DT.day + 14,hour=hour)
                end_ts = start_ts + duration * 60
                event = sqlite_schema_utils.CalendarEvent(
                    start_ts=start_ts,
                    end_ts=end_ts,
                    title=title,
                    description=description
                )
                n_noise_events = random.randint(0, 20)
                params = {
                    "year": device_constants.DT.year,
                    'month': device_constants.DT.month,
                    "day": event.start_datetime.day,
                    'hour': hour,
                    'duration_mins': duration,
                    'event_title': title,
                    'event_description': description,
                    'row_objects': [event],
                    'noise_row_objects': generate_noise_events(
                        [event], n_noise_events
                    ),
                }

        if task_value == 'SimpleCalendarAddOneEventRelativeDay':
            week, hour, title, description, duration = extract_event_details(row['instruction'])
            if hour is not None and title is not None and description is not None and duration is not None and week is not None:
                print(f"hour: {hour}, title: {title}, week: {week}")

                date_num = weekday_to_number(week)
                start_ts = _create_unix_ts(day= device_constants.DT.day + date_num,hour=hour)
                end_ts = start_ts + duration * 60
                event = sqlite_schema_utils.CalendarEvent(
                    start_ts=start_ts,
                    end_ts=end_ts,
                    title=title,
                    description=description
                )
                n_noise_events = random.randint(0, 20)
                params = {
                    "year": device_constants.DT.year,
                    'month': device_constants.DT.month,
                    "day": event.start_datetime.day,
                    'hour': hour,
                    'duration_mins': duration,
                    'event_title': title,
                    'event_description': description,
                    'row_objects': [event],
                    'noise_row_objects': generate_noise_events(
                        [event], n_noise_events
                    ),
                }

        if task_value == 'SimpleCalendarAddRepeatingEvent':
            title, year, month, day, hour, recurrence, duration, description = extract_repeat_event_info(row['instruction'])
            if year is not None and month is not None and day is not None and hour is not None and title is not None and description is not None and duration is not None and recurrence is not None:
                print(f"year: {year}, month: {month}")

                start_ts = _create_unix_ts(day=day, hour=hour)
                end_ts = start_ts + duration * 60
                template = sqlite_schema_utils.CalendarEvent(
                    start_ts=start_ts,
                    end_ts=end_ts,
                    title=title,
                    description=description
                )
                if recurrence == "weekly":
                    repeat_rule = calendar_utils.generate_simple_calendar_weekly_repeat_rule(
                        template.start_datetime.isoweekday()
                    )
                else:
                    repeat_rule = 0

                event = dataclasses.replace(
                    template,
                    repeat_interval=_REPEAT_INTERVALS[recurrence],
                    repeat_rule=repeat_rule,
                )

                n_noise_events = random.randint(0, 20)
                params = {
                    "year": year,
                    'month': month,
                    "day": day,
                    'hour': hour,
                    'duration_mins': duration,
                    'event_title': title,
                    'event_description': description,
                    'row_objects': [event],
                    'noise_row_objects': generate_noise_events(
                        [event], n_noise_events
                    ),
                }


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

        # if task_value == 'VlcCreatePlaylist':
        #     playlist_name, files = extract_vlc_playlist_create(row['instruction'])
        #     if playlist_name is not None and files:
        #         print(f"Playlist Name: {playlist_name}, Files: {files}")
        #         params = {
        #             'playlist_name': playlist_name,
        #             'files': files,
        #             'noise_files': [generate_file_name() for _ in range(len(files))]
        #         }
        
        # if task_value == 'ExpenseAddMultiple' or task_value == 'ExpenseAddSingle':
        #     results = extract_expense_add_multiple(row['instruction'])
        #     if len(results) > 0:
        #         target_rows = []
        #         for result in results:
        #             if result['amount'] is not None and result['category'] is not None and result['note'] is not None:
        #                 expense_unix_time_s = _get_random_timestamp()
        #                 expense_unix_time_ms = expense_unix_time_s * 1000
        #                 print(f'result >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> {result}')
        #                 category_id = sqlite_schema_utils.Expense.category_name_to_id[result['category']]
        #                 target_rows.append(sqlite_schema_utils.Expense(
        #                     result['name'],
        #                     int(float(result['amount'])*100),
        #                     category_id,
        #                     result['note'],
        #                     expense_unix_time_ms,
        #                     expense_unix_time_ms,
        #                 ))
        #                 noise_rows = sqlite_schema_utils.get_random_items(
        #                     10,
        #                     _generate_expense,
        #                     replacement=False,
        #                     filter_fn=lambda r: all(r.name != t.name for t in target_rows),
        #                 )
        #         params = {
        #             sqlite_validators.ROW_OBJECTS: target_rows,
        #             sqlite_validators.NOISE_ROW_OBJECTS: noise_rows,
        #             'text_representation_type': random.choice(['csv', 'text_block']),
        #         }

        if params is None:
            continue

        task = task_type(params)
        task.initialize_task(env)
        openai_agent = t3a.T3A(env, infer.Gpt4Wrapper('gpt-4o-mini-2024-07-18'))
        openai4o_agent = t3a.T3A(env, infer.Gpt4Wrapper('gpt-4o-2024-11-20'))
        doubao_agent = Doubao(env, infer.DoubaoWrapper('doubao-1-5-ui-tars-250428'))
        agent = openai_agent if _AGENT_TYPE.value == 'openai' else doubao_agent
        if _AGENT_TYPE.value == 'openai4o':
            agent = openai4o_agent

        print('Goal: ' + str(task.goal))
        is_done = False
        for _ in range(min(int(task.complexity * 10), _MAX_STEP_COUNT.value)):
            response = agent.step(task.goal)
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
                if exist_retry_step:
                    export['trajectory_type'] = 1
                else:
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
    with open('cost_token', "w", encoding="utf-8") as f:
        json.dump({
            'prompt_tokens': sum(prompt_tokens),
            'prompt_tokens_avg': sum(prompt_tokens) / len(prompt_tokens),
            'completion_tokens': sum(completion_tokens),
            'completion_tokens_avg': sum(completion_tokens) / len(completion_tokens)
        }, f, ensure_ascii=False, indent=2)

    print('消耗prompt_tokens Token：', sum(prompt_tokens))
    print('消耗completion_tokens Token：', sum(completion_tokens))


def main(argv: Sequence[str]) -> None:
    del argv
    _main()


if __name__ == '__main__':
    app.run(main)
