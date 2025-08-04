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
import io
import json
from collections.abc import Sequence
import os
import random
import re
from typing import Type

from PIL import Image
from absl import app
from absl import flags
from absl import logging
from android_world import registry
from android_world.agents import infer, m3a_utils
from android_world.agents import t3a
from android_world.agents import doubao_agent
from android_world.agents.doubao_agent import Doubao
from android_world.env import env_launcher
from android_world.task_evals import task_eval
import xml.etree.ElementTree as ET
import subprocess

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
      os.path.expanduser('/Users/zhaojunjie/Android/platform-tools/adb')
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

_TASK = flags.DEFINE_string(
    'task',
    None,
    'A specific task to run.',
)

_EPISODE_ID = flags.DEFINE_string(
    'episode_id',
    None,
    'A episode id',
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
  if _TASK.value:
    if _TASK.value not in aw_registry:
      raise ValueError('Task {} not found in registry.'.format(_TASK.value))
    task_type: Type[task_eval.TaskEval] = aw_registry[_TASK.value]
  else:
    task_type: Type[task_eval.TaskEval] = random.choice(
        list(aw_registry.values())
    )
  params = task_type.generate_random_params()
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
  save_task_history(agent, _EPISODE_ID.value, task.goal,agent_successful)

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

def save_task_history(agent, task_id: str, task_goal:str,success:bool,output_dir: str = "./task_histories"):
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
        'os':'Android 13.0',
        "episode_id":_EPISODE_ID.value,
        "screen_resolution":screen_size,
        "instruction":task_goal,
        "trajectory":trajectory,
        "trajectory_type":2
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
        for key in ("before_screenshot",'before_screenshot_mark'):
            img_array = step.get(key)
            if img_array is not None:
                filename = f"{key}_{idx}.png"  # e.g. before_screenshot_1.png
                file_path = os.path.join(task_dir, filename)
                # 将 ndarray 转为 PIL 并保存
                Image.fromarray(img_array.astype("uint8")).save(file_path)
                if key == "before_screenshot_mark":
                    rec['observation_mark'] = f"./{filename}"
                else:
                    rec['observation'] = f"./{filename}"
        trajectory.append(rec)

        # 保存页面结构 XML
        for key in ("before_element_list",''):
            elements = step.get(key)
            if elements is not None:
                filename = f"{key}_{idx}.xml"
                file_path = os.path.join(task_dir, filename)
                ui_elements_to_xml(elements, file_path)
                rec['xml'] = f"./{filename}"

    out_path = os.path.join(task_dir, f"{task_id}_history.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)
    print(f"✅ Task history saved to {out_path}")
    with open('cost_token',"w", encoding="utf-8") as f:
        json.dump({
            'prompt_tokens':sum(prompt_tokens),
            'prompt_tokens_avg':sum(prompt_tokens)/len(prompt_tokens),
            'completion_tokens':sum(completion_tokens),
            'completion_tokens_avg':sum(completion_tokens)/len(completion_tokens)
        }, f, ensure_ascii=False, indent=2)

    print('消耗prompt_tokens Token：',sum(prompt_tokens))
    print('消耗completion_tokens Token：',sum(completion_tokens))

def main(argv: Sequence[str]) -> None:
  del argv
  _main()


if __name__ == '__main__':
  app.run(main)
