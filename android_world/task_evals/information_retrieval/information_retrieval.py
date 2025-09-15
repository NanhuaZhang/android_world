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

"""Evaluators for information retrieval tasks."""

import abc
import random
from typing import Any
from android_world.env import interface
from android_world.task_evals import task_eval
from android_world.task_evals.information_retrieval import activity_app_utils
from android_world.task_evals.information_retrieval import calendar_utils as calendar_utils_ir
from android_world.task_evals.information_retrieval import datetime_utils as datetime_utils_ir
from android_world.task_evals.information_retrieval import joplin_app_utils
from android_world.task_evals.information_retrieval import proto_utils
from android_world.task_evals.information_retrieval import task_app_utils
from android_world.task_evals.information_retrieval.proto import state_pb2, task_pb2
from android_world.task_evals.single.calendar import calendar_utils


class InformationRetrieval(task_eval.TaskEval, abc.ABC):
  """Task for information retrieval.

  Each information retrieval task is dynamically generated using the task
  parameters and success criteria are tailored to the specific requirements of
  the task. The class supports initializing tasks with app-specific states and
  handling conditional task logic based on the initial state's app context.
  """
  template = ''
  schema = {}
  app_names = ()
  complexity = 1.0  # Overridden in the registry.

  @property
  @abc.abstractmethod
  def task_template(self) -> task_pb2.Task:
    """The Task proto defining this Information Retrieval task."""

  @property
  def task(self) -> task_pb2.Task:
    return self._task

  def is_calendar_task(self) -> bool:
    return self.task.relevant_state.state.HasField('calendar')

  def is_tasks_task(self) -> bool:
    return self.task.relevant_state.state.HasField('tasks_app')

  def is_sports_task(self) -> bool:
    return self.task.relevant_state.state.HasField('sports_activity_app')

  def is_notes_task(self) -> bool:
    return self.task.relevant_state.state.HasField('notes_app')

  def __init__(self, params: dict[str, Any]):
    super().__init__(params)
    self._task = task_pb2.Task()
    self._task.CopyFrom(self.task_template)
    self.template = self.task.prompt
    self.complexity = self.task.complexity

    # Set app names based on task type
    if self.is_calendar_task():
      self.app_names = (self.task.relevant_state.state.calendar.app_name,)
    if self.is_tasks_task():
      self.app_names = ('tasks',)
    if self.is_sports_task():
      self.app_names = ('open tracks sports tracker',)
    if self.is_notes_task():
      self.app_names = ('joplin',)

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    proto_utils.initialize_proto(self.task, self.params)
    # _maybe_replace_date(self.params)

    # Initialize app-specific state
    relevant_state = self.task.relevant_state.state
    exclusions = list(self.task.relevant_state.exclusion_conditions)

    if (
        self.is_calendar_task()
        and relevant_state.calendar.app_name == 'simple calendar pro'
    ):
      calendar_utils_ir.setup_task_state(
          relevant_state.calendar, exclusions, env
      )
    if self.is_tasks_task():
      task_app_utils.setup_task_state(relevant_state.tasks_app, exclusions, env)
    if self.is_sports_task():
      categories =[
        "running",
        "cycling",
        "swimming",
        "hiking",
        "mountain biking",
        "kayaking",
        "skiing",
        "snow boarding",
        "skate boarding",
        "climbing",
        "inline skating",
        "sailing"
      ]
      default_sports_activities = []
      existing_categories = []
      # 获取 activities 中存在的 category
      for activity in relevant_state.sports_activity_app.sports_activities:
        # 假设每个活动都有一个 'category' 属性
        if hasattr(activity, 'category'):
            existing_categories.append(activity.category)

      # 过滤 categories，留下 activities 中没有的 category
      filtered_categories = [category for category in categories if category not in existing_categories]
      # 在第一项插入非目标分类数据，避免app在取消选中后面分类时会同时取消第一项分类的bug出现
      for i in [1,2]:
        default_sports_activities.append(state_pb2.SportsActivity(
          start_date=random.choice([
            "October 9 2023",
            "October 10 2023",
            "October 11 2023",
            "October 12 2023",
            "October 13 2023",
            "October 14 2023",
            "October 15 2023"
          ]),
          start_time=random.choice([
            "8:00am", "10:30am", "5:00pm", "6:30am", "9:45am", "2:15pm", "7:00am", "11:00am", "4:00pm", "1:30pm"
          ]),
          duration=random.choice([
            "15", "30", "45", "60", "90", "120", "40", "50", "75", "80", "20", "105", "135"
          ]),
          total_distance=random.choice([
            "100", "300", "500", "800", "1000", "1200", "1500", "2000", "2500", "3000"
          ]),
          name = random.choice([
              "More tired than usual today",
              "Need more strength and conditioning",
              "Slow day",
              "Laps around the lake",
              "Trying and failing to keep up with John",
              "Quick outing",
              "Recovery day",
              "Active Rest Day",
              "Skill work"
          ]),
          description=random.choice([
            "Shared laughs and made memories with friends.",
            "Enjoyed a fun outing with good company.",
            "Had a blast with my favorite people.",
            "Created lasting memories that I'll cherish.",
            "Experienced something unforgettable.",
            "Captured moments that will bring a smile to my face.",
            "Wandered off the beaten path.",
            "Ventured into uncharted territory.",
            "Stepped outside my comfort zone.",
            "Pushed my boundaries and tried something different.",
            "Tested my limits and grew as a person."
          ]),
          category=filtered_categories[0]
        ))
      activity_app_utils.setup_task_state(
          relevant_state.sports_activity_app,
          exclusions,
          env,
          default_sports_activities,
      )
    if self.is_notes_task():
      joplin_app_utils.setup_task_state(
          relevant_state.notes_app, exclusions, env
      )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if not env.interaction_cache:
      return 0.0
    try:
      answers_are_equal = proto_utils.check_agent_answer(
          env.interaction_cache, self.task
      )
      return 1.0 if answers_are_equal else 0.0
    except ValueError:
      return 0.0

  def tear_down(self, env: interface.AsyncEnv) -> None:
    if self.is_calendar_task():
      calendar_utils.clear_calendar_db(env)
    if self.is_tasks_task():
      task_app_utils.clear_task_db(env)
    if self.is_sports_task():
      activity_app_utils.clear_db(env)
    if self.is_notes_task():
      joplin_app_utils.clear_dbs(env)
    super().tear_down(env)


def _maybe_replace_date(params: dict[str, Any]) -> None:
  """Maybe replaces date parameters with a natural language equivalent."""
  for param_name, param_value in params.items():
    if param_name == 'seed':
      continue
    if not isinstance(param_value, str):
      continue
    try:
      if not param_value:
        continue
      params[param_name] = datetime_utils_ir.generate_reworded_date(param_value)
    except ValueError:
      pass  # Skip if there's no date parameter.
