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

import time

"""T3A: Text-only Autonomous Agent for Android."""
from android_world.agents import agent_utils
from android_world.agents import base_agent
from android_world.agents import infer
from android_world.agents import m3a_utils
from android_world.env import adb_utils
from android_world.env import interface
from android_world.env import json_action
from android_world.env import representation_utils

PROMPT_PREFIX = (
    'You are an agent who can operate an Android phone on behalf of a user.'
    " Based on user's goal/request, you may\n"
    '- Answer back if the request/goal is a question (or a chat message), like'
    ' user asks "What is my schedule for today?".\n'
    '- Complete some tasks described in the requests/goals by performing'
    ' actions (step by step) on the phone.\n\n'
    'When given a user request, you will try to complete it step by step. At'
    ' each step, a list of descriptions for most UI elements on the'
    ' current screen will be given to you (each element can be specified by an'
    ' index), together with a history of what you have done in previous steps.'
    ' Based on these pieces of information and the goal, you must choose to'
    ' perform one of the action in the following list (action description'
    ' followed by the JSON format) by outputing the action in the correct JSON'
    ' format.\n'
    '- If you think the task has been completed, finish the task by using the'
    ' status action with complete as goal_status:'
    ' `{{"action_type": "status", "goal_status": "complete"}}`\n'
    '- If you think the task is not'
    " feasible (including cases like you don't have enough information or can"
    ' not perform some necessary actions), finish by using the `status` action'
    ' with infeasible as goal_status:'
    ' `{{"action_type": "status", "goal_status": "infeasible"}}`\n'
    "- Answer user's question:"
    ' `{{"action_type": "answer", "text": "<answer_text>"}}`\n'
    '- Click/tap on a UI element (specified by its index) on the screen:'
    ' `{{"action_type": "click", "index": <target_index>}}`.\n'
    '- Long press on a UI element (specified by its index) on the screen:'
    ' `{{"action_type": "long_press", "index": <target_index>}}`.\n'
    '- Type text into an editable text field (specified by its index), this'
    ' action contains clicking the text field, typing in the text and pressing'
    ' the enter, so no need to click on the target field to start:'
    ' `{{"action_type": "input_text", "text": <text_input>, "index":'
    ' <target_index>}}`\n'
    '- Press the Enter key: `{{"action_type": "keyboard_enter"}}`\n'
    '- Navigate to the home screen: `{{"action_type": "navigate_home"}}`\n'
    '- Navigate back: `{{"action_type": "navigate_back"}}`\n'
    '- Scroll the screen or a scrollable UI element in one of the four'
    ' directions, use the same numeric index as above if you want to scroll a'
    ' specific UI element, leave it empty when scroll the whole screen:'
    ' `{{"action_type": "scroll", "direction": <up, down, left, right>,'
    ' "index": <optional_target_index>}}`\n'
    '- Open an app (nothing will happen if the app is not installed):'
    ' `{{"action_type": "open_app", "app_name": <name>}}`\n'
    '- Wait for the screen to update: `{{"action_type": "wait"}}`\n'
)

GUIDANCE = (
    'Here are some useful guidelines you need to follow:\n'
    'General\n'
    '- Usually there will be multiple ways to complete a task, pick the'
    ' easiest one. Also when something does not work as expected (due'
    ' to various reasons), sometimes a simple retry can solve the problem,'
    " but if it doesn't (you can see that from the history), try to"
    ' switch to other solutions.\n'
    '- Sometimes you may need to navigate the phone to gather information'
    ' needed to complete the task, for example if user asks'
    ' "what is my schedule tomorrow", then you may want to open the calendar'
    ' app (using the `open_app` action), look up information there, answer'
    " user's question (using the `answer` action) and finish (using"
    ' the `status` action with complete as goal_status).\n'
    '- For requests that are questions (or chat messages), remember to use'
    ' the `answer` action to reply to user explicitly before finish!'
    ' Merely displaying the answer on the screen is NOT sufficient (unless'
    ' the goal is something like "show me ...").\n'
    '- If the desired state is already achieved (e.g., enabling Wi-Fi when'
    " it's already on), you can just complete the task.\n"
    'Action Related\n'
    '- Use the `open_app` action whenever you want to open an app'
    ' (nothing will happen if the app is not installed), do not use the'
    ' app drawer to open an app unless all other ways have failed.\n'
    '- Use the `input_text` action whenever you want to type'
    ' something (including password) instead of clicking characters on the'
    ' keyboard one by one. Sometimes there is some default text in the text'
    ' field you want to type in, remember to delete them before typing.\n'
    '- For `click`, `long_press` and `input_text`, the index parameter you'
    ' pick must be VISIBLE in the screenshot and also in the UI element'
    ' list given to you (some elements in the list may NOT be visible on'
    ' the screen so you can not interact with them).\n'
    '- Consider exploring the screen by using the `scroll`'
    ' action with different directions to reveal additional content.\n'
    '- The direction parameter for the `scroll` action can be confusing'
    " sometimes as it's opposite to swipe, for example, to view content at the"
    ' bottom, the `scroll` direction should be set to "down". It has been'
    ' observed that you have difficulties in choosing the correct direction, so'
    ' if one does not work, try the opposite as well.\n'
    'Text Related Operations\n'
    '- Normally to select some text on the screen: <i> Enter text selection'
    ' mode by long pressing the area where the text is, then some of the words'
    ' near the long press point will be selected (highlighted with two pointers'
    ' indicating the range) and usually a text selection bar will also appear'
    ' with options like `copy`, `paste`, `select all`, etc.'
    ' <ii> Select the exact text you need. Usually the text selected from the'
    ' previous step is NOT the one you want, you need to adjust the'
    ' range by dragging the two pointers. If you want to select all text in'
    ' the text field, simply click the `select all` button in the bar.\n'
    "- At this point, you don't have the ability to drag something around the"
    ' screen, so in general you can not select arbitrary text.\n'
    '- To delete some text: the most traditional way is to place the cursor'
    ' at the right place and use the backspace button in the keyboard to'
    ' delete the characters one by one (can long press the backspace to'
    ' accelerate if there are many to delete). Another approach is to first'
    ' select the text you want to delete, then click the backspace button'
    ' in the keyboard.\n'
    '- To copy some text: first select the exact text you want to copy, which'
    ' usually also brings up the text selection bar, then click the `copy`'
    ' button in bar.\n'
    '- To paste text into a text box, first long press the'
    ' text box, then usually the text selection bar will appear with a'
    ' `paste` button in it.\n'
    '- When typing into a text field, sometimes an auto-complete dropdown'
    ' list will appear. This usually indicating this is a enum field and you'
    ' should try to select the best match by clicking the corresponding one'
    ' in the list.\n'
)

ACTION_SELECTION_PROMPT_TEMPLATE = (
    PROMPT_PREFIX
    + '\nThe current user goal/request is: {goal}'
    + '\n\nHere is a history of what you have done so far:\n{history}'
    + '\n\nHere is a list of descriptions for some UI elements on the current'
    ' screen:\n{ui_elements_description}\n'
    + GUIDANCE
    + '{additional_guidelines}'
    + '\n\nNow output an action from the above list in the correct JSON format,'
    ' following the reason why you do that. Your answer should look like:\n'
    'Reason: ...\nAction: {{"action_type":...}}\n\n'
    'Your Answer:\n'
)

SUMMARIZATION_PROMPT_TEMPLATE = (
    PROMPT_PREFIX
    + '\nThe (overall) user goal/request is:{goal}\n'
    'Now I want you to summerize the latest step based on the action you'
    ' pick with the reason and descriptions for the before and after (the'
    ' action) screenshots.\n'
    'Here is the description for the before'
    ' screenshot:\n{before_elements}\n'
    'Here is the description for the after screenshot:\n{after_elements}\n'
    'This is the action you picked: {action}\n'
    'Based on the reason: {reason}\n\n'
    '\nBy comparing the descriptions for the two screenshots and the action'
    ' performed, give a brief summary of this step.'
    ' This summary will be added to action history and used in future action'
    ' selection, so try to include essential information you think that will'
    ' be most useful for future action selection like'
    ' what you intended to do, why, if it worked as expected, if not'
    ' what might be the reason (be critical, the action/reason might not be'
    ' correct), what should/should not be done next and so on. Some more'
    ' rules/tips you should follow:\n'
    '- Keep it short and in one line.\n'
    "- Some actions (like `answer`, `wait`) don't involve screen change,"
    ' you can just assume they work as expected.\n'
    '- Given this summary will be added into action history, it can be used as'
    ' memory to include information that needs to be remembered, or shared'
    ' between different apps.\n\n'
    'Summary of this step: '
)


def _generate_ui_elements_description_list_full(
    ui_elements: list[representation_utils.UIElement],
    screen_width_height_px: tuple[int, int],
) -> str:
  """Generate description for a list of UIElement using full information.

  Args:
    ui_elements: UI elements for the current screen.
    screen_width_height_px: Logical screen size.

  Returns:
    Information for each UIElement.
  """
  tree_info = ''
  for index, ui_element in enumerate(ui_elements):
    if m3a_utils.validate_ui_element(ui_element, screen_width_height_px):
      tree_info += f'UI element {index}: {str(ui_element)}\n'
  return tree_info


def get_target_element(
    converted_action: dict[str, str|None|bool|int],
    ui_elements: list[representation_utils.UIElement],
) -> tuple[representation_utils.UIElement, int]:
    """Get the target element from the list of UI elements."""
    target_element = None
    action_index = None
    if converted_action.element_text is not None:
        for index, element in enumerate(ui_elements):
            if element.text is not None and element.text.lower() == converted_action.element_text.lower() and (converted_action.element_resource_name is None or element.resource_name == converted_action.element_resource_name):
                target_element = element
                action_index = index
                break
            elif element.content_description is not None and converted_action.element_text.lower() in element.content_description.lower():
                target_element = element
                action_index = index
                break
    elif converted_action.element_resource_name is not None:
        for index, element in enumerate(ui_elements):
            if element.resource_name == converted_action.element_resource_name:
                target_element = element
                action_index = index
                break
    return target_element, action_index

class ChangeMarkorContent(base_agent.EnvironmentInteractingAgent):
  """Text only autonomous agent for Android."""

  def __init__(
      self,
      env: interface.AsyncEnv,
      llm: infer.LlmWrapper,
      name: str = 'T3A',
  ):
    """Initializes a RandomAgent.

    Args:
      env: The environment.
      llm: The text only LLM.
      name: The agent name.
    """
    super().__init__(env, name)
    self.llm = llm
    self.history = []
    self.additional_guidelines = None

  def reset(self, go_home_on_reset: bool = False):
    super().reset(go_home_on_reset)
    self.env.hide_automation_ui()
    self.history = []

  def set_task_guidelines(self, task_guidelines: list[str]) -> None:
    self.additional_guidelines = task_guidelines

  def steps(self, file_count: int, file_name: str, update_content: str, update_name: str):
      element_index = 14
      if file_count == 1:
        element_index = 12
      reason = [
          "To begin the task of updating and renaming the file in Markor, the first step is to open the Markor app. Since the current screen is the home screen, using the `open_app` action with app_name \"Markor\" is the appropriate next step.",
          f"To start updating and renaming the file \"{file_name}\", the first step is to open it. this element corresponds to this file, and clicking on it will allow access to its content for editing.",
          "To update the content, I need to enter text selection mode first. Long pressing on the editable text field will allow me to access text selection options, including \"Select all\", which is required to replace the entire content.",
          "To update the content as required, we first need to select all existing text in the editable field. this element is labeled \"Select all\" and is visible, so clicking it will select all content, preparing for replacing it with the new text.",
          f"Now that all text in the editable field is selected, we need to replace it with the new content \"{update_content}\". Using the `input_text` action will directly input the new text, replacing the selected content, which aligns with the task's requirement to update the file's content.",
          "After updating the content of the file, the next step as per the task instructions is to click the save icon to ensure the content changes are saved before proceeding to rename the file. this element corresponds to the \"Save\" action, which is visible and clickable, so clicking it is the appropriate action now.",
          f"To proceed with renaming the file, we need to return to the file list screen. Navigating back from the current note editing screen will take us to the list where we can locate \"{file_name}\" and perform the renaming action.",
          "To rename the file, we need to be in the file list screen. Currently, we're still in the edit screen, so we should navigate back again to reach the file list where the file can be long - pressed for renaming.",
          f"To rename \"{file_name}\", I need to initiate the renaming action. Long - pressing on the file is the standard way to bring up file - operation options like renaming in Markor. This aligns with the task's requirement to rename the file, so I will long - press \"Select all\".",
          f"To proceed with renaming the selected file \"{file_name}\", I need to click on the \"Rename\" option which is visible and clickable on the current screen. This action will allow me to edit the file name, aligning with the task's requirement to change the name to \"{update_name}\".",
          f"To rename the file, we need to first select all the existing text in the editable field to replace it with \"{update_name}\". Long - pressing on the editable field will bring up text selection options, allowing us to select all the current text for replacement, which aligns with the task's requirement for renaming.",
          f"To rename the file, we need to replace the existing text in the editable field with \"{update_name}\". Since we already initiated text selection via long press, using `input_text` will overwrite the selected text with the new name, which aligns with the task's requirement to select all and enter the new name correctly.",
          f"To finalize the renaming of the file, we need to confirm the action by clicking the \"OK\" button in the rename dialog. This will apply the new file name \"{update_name}\" and complete the renaming step, which is the last part of the task.",
          f"All required steps for updating the content of {file_name} to \"{update_content}\" and renaming it to {update_name} have been successfully executed, including selecting all text for content update, saving, selecting all text for renaming, inputting the new name, and confirming the rename. The current screen confirms the file name is updated, so the task is complete.",
      ]

      steps = [
          f"""Reason: {reason[0]}
            Action: {{"action_type": "open_app", "app_name": "Markor"}}""",
          f"""Reason: {reason[1]}
            Action: {{"action_type": "click", "index": {element_index}}}""",
          f"""Reason: {reason[2]}
            Action: {{"action_type": "long_press", "index": 8}}""",
          f"""Reason: {reason[3]}
            Action: {{"action_type": "click", "index": 19}}""",
          f"""Reason: {reason[4]}
            Action: {{"action_type": "input_text", "text": "{update_content}", "index": 9}}""",
          f"""Reason: {reason[5]}
            Action: {{"action_type": "click", "index": 6}}""",
          f"""Reason: {reason[6]}
            Action: {{"action_type": "navigate_back" }}""",
          f"""Reason: {reason[7]}
            Action: {{"action_type": "navigate_back" }}""",
          f"""Reason: {reason[8]}
            Action: {{"action_type": "long_press", "index": {element_index}}}""",
          f"""Reason: {reason[9]}
            Action: {{"action_type": "click", "index": 5 }}""",
          f"""Reason: {reason[10]}
            Action: {{"action_type": "long_press", "index": 2 }}""",
          f"""Reason: {reason[11]}
            Action: {{"action_type": "input_text", "text": "{update_name}", "index": 2}}""",
          f"""Reason: {reason[12]}
            Action: {{"action_type": "click", "index": 4}}""",
          f"""Reason: {reason[13]}
            Action: {{"action_type": "status", "goal_status": "complete"}}""",
      ]

      for index,step in enumerate(steps):
        self.step(step)


  def step(self, action_output:str) -> base_agent.AgentInteractionResult:
    step_data = {
        'before_screenshot': None,
        'after_screenshot': None,
        'before_element_list': None,
        'after_element_list': None,
        'action_prompt': None,
        'action_output': None,
        'action_raw_response': None,
        'summary_prompt': None,
        'summary': None,
        'summary_raw_response': None,
    }
    print('----------step ' + str(len(self.history) + 1))
    time.sleep(1.0)
    state = self.get_post_transition_state()
    time.sleep(1.0)
    logical_screen_size = self.env.logical_screen_size

    ui_elements = state.ui_elements
    before_element_list = _generate_ui_elements_description_list_full(
        ui_elements,
        logical_screen_size,
    )
    # Only save the screenshot for result visualization.
    step_data['before_screenshot'] = state.pixels.copy()
    step_data['before_screenshot_mark'] = state.pixels.copy()
    step_data['before_element_list'] = ui_elements

    step_data['action_output'] = action_output
    time.sleep(1.0)
    reason, action = m3a_utils.parse_reason_action_output(action_output)
    time.sleep(1.0)

    # If the output is not in the right format, add it to step summary which
    # will be passed to next step and return.
    print('Action: ' + action)
    print('Reason: ' + reason)

    try:
      converted_action = json_action.JSONAction(
          **agent_utils.extract_json(action),
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      print('Failed to convert the output to a valid action.')
      print(str(e))
      step_data['summary'] = (
          'Can not parse the output to a valid action. Please make sure to pick'
          ' the action from the list with the correct json format!'
      )
      self.history.append(step_data)

      return base_agent.AgentInteractionResult(
          False,
          step_data,
      )

    step_data['direction'] = converted_action.direction
    step_data['keycode'] = converted_action.keycode
    step_data['content'] = converted_action.text
    step_data['action'] = converted_action.action_type

    if converted_action.action_type in ['click', 'long_press', 'input_text']:
      if converted_action.index is not None and converted_action.index >= len(
          ui_elements
      ):
        print('Index out of range.')
        step_data['summary'] = (
            'The parameter index is out of range. Remember the index must be in'
            ' the UI element list!'
        )
        self.history.append(step_data)
        return base_agent.AgentInteractionResult(False, step_data)
      else:
        target_element = ui_elements[converted_action.index]
        center_x, center_y = target_element.bbox_pixels.center  # (x1, y1, x2, y2)
        click_point = (center_x, center_y)

        # 记录点击前坐标
        if converted_action.action_type == 'input_text':
          click_point = None

        step_data['start_coords'] = click_point
        step_data['end_coords'] = click_point

        # Add mark for the target ui element, just used for visualization.ƒ
        m3a_utils.add_ui_element_mark(
            step_data['before_screenshot_mark'],
            ui_elements[converted_action.index],
            converted_action.index,
            logical_screen_size,
            adb_utils.get_physical_frame_boundary(self.env.controller),
            adb_utils.get_orientation(self.env.controller),
            click_point,
        )

    if converted_action.action_type == 'open_app':
        step_data['app_name'] = converted_action.app_name

    if converted_action.action_type == 'status':
      if converted_action.goal_status == 'infeasible':
        print('Agent stopped since it thinks mission impossible.')
      step_data['summary'] = 'Agent thinks the request has been completed.'
      self.history.append(step_data)
      step_data['status'] = converted_action.goal_status
      return base_agent.AgentInteractionResult(
          True,
          step_data,
      )

    if converted_action.action_type == 'answer':
      print('Agent answered with: ' + converted_action.text)

    try:
      result = self.env.execute_action(converted_action)
      if converted_action.action_type == 'scroll':
          step_data['start_coords'] = (result[0],result[1])
          step_data['end_coords'] = (result[2],result[3])

      if converted_action.action_type == 'input_text':
        if step_data is not None and result is not None and result['input_text_rename'] != True:
          click_step = result
          click_step['action_output'] = step_data['action_output']
          before_screenshot = step_data['before_screenshot']
          step_data['before_screenshot'] = click_step['before_screenshot']

          click_step['before_screenshot'] = before_screenshot
          click_step['before_screenshot_mark'] = step_data['before_screenshot_mark']

          step_data['before_screenshot_mark'] = step_data['before_screenshot']
          step_data['action_output'] = f'Reason: Input {converted_action.text}\n Action:xxx'
          self.history.append(click_step)

    except Exception as e:  # pylint: disable=broad-exception-caught
      print(
          'Some error happened executing the action ',
          converted_action.action_type,
      )
      print(str(e))
      step_data['summary'] = (
          'Some error happened executing the action '
          + converted_action.action_type
      )
      self.history.append(step_data)

      return base_agent.AgentInteractionResult(
          False,
          step_data,
      )

    time.sleep(2.0)
    state = self.get_post_transition_state()
    ui_elements = state.ui_elements

    # Save screenshot only for result visualization.
    step_data['after_screenshot'] = state.pixels.copy()
    step_data['after_element_list'] = ui_elements
    time.sleep(2.0)

    self.history.append(step_data)

    return base_agent.AgentInteractionResult(
        False,
        step_data,
    )


class MarkorAddNoteHeader(base_agent.EnvironmentInteractingAgent):
  """Text only autonomous agent for Android."""

  def __init__(
      self,
      env: interface.AsyncEnv,
      llm: infer.LlmWrapper,
      name: str = 'T3A',
  ):
    """Initializes a RandomAgent.

    Args:
      env: The environment.
      llm: The text only LLM.
      name: The agent name.
    """
    super().__init__(env, name)
    self.llm = llm
    self.history = []
    self.additional_guidelines = None

  def reset(self, go_home_on_reset: bool = False):
    super().reset(go_home_on_reset)
    self.env.hide_automation_ui()
    self.history = []

  def set_task_guidelines(self, task_guidelines: list[str]) -> None:
    self.additional_guidelines = task_guidelines

  def steps(self, file_name: str, update_content: str, update_name: str):
      element_index = 14
      update_str = update_content + "\\n\\n"
      reason = [
          "To start updating the Markor note, I need to open the Markor app. Since the current screen is the home screen, using the `open_app` action with app_name \"Markor\" is the first step to access the app where the note is located.",
          f"To begin updating the specified note, I need to open the file \"{file_name}\". The element corresponds to this file, and clicking on it will allow me to access its content for editing.",
          "To begin editing the note's content, I need to activate the editable text field by clicking on it. This will allow me to place the cursor at the start and input the required new text along with a blank line before the existing content.",
          "To add content to the beginning of the text, I need to click on the note to focus on the note header.",
          f"To update the note as required, we need to insert the specified text \"{update_str}\" followed by a new blank line (a newline) before the existing content. The editable text field (index 9) is active, so we can use the `input_text` action to type the required text at the current cursor position, which should be at the start of the field after clicking it earlier. This action will add the new text and the blank line as needed.",
          "To ensure the content changes (adding the required text) are saved before proceeding to rename the file, I need to click the \"Save\" button (this element) on the current screen. This aligns with the task's instruction to click the save icon before returning.",
          f"To rename the file, we need to return to the file list screen where the file is located. Navigating back from the current edit screen will take us to the main Markor screen with the list of files, allowing us to access the rename option for `{file_name}`.",
          "To rename the file, we need to be on the file list screen where the file is listed. Currently, we're still in the note edit screen, so we should navigate back to the file list screen.",
          f"To rename the file {file_name}, I need to initiate the renaming process. Long-pressing on the file (this element) is the standard way to bring up file operation options (like rename) in Markor, so this action will allow me to proceed with renaming.",
          f"To proceed with renaming the selected file {file_name}, we need to use the \"Rename\" option from the toolbar. this element corresponds to the \"Rename\" button, which is visible and clickable, so clicking it will initiate the renaming process.",
          "To rename the file correctly, I need to first select all the existing text in the editable field (this element) so that entering the new name replaces it entirely. Long - pressing on the editable field will bring up text selection options, allowing me to select all the current text.",
          f"Now that the text in the filename editable field is selected (after long - press), we need to input the new name \"{update_name}\" to replace the selected text. Using the `input_text` action on index 2 with the new name will achieve this, as it will overwrite the selected content with the entered text, which is the correct step for renaming as per the task requirements.",
          "To finalize the renaming of the file, we need to confirm the action by clicking the \"OK\" button (this element) in the rename dialog. This will complete the renaming step as required.",
          f"All required steps for updating the note's content (adding the specified text with a new blank line) and renaming the file to {update_name} have been successfully executed, as evidenced by the updated file name in the Markor app's file list. No further actions are needed, so the task can be marked as complete.",
      ]

      steps = [
          f"""Reason: {reason[0]}
            Action: {{"action_type": "open_app", "app_name": "Markor"}}""",
          f"""Reason: {reason[1]}
            Action: {{"action_type": "click", "index": {element_index}, "element_text": "{file_name}" }}""",
          f"""Reason: {reason[2]}
            Action: {{"action_type": "click", "index": 18, "click_x": 20, "click_y": 310, "element_resource_name": "net.gsantner.markor:id/document__fragment__edit__highlighting_editor"}}""",
          f"""Reason: {reason[4]}
            Action: {{"action_type": "input_text", "text": "{update_content}", "index": 9, "element_resource_name": "net.gsantner.markor:id/document__fragment__edit__highlighting_editor"}}""",
          f"""Reason: {reason[5]}
            Action: {{"action_type": "click", "index": 6}}""",
          f"""Reason: {reason[6]}
            Action: {{"action_type": "navigate_back" }}""",
          f"""Reason: {reason[7]}
            Action: {{"action_type": "navigate_back" }}""",
          f"""Reason: {reason[8]}
            Action: {{"action_type": "long_press", "index": {element_index}, "element_text": "{file_name}"}}""",
          f"""Reason: {reason[9]}
            Action: {{"action_type": "click", "index": 5, "element_text": "Save" }}""",
          f"""Reason: {reason[10]}
            Action: {{"action_type": "long_press", "index": 2 }}""",
          f"""Reason: {reason[11]}
            Action: {{"action_type": "input_text", "text": "{update_name}", "index": 2}}""",
          f"""Reason: {reason[12]}
            Action: {{"action_type": "click", "index": 4}}""",
          f"""Reason: {reason[13]}
            Action: {{"action_type": "status", "goal_status": "complete"}}""",
      ]

      for index,step in enumerate(steps):
        self.step(step)


  def step(self, action_output:str) -> base_agent.AgentInteractionResult:
    step_data = {
        'before_screenshot': None,
        'after_screenshot': None,
        'before_element_list': None,
        'after_element_list': None,
        'action_prompt': None,
        'action_output': None,
        'action_raw_response': None,
        'summary_prompt': None,
        'summary': None,
        'summary_raw_response': None,
    }
    print('----------step ' + str(len(self.history) + 1))
    time.sleep(1.0)
    state = self.get_post_transition_state()
    time.sleep(1.0)
    logical_screen_size = self.env.logical_screen_size

    ui_elements = state.ui_elements
    before_element_list = _generate_ui_elements_description_list_full(
        ui_elements,
        logical_screen_size,
    )
    # Only save the screenshot for result visualization.
    step_data['before_screenshot'] = state.pixels.copy()
    step_data['before_screenshot_mark'] = state.pixels.copy()
    step_data['before_element_list'] = ui_elements

    step_data['action_output'] = action_output
    time.sleep(1.0)
    reason, action = m3a_utils.parse_reason_action_output(action_output)
    time.sleep(1.0)

    # If the output is not in the right format, add it to step summary which
    # will be passed to next step and return.
    print('Action: ' + action)
    print('Reason: ' + reason)

    try:
      converted_action = json_action.JSONAction(
          **agent_utils.extract_json(action),
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f"action>>>>>>>>>>>>>>> {action}")
      print('Failed to convert the output to a valid action.')
      print(str(e))
      step_data['summary'] = (
          'Can not parse the output to a valid action. Please make sure to pick'
          ' the action from the list with the correct json format!'
      )
      self.history.append(step_data)

      return base_agent.AgentInteractionResult(
          False,
          step_data,
      )

    step_data['direction'] = converted_action.direction
    step_data['keycode'] = converted_action.keycode
    step_data['content'] = converted_action.text
    step_data['action'] = converted_action.action_type

    if converted_action.action_type in ['click', 'long_press', 'input_text']:
      if converted_action.index is not None and converted_action.index >= len(
          ui_elements
      ):
        print('Index out of range.')
        step_data['summary'] = (
            'The parameter index is out of range. Remember the index must be in'
            ' the UI element list!'
        )
        self.history.append(step_data)
        return base_agent.AgentInteractionResult(False, step_data)
      else:
        target_element = ui_elements[converted_action.index]
        if converted_action.element_text is not None:
          element_by_text = None
          action_index = converted_action.index
          for index, element in enumerate(ui_elements):
            if element.text == converted_action.element_text:
                element_by_text = element
                action_index = index
                break
            elif element.content_description is not None and converted_action.element_text in element.content_description:
                element_by_text = element
                action_index = index
                break
          if element_by_text is not None:
            target_element = element_by_text
            converted_action.index = action_index
            
        elif converted_action.element_resource_name is not None:
          element_by_resource_name = None
          action_index = converted_action.index
          for index, element in enumerate(ui_elements):
            if element.resource_name == converted_action.element_resource_name:
                element_by_resource_name = element
                action_index = index
                break
          if element_by_resource_name is not None:
            target_element = element_by_resource_name
            converted_action.index = action_index
        
        center_x, center_y = target_element.bbox_pixels.center  # (x1, y1, x2, y2)
        click_point = (center_x, center_y)

        if converted_action.click_x is not None and converted_action.click_y is not None:
          click_point = (converted_action.click_x, converted_action.click_y)

        # 记录点击前坐标
        if converted_action.action_type == 'input_text':
          click_point = None

        step_data['start_coords'] = click_point
        step_data['end_coords'] = click_point

        # Add mark for the target ui element, just used for visualization.ƒ
        m3a_utils.add_ui_element_mark(
            step_data['before_screenshot_mark'],
            ui_elements[converted_action.index],
            converted_action.index,
            logical_screen_size,
            adb_utils.get_physical_frame_boundary(self.env.controller),
            adb_utils.get_orientation(self.env.controller),
            click_point,
        )

    if converted_action.action_type == 'open_app':
        step_data['app_name'] = converted_action.app_name

    if converted_action.action_type == 'status':
      if converted_action.goal_status == 'infeasible':
        print('Agent stopped since it thinks mission impossible.')
      step_data['summary'] = 'Agent thinks the request has been completed.'
      self.history.append(step_data)
      step_data['status'] = converted_action.goal_status
      return base_agent.AgentInteractionResult(
          True,
          step_data,
      )

    if converted_action.action_type == 'answer':
      print('Agent answered with: ' + converted_action.text)

    try:
      result = self.env.execute_action(converted_action)
      if converted_action.action_type == 'scroll':
          step_data['start_coords'] = (result[0],result[1])
          step_data['end_coords'] = (result[2],result[3])

      if converted_action.action_type == 'input_text':
        if step_data is not None and result is not None and not result['input_text_rename']:
          click_step = result
          click_step['action_output'] = step_data['action_output']
          before_screenshot = step_data['before_screenshot']
          step_data['before_screenshot'] = click_step['before_screenshot']

          click_step['before_screenshot'] = before_screenshot
          click_step['before_screenshot_mark'] = step_data['before_screenshot_mark']

          step_data['before_screenshot_mark'] = step_data['before_screenshot']
          step_data['action_output'] = f'Reason: Input {converted_action.text}\n Action:xxx'
          self.history.append(click_step)

    except Exception as e:  # pylint: disable=broad-exception-caught
      print(
          'Some error happened executing the action ',
          converted_action.action_type,
      )
      print(str(e))
      step_data['summary'] = (
          'Some error happened executing the action '
          + converted_action.action_type
      )
      self.history.append(step_data)

      return base_agent.AgentInteractionResult(
          False,
          step_data,
      )

    time.sleep(2.0)
    state = self.get_post_transition_state()
    ui_elements = state.ui_elements

    # Save screenshot only for result visualization.
    step_data['after_screenshot'] = state.pixels.copy()
    step_data['after_element_list'] = ui_elements
    time.sleep(2.0)

    self.history.append(step_data)

    return base_agent.AgentInteractionResult(
        False,
        step_data,
    )


class MarkorMergeNotes(base_agent.EnvironmentInteractingAgent):
  """Text only autonomous agent for Android."""

  def __init__(
      self,
      env: interface.AsyncEnv,
      llm: infer.LlmWrapper,
      name: str = 'T3A',
  ):
    """Initializes a RandomAgent.

    Args:
      env: The environment.
      llm: The text only LLM.
      name: The agent name.
    """
    super().__init__(env, name)
    self.llm = llm
    self.history = []
    self.additional_guidelines = None

  def reset(self, go_home_on_reset: bool = False):
    super().reset(go_home_on_reset)
    self.env.hide_automation_ui()
    self.history = []

  def set_task_guidelines(self, task_guidelines: list[str]) -> None:
    self.additional_guidelines = task_guidelines

  def steps(
    self,
    file_name1: str,
    file_name2: str,
    file_name3: str,
    new_file_name: str,
  ):
      # if last_merge:
      #   update_str = update_content
      # else:
      #   update_str = update_content + "\\n\\n"
      #   update_content = update_content + "\n\n"
      
      reason = [
          "To start updating the Markor note, I need to open the Markor app. Since the current screen is the home screen, using the `open_app` action with app_name \"Markor\" is the first step to access the app where the note is located.",
          "To create a new empty note, the first step is to initiate the creation process. The UI element with index 1, described as \"Create a new file or folder\", is the appropriate element to click to start creating a new note in Markor.",
          "Click the \"Title\" input box to activate the editing state",
          f"To create a new note named \"{new_file_name}\", we need to input the name into the \"Title\" editable text field. this element is the correct target for this input, so using the `input_text` action here will set the note's name appropriately.",
          f"To finalize the creation of the new note named \"{new_file_name}\", we need to confirm the action by clicking the \"OK\" button, which is this element. This will save the new note and allow us to proceed with the next steps.",
          f"After creating the note \"{new_file_name}\", the next step as per the task is to return to the notes list screen. Using the `navigate_back` action will take us back to the previous screen, which should be the notes list screen in Markor.",
          f"Currently, we are still in the edit screen of \"{new_file_name}\". we need to copy the content of \"{file_name1}\", so we need to navigate back to the notes list screen. Using the `navigate_back` action will take us back to the previous screen, which should be the notes list screen in Markor.",
          f"To proceed with copying the content of \"{file_name1}\", I need to open this file. this element corresponds to this file, so clicking on it will allow me to access its content for further operations.",
          f"To copy the content of \"{file_name1}\", I need to enter text selection mode. Long pressing on the editable text field  will allow me to access text selection options like \"Select all\" and \"Copy\". This is the first step to copy the content.",
          f"To copy the content of \"{file_name1}\", after long - pressing the text field, the next step is to select all text. this element is \"Select all\" and is clickable, so clicking it will select all the text in the field, preparing for copying.",
          f"After selecting all text in \"{file_name1}\", the next step is to copy the selected content. this element corresponds to the \"Copy\" option, which is clickable and visible, so clicking it will copy the text to the clipboard for later pasting.",
          f"Need to return to the notes list screen to access the newly created note \"{new_file_name}\" for pasting the copied content. Navigating back from the current edit screen of \"{file_name1}\" will achieve this.",
          f"To proceed with pasting the copied content into \"{new_file_name}\", we first need to return to the notes list screen. Since the current screen is still the edit screen for \"{file_name1}\", navigating back again will take us to the notes list, allowing us to locate and open \"{new_file_name}\".",
          f"To proceed with pasting the copied content into \"{new_file_name}\", I need to open the note for editing. this element corresponds to \"{new_file_name}.md\", which is clickable and visible, so clicking it will navigate to its edit screen.",
          f"To proceed with pasting the copied content into \"{new_file_name}\", I need to switch the note to edit mode. Now click on the note to trigger the edit.",
          f"To paste the copied content into the \"{new_file_name}\" note, we need to long - press on the editable text field  to bring up the text selection menu with the \"Paste\" option. This is the appropriate action to insert the copied text into the note.",
          f"To paste the copied content of \"{file_name1}\" into the \"{new_file_name}\" note, we need to click on the \"Paste\" UI element which is visible and clickable on the current screen, as it appears in the text selection menu after long - pressing the editable text field.",
          f"After pasting the content from \"{file_name1}\" into \"{new_file_name}\", the next step as per the task is to use 'keyboard_enter' to add a new line. This aligns with the requirement, so performing the keyboard_enter action is necessary now.",
          f"After pasting the content and adding a new line, the next step as per the task is to save the note \"{new_file_name}\". The \"Save\" button is visible and clickable on the current screen, so clicking it will save the changes made to the note.",
          f"After saving the ndblCnTF note with the pasted content and new line, the next step as per the task is to return to the notes list screen to prepare for accessing \"{file_name2}\". Using the `navigate_back` action will take us back to the notes list screen, which aligns with the task's requirement to return to the list page after each save.",
          f"Currently, we are still in the edit screen of \"{new_file_name}\". we need to copy the content of \"{file_name2}\", so we need to navigate back to the notes list screen. Using the `navigate_back` action will take us back to the previous screen, which should be the notes list screen in Markor.",
          f"To proceed with the task of copying the content of \"{file_name2}\", I need to open this file. this element corresponds to \"{file_name2}\" and is visible on the current screen, so clicking on it will navigate to its edit screen, allowing further actions like selecting and copying its content.",
          f"To copy the content of \"{file_name2}\", we need to enter text selection mode. Long - pressing on the editable text field  will bring up the text selection menu, which is the first step for selecting and copying the text, similar to the process used for \"{file_name1}\".",
          f"To copy the content of \"{file_name2}\", after long - pressing the text field  to bring up text selection options, the next logical step is to click \"Select all\"  to select all text in the field, preparing for copying.",
          f"To copy the content of \"{file_name2}\", we need to click the \"Copy\" UI element  which is visible and accessible on the current screen, as we have already selected all text in the previous step. This action will copy the selected content to the clipboard, preparing it for pasting into \"{new_file_name}\".",
          f"After copying the content of \"{file_name2}\", we need to return to the notes list screen to access \"{new_file_name}\" for pasting. Navigating back is the appropriate action to achieve this, as it will take us from the current note edit screen to the notes list screen.",
          f"Currently, we are still in the edit screen of \"{file_name2}\" (evidenced by UI elements like the note title and edit - related buttons). To proceed with pasting its content into \"{new_file_name}\", we first need to return to the notes list screen. Using the `navigate_back` action will take us back to the notes list, which is necessary to locate and open \"{new_file_name}\" for the next steps.",
          f"To proceed with pasting the copied content of \"{file_name2}\" into \"{new_file_name}\", we need to open the \"{new_file_name}\" note for editing. this element corresponds to \"{new_file_name}.md\", which is visible on the current screen, so clicking it will navigate to its edit screen.",
          f"To proceed with pasting the copied content into \"{new_file_name}\", I need to switch the note to edit mode. Now click on the note to trigger the edit.",
          f"To paste the copied content of {file_name2} into {new_file_name}, we first need to bring up the text selection menu by long - pressing on the editable text field , which is the standard way to access paste option in Markor's text editor.",
          f"To paste the copied content of {file_name2} into {new_file_name}, we need to click the \"Paste\" UI element which is visible and accessible in the current text selection menu. This action aligns with the task of pasting the content into the note.",
          f"After pasting the content from {file_name2} into {new_file_name}, the next step as per the task is to use 'keyboard_enter' to add a new line. This aligns with the task's requirement for this step, so we need to perform the keyboard_enter action.",
          f"After pasting the content of {file_name2} and adding a new line in {new_file_name}, the next step as per the task instructions is to save the note. The \"Save\" button is visible and clickable on the current screen, so clicking it will save the changes made to {new_file_name}.",
          f"After saving the changes to {new_file_name}, the task requires returning to the notes list screen. The current screen is the note editing screen, so using the `navigate_back` action will navigate back to the notes list screen, which aligns with the task's requirement to return to the list page after each save.",
          f"Currently, we are still in the edit screen of \"{new_file_name}\". we need to copy the content of \"{file_name3}\", so we need to navigate back to the notes list screen. Using the `navigate_back` action will take us back to the previous screen, which should be the notes list screen in Markor.",
          f"To proceed with the task of copying content from {file_name3}, the next step is to open this file. this element corresponds to \"{file_name3}\", and clicking it will navigate to its edit screen, allowing access to its content for copying.",
          f"To copy the content of \"{file_name3}\", first enter text selection mode by long - pressing on the editable text field, which will bring up the text selection menu for further actions like selecting all and copying.",
          f"To copy the content of \"{file_name3}\", I need to select all text in its editable field. this element is \"Select all\", which is visible and clickable, so clicking it will select all text, preparing for the copy action.",
          f"To copy the content of \"{file_name3}\" which has all text selected, we need to click the \"Copy\" UI element to store the content in the clipboard for later pasting.",
          f"After copying the content of \"{file_name3}\", we need to return to the notes list screen to proceed with pasting the content into \"{new_file_name}\". Using the `navigate_back` action will take us back to the previous screen (notes list screen), which is necessary for the next steps.",
          f"Currently, we are still in the edit screen of \"{file_name3}\" (as indicated by this element's text). To proceed with pasting its content into \"{new_file_name}\", we need to return to the notes list screen. Using `navigate_back` will take us back to the previous screen (notes list), which is necessary for the next steps.",
          f"To proceed with pasting the content of \"{file_name3}\" into \"{new_file_name}\", we need to open the {new_file_name}.md note for editing. this element corresponds to \"{new_file_name}.md\", which is visible on the current screen, so clicking it will navigate to its edit screen.",
          f"To proceed with pasting the copied content into \"{new_file_name}\", I need to switch the note to edit mode. Now click on the note to trigger the edit.",
          f"To paste the copied content of \"{file_name3}\" into \"{new_file_name}\", we need to bring up the text selection menu by long - pressing on the editable text field , which is visible on the current screen. This will allow us to access the \"Paste\" option for inserting the copied content.",
          f"To paste the copied content of \"{file_name3}\" into \"{new_file_name}\", we need to click the \"Paste\" option (this element) which is visible and clickable on the current screen, as the text selection menu is already open from the long - press action.",
          f"After pasting the content of \"{file_name3}\" into \"{new_file_name}\", the next step as per the task is to save the note. The \"Save\" button is identified by index 5, which is visible and clickable on the current screen. Clicking this button will save the changes made to the note.",
          "All the steps specified in the user's goal have been successfully executed, including creating the new note, copying and pasting content from the three specified files with the required new lines, saving after each operation, and returning to the notes list screen. There are no remaining tasks, so the goal can be marked as complete.",
      ]

      steps = [
          f"""Reason: {reason[0]}
            Action: {{"action_type": "open_app", "app_name": "Markor"}}""",
          f"""Reason: {reason[1]}
            Action: {{"action_type": "click", "index": 10, "element_text": "Create a new file or folder" }}""",
          f"""Reason: {reason[2]}
            Action: {{"action_type": "click", "index": 10, "element_text": "Title" }}""",
          f"""Reason: {reason[3]}
            Action: {{"action_type": "input_text", "text": "{new_file_name}", "index": 10, "element_text": "Title" }}""",
          f"""Reason: {reason[4]}
            Action: {{"action_type": "click", "index": 10, "element_text": "OK" }}""",
          f"""Reason: {reason[5]}
            Action: {{"action_type": "navigate_back" }}""",
          f"""Reason: {reason[6]}
            Action: {{"action_type": "navigate_back" }}""",
          f"""Reason: {reason[7]}
            Action: {{"action_type": "click", "index": 10, "element_text": "{file_name1}" }}""",
          f"""Reason: {reason[8]}
            Action: {{"action_type": "long_press", "index": 10, "element_resource_name": "net.gsantner.markor:id/document__fragment__edit__highlighting_editor" }}""",
          f"""Reason: {reason[9]}
            Action: {{"action_type": "click", "index": 10, "element_text": "Select all" }}""",
          f"""Reason: {reason[10]}
            Action: {{"action_type": "click", "index": 10, "element_text": "Copy" }}""",
          f"""Reason: {reason[11]}
            Action: {{"action_type": "navigate_back" }}""",
          f"""Reason: {reason[12]}
            Action: {{"action_type": "navigate_back" }}""",
          f"""Reason: {reason[13]}
            Action: {{"action_type": "click", "index": 10, "element_text": "{new_file_name}" }}""",
          f"""Reason: {reason[14]}
            Action: {{"action_type": "click", "index": 10, "element_resource_name": "net.gsantner.markor:id/document__fragment__edit__highlighting_editor" }}""",
          f"""Reason: {reason[15]}
            Action: {{"action_type": "long_press", "index": 10, "element_resource_name": "net.gsantner.markor:id/document__fragment__edit__highlighting_editor" }}""",
          f"""Reason: {reason[16]}
            Action: {{"action_type": "click", "index": 10, "element_text": "Paste" }}""",
          f"""Reason: {reason[17]}
            Action: {{"action_type": "keyboard_enter" }}""",
          f"""Reason: {reason[18]}
            Action: {{"action_type": "click", "index": 10, "element_text": "Save" }}""",
          f"""Reason: {reason[19]}
            Action: {{"action_type": "navigate_back" }}""",
          f"""Reason: {reason[20]}
            Action: {{"action_type": "navigate_back" }}""",
          f"""Reason: {reason[21]}
            Action: {{"action_type": "click", "index": 10, "element_text": "{file_name2}" }}""",
          f"""Reason: {reason[22]}
            Action: {{"action_type": "long_press", "index": 10, "element_resource_name": "net.gsantner.markor:id/document__fragment__edit__highlighting_editor" }}""",
          f"""Reason: {reason[23]}
            Action: {{"action_type": "click", "index": 10, "element_text": "Select all" }}""",
          f"""Reason: {reason[24]}
            Action: {{"action_type": "click", "index": 10, "element_text": "Copy" }}""",
          f"""Reason: {reason[25]}
            Action: {{"action_type": "navigate_back" }}""",
          f"""Reason: {reason[26]}
            Action: {{"action_type": "navigate_back" }}""",
          f"""Reason: {reason[27]}
            Action: {{"action_type": "click", "index": 10, "element_text": "{new_file_name}" }}""",
          f"""Reason: {reason[28]}
            Action: {{"action_type": "click", "index": 10, "element_resource_name": "net.gsantner.markor:id/document__fragment__edit__highlighting_editor" }}""",
          f"""Reason: {reason[29]}
            Action: {{"action_type": "long_press", "index": 10, "element_resource_name": "net.gsantner.markor:id/document__fragment__edit__highlighting_editor" }}""",
          f"""Reason: {reason[30]}
            Action: {{"action_type": "click", "index": 10, "element_text": "Paste" }}""",
          f"""Reason: {reason[31]}
            Action: {{"action_type": "keyboard_enter" }}""",
          f"""Reason: {reason[32]}
            Action: {{"action_type": "click", "index": 10, "element_text": "Save" }}""",
          f"""Reason: {reason[33]}
            Action: {{"action_type": "navigate_back" }}""",
          f"""Reason: {reason[34]}
            Action: {{"action_type": "navigate_back" }}""",
          f"""Reason: {reason[35]}
            Action: {{"action_type": "click", "index": 10, "element_text": "{file_name3}" }}""",
          f"""Reason: {reason[36]}
            Action: {{"action_type": "long_press", "index": 10, "element_resource_name": "net.gsantner.markor:id/document__fragment__edit__highlighting_editor" }}""",
          f"""Reason: {reason[37]}
            Action: {{"action_type": "click", "index": 10, "element_text": "Select all" }}""",
          f"""Reason: {reason[38]}
            Action: {{"action_type": "click", "index": 10, "element_text": "Copy" }}""",\
          f"""Reason: {reason[39]}
            Action: {{"action_type": "navigate_back" }}""",
          f"""Reason: {reason[40]}
            Action: {{"action_type": "navigate_back" }}""",
          f"""Reason: {reason[41]}
            Action: {{"action_type": "click", "index": 10, "element_text": "{new_file_name}" }}""",
          f"""Reason: {reason[42]}
            Action: {{"action_type": "click", "index": 10, "element_resource_name": "net.gsantner.markor:id/document__fragment__edit__highlighting_editor" }}""",
          f"""Reason: {reason[43]}
            Action: {{"action_type": "long_press", "index": 10, "element_resource_name": "net.gsantner.markor:id/document__fragment__edit__highlighting_editor" }}""",
          f"""Reason: {reason[44]}
            Action: {{"action_type": "click", "index": 10, "element_text": "Paste" }}""",
          f"""Reason: {reason[45]}
            Action: {{"action_type": "click", "index": 10, "element_text": "Save" }}""",
          f"""Reason: {reason[46]}
            Action: {{"action_type": "status", "goal_status": "complete"}}""",
      ]

      for index,step in enumerate(steps):
        self.step(step, index)


  def step(self, action_output:str, index:int) -> base_agent.AgentInteractionResult:
    step_data = {
        'before_screenshot': None,
        'after_screenshot': None,
        'before_element_list': None,
        'after_element_list': None,
        'action_prompt': None,
        'action_output': None,
        'action_raw_response': None,
        'summary_prompt': None,
        'summary': None,
        'summary_raw_response': None,
    }
    print('----------step ' + str(index + 1))
    time.sleep(1.0)
    state = self.get_post_transition_state()
    time.sleep(1.0)
    logical_screen_size = self.env.logical_screen_size
    ui_elements = state.ui_elements


    step_data['action_output'] = action_output
    time.sleep(1.0)
    reason, action = m3a_utils.parse_reason_action_output(action_output)
    time.sleep(1.0)
    # If the output is not in the right format, add it to step summary which
    # will be passed to next step and return.
    print('Action: ' + action)
    print('Reason: ' + reason)

    try:
      converted_action = json_action.JSONAction(
          **agent_utils.extract_json(action),
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      print('Failed to convert the output to a valid action.')
      print(str(e))
      step_data['summary'] = (
          'Can not parse the output to a valid action. Please make sure to pick'
          ' the action from the list with the correct json format!'
      )
      self.history.append(step_data)

      return base_agent.AgentInteractionResult(
          False,
          step_data,
      )
      
    if converted_action.action_type == 'navigate_back':
      isCreateButtonExist = False
      for element in ui_elements:
        if element.content_description == 'Create a new file or folder':
          isCreateButtonExist = True
          break
      if isCreateButtonExist:
        return
    
    if index == 41:
      isCreateButtonExist = False
      for element in ui_elements:
        if element.content_description == 'Create a new file or folder':
          isCreateButtonExist = True
          break
      if not isCreateButtonExist:
        time.sleep(2.0)
        state = self.get_post_transition_state()
        time.sleep(1.0)
        logical_screen_size = self.env.logical_screen_size
        ui_elements = state.ui_elements
        
  
    # before_element_list = _generate_ui_elements_description_list_full(
    #     ui_elements,
    #     logical_screen_size,
    # )
    # Only save the screenshot for result visualization.
    step_data['before_screenshot'] = state.pixels.copy()
    step_data['before_screenshot_mark'] = state.pixels.copy()
    step_data['before_element_list'] = ui_elements
 

    step_data['direction'] = converted_action.direction
    step_data['keycode'] = converted_action.keycode
    step_data['content'] = converted_action.text
    step_data['action'] = converted_action.action_type

    if converted_action.action_type in ['click', 'long_press', 'input_text']:
      if converted_action.index is not None and converted_action.index >= len(
          ui_elements
      ):
        print('Index out of range.')
        step_data['summary'] = (
            'The parameter index is out of range. Remember the index must be in'
            ' the UI element list!'
        )
        self.history.append(step_data)
        return base_agent.AgentInteractionResult(False, step_data)
      else:
        target_element = ui_elements[converted_action.index]
        
        _element,_index = get_target_element(converted_action, ui_elements)
        if _element is not None and _index is not None:
          target_element = _element
          converted_action.index = _index 
        
        center_x, center_y = target_element.bbox_pixels.center  # (x1, y1, x2, y2)
        click_point = (center_x, center_y)

        if converted_action.click_x is not None and converted_action.click_y is not None:
          click_point = (converted_action.click_x, converted_action.click_y)

        # 记录点击前坐标
        if converted_action.action_type == 'input_text':
          click_point = None

        step_data['start_coords'] = click_point
        step_data['end_coords'] = click_point

        # Add mark for the target ui element, just used for visualization.ƒ
        m3a_utils.add_ui_element_mark(
            step_data['before_screenshot_mark'],
            ui_elements[converted_action.index],
            converted_action.index,
            logical_screen_size,
            adb_utils.get_physical_frame_boundary(self.env.controller),
            adb_utils.get_orientation(self.env.controller),
            click_point,
        )

    if converted_action.action_type == 'open_app':
        step_data['app_name'] = converted_action.app_name

    if converted_action.action_type == 'status':
      if converted_action.goal_status == 'infeasible':
        print('Agent stopped since it thinks mission impossible.')
      step_data['summary'] = 'Agent thinks the request has been completed.'
      self.history.append(step_data)
      step_data['status'] = converted_action.goal_status
      return base_agent.AgentInteractionResult(
          True,
          step_data,
      )

    if converted_action.action_type == 'answer':
      print('Agent answered with: ' + converted_action.text)

    try:
      result = self.env.execute_action(converted_action)
      if converted_action.action_type == 'scroll':
          step_data['start_coords'] = (result[0],result[1])
          step_data['end_coords'] = (result[2],result[3])

      if converted_action.action_type == 'input_text':
        if step_data is not None and result is not None and not result['input_text_rename']:
          click_step = result
          click_step['action_output'] = step_data['action_output']
          before_screenshot = step_data['before_screenshot']
          step_data['before_screenshot'] = click_step['before_screenshot']

          click_step['before_screenshot'] = before_screenshot
          click_step['before_screenshot_mark'] = step_data['before_screenshot_mark']

          step_data['before_screenshot_mark'] = step_data['before_screenshot']
          step_data['action_output'] = f'Reason: Input {converted_action.text}\n Action:xxx'
          self.history.append(click_step)

    except Exception as e:  # pylint: disable=broad-exception-caught
      print(
          'Some error happened executing the action ',
          converted_action.action_type,
      )
      print(str(e))
      step_data['summary'] = (
          'Some error happened executing the action '
          + converted_action.action_type
      )
      self.history.append(step_data)

      return base_agent.AgentInteractionResult(
          False,
          step_data,
      )

    time.sleep(2.0)
    state = self.get_post_transition_state()
    ui_elements = state.ui_elements

    # Save screenshot only for result visualization.
    step_data['after_screenshot'] = state.pixels.copy()
    step_data['after_element_list'] = ui_elements
    time.sleep(2.0)

    self.history.append(step_data)

    return base_agent.AgentInteractionResult(
        False,
        step_data,
    )


class MarkorCreateNoteFromClipboard(base_agent.EnvironmentInteractingAgent):
  """Text only autonomous agent for Android."""

  def __init__(
      self,
      env: interface.AsyncEnv,
      llm: infer.LlmWrapper,
      name: str = 'T3A',
  ):
    """Initializes a RandomAgent.

    Args:
      env: The environment.
      llm: The text only LLM.
      name: The agent name.
    """
    super().__init__(env, name)
    self.llm = llm
    self.history = []
    self.additional_guidelines = None

  def reset(self, go_home_on_reset: bool = False):
    super().reset(go_home_on_reset)
    self.env.hide_automation_ui()
    self.history = []

  def set_task_guidelines(self, task_guidelines: list[str]) -> None:
    self.additional_guidelines = task_guidelines

  def steps(self, filename: str, text: str):
      file_name, type = filename.split('.')
 
      if type == 'md':
        type_label = 'MarkDown'
        index = 0
      else:
        type_label = 'Plain Text'
        index = 1

      reason = [
        "To start creating a new note in Markor, the first step is to open the Markor app. Since the current screen is the home screen and there's no visible Markor app icon, using the `open_app` action with app_name \"Markor\" is the correct approach to launch the app.",
        "To create a new note, the first step after opening Markor is to initiate the creation process. The UI element with index 1, described as \"Create a new file or folder\", is the appropriate element to click to start creating a new note.",
        "Click the \"Title\" input box to activate the editing state",
        f'To set the correct name for the new note, I need to input "{file_name}" into the editable text field for the name, which is currently visible and ready for input.',
        f"To change the note type to {type_label}, I need to interact with the \"Type\" spinner. Clicking on this spinner will likely open a dropdown or selection menu where \"{type_label}\" can be chosen, which aligns with the task's requirement.",
        f"To set the note type to {type_label} as required, we need to select the \"{type_label}\" option from the type selection menu. UI element 1 corresponds to \"{type_label}\", which is visible and clickable, so clicking on it will achieve this.",
        'To proceed with creating the new note, we need to confirm the entered details (name and type) by clicking the "OK" button. This action will finalize the note creation and allow us to input the content next.',
        f'To add the required text to the note, I need to input "{text}" into the editable text field, which is visible and ready for text entry.',
        f"To paste the copied content into the \"{file_name}\" note, we need to long - press on the editable text field  to bring up the text selection menu with the \"Paste\" option. This is the appropriate action to insert the copied text into the note.",
        f"To paste the copied content into the \"{file_name}\" note, we need to click on the \"Paste\" UI element which is visible and clickable on the current screen, as it appears in the text selection menu after long - pressing the editable text field.",
        'The note has been created with the correct name, type, and content. To finalize the task, saving the note is necessary. Click "More" to find the "Save" button.',
        'The "Save" button may not be here. Go back to find the "Save" button.',
        'The "Save" button is visible and clickable, so clicking it will complete the task.',
        'All required steps to create the note (setting name, type, content, and saving) have been successfully executed. The current screen confirms the note is open with the correct content, and no further actions are needed. Thus, the task can be marked as complete.',
      ]

      steps = [
          f"""Reason: {reason[0]}
            Action: {{"action_type": "open_app", "app_name": "Markor"}}""",
          f"""Reason: {reason[1]}
            Action: {{"action_type": "click", "index": 1, "element_text": "Create a new file or folder" }}""",
          f"""Reason: {reason[2]}
            Action: {{"action_type": "click", "index": 10, "element_text": "Title" }}""",
          f"""Reason: {reason[3]}
            Action: {{"action_type": "input_text", "text": "{file_name}", "index": 1, "element_text": "Title" }}""",
          f"""Reason: {reason[4]}
            Action: {{"action_type": "click", "index": 4, "element_resource_name": "net.gsantner.markor:id/new_file_dialog__type" }}""",
          f"""Reason: {reason[5]}
            Action: {{"action_type": "click", "index": {index}, "element_text": "{type_label}", "element_resource_name": "android:id/text1" }}""",
          f"""Reason: {reason[6]}
            Action: {{"action_type": "click", "index": 11, "element_text": "OK" }}""",
          f"""Reason: {reason[7]}
            Action: {{"action_type": "click", "index": 8, "element_resource_name": "net.gsantner.markor:id/document__fragment__edit__highlighting_editor" }}""",
          f"""Reason: {reason[8]}
            Action: {{"action_type": "long_press", "index": 10, "element_resource_name": "net.gsantner.markor:id/document__fragment__edit__highlighting_editor" }}""",
          f"""Reason: {reason[9]}
            Action: {{"action_type": "click", "index": 10, "element_text": "Paste" }}""",
          f"""Reason: {reason[10]}
            Action: {{"action_type": "click", "index": 6, "element_text": "More"}}""",
          f"""Reason: {reason[11]}
            Action: {{"action_type": "navigate_back"}}""",
          f"""Reason: {reason[12]}
            Action: {{"action_type": "click", "index": 4, "element_text": "Save"}}""",
          f"""Reason: {reason[13]}
            Action: {{"action_type": "status", "goal_status": "complete"}}""",
      ]

      for index,step in enumerate(steps):
        self.step(step, index)


  def step(self, action_output:str, index:int) -> base_agent.AgentInteractionResult:
    step_data = {
        'before_screenshot': None,
        'after_screenshot': None,
        'before_element_list': None,
        'after_element_list': None,
        'action_prompt': None,
        'action_output': None,
        'action_raw_response': None,
        'summary_prompt': None,
        'summary': None,
        'summary_raw_response': None,
    }
    print('----------step ' + str(index + 1))
    time.sleep(1.0)
    state = self.get_post_transition_state()
    time.sleep(1.0)
    logical_screen_size = self.env.logical_screen_size
    ui_elements = state.ui_elements


    step_data['action_output'] = action_output
    time.sleep(1.0)
    reason, action = m3a_utils.parse_reason_action_output(action_output)
    time.sleep(1.0)
    # If the output is not in the right format, add it to step summary which
    # will be passed to next step and return.
    print('Action: ' + action)
    print('Reason: ' + reason)

    try:
      converted_action = json_action.JSONAction(
          **agent_utils.extract_json(action),
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      print('Failed to convert the output to a valid action.')
      print(str(e))
      step_data['summary'] = (
          'Can not parse the output to a valid action. Please make sure to pick'
          ' the action from the list with the correct json format!'
      )
      self.history.append(step_data)

      return base_agent.AgentInteractionResult(
          False,
          step_data,
      )
      
    if converted_action.action_type == 'navigate_back':
      isCreateButtonExist = False
      for element in ui_elements:
        if element.content_description == 'Create a new file or folder':
          isCreateButtonExist = True
          break
      if isCreateButtonExist:
        return
    
    if index == 41:
      isCreateButtonExist = False
      for element in ui_elements:
        if element.content_description == 'Create a new file or folder':
          isCreateButtonExist = True
          break
      if not isCreateButtonExist:
        time.sleep(2.0)
        state = self.get_post_transition_state()
        time.sleep(1.0)
        logical_screen_size = self.env.logical_screen_size
        ui_elements = state.ui_elements
        
  
    # before_element_list = _generate_ui_elements_description_list_full(
    #     ui_elements,
    #     logical_screen_size,
    # )
    # Only save the screenshot for result visualization.
    step_data['before_screenshot'] = state.pixels.copy()
    step_data['before_screenshot_mark'] = state.pixels.copy()
    step_data['before_element_list'] = ui_elements
 

    step_data['direction'] = converted_action.direction
    step_data['keycode'] = converted_action.keycode
    step_data['content'] = converted_action.text
    step_data['action'] = converted_action.action_type

    if converted_action.action_type in ['click', 'long_press', 'input_text']:
      if converted_action.index is not None and converted_action.index >= len(
          ui_elements
      ):
        print('Index out of range.')
        step_data['summary'] = (
            'The parameter index is out of range. Remember the index must be in'
            ' the UI element list!'
        )
        self.history.append(step_data)
        return base_agent.AgentInteractionResult(False, step_data)
      else:
        target_element = ui_elements[converted_action.index]
        
        _element,_index = get_target_element(converted_action, ui_elements)
        if _element is not None and _index is not None:
          target_element = _element
          converted_action.index = _index 
        
        center_x, center_y = target_element.bbox_pixels.center  # (x1, y1, x2, y2)
        click_point = (center_x, center_y)

        if converted_action.click_x is not None and converted_action.click_y is not None:
          click_point = (converted_action.click_x, converted_action.click_y)

        # 记录点击前坐标
        if converted_action.action_type == 'input_text':
          click_point = None

        step_data['start_coords'] = click_point
        step_data['end_coords'] = click_point

        # Add mark for the target ui element, just used for visualization.ƒ
        m3a_utils.add_ui_element_mark(
            step_data['before_screenshot_mark'],
            ui_elements[converted_action.index],
            converted_action.index,
            logical_screen_size,
            adb_utils.get_physical_frame_boundary(self.env.controller),
            adb_utils.get_orientation(self.env.controller),
            click_point,
        )

    if converted_action.action_type == 'open_app':
        step_data['app_name'] = converted_action.app_name

    if converted_action.action_type == 'status':
      if converted_action.goal_status == 'infeasible':
        print('Agent stopped since it thinks mission impossible.')
      step_data['summary'] = 'Agent thinks the request has been completed.'
      self.history.append(step_data)
      step_data['status'] = converted_action.goal_status
      return base_agent.AgentInteractionResult(
          True,
          step_data,
      )

    if converted_action.action_type == 'answer':
      print('Agent answered with: ' + converted_action.text)

    try:
      result = self.env.execute_action(converted_action)
      if converted_action.action_type == 'scroll':
          step_data['start_coords'] = (result[0],result[1])
          step_data['end_coords'] = (result[2],result[3])

      if converted_action.action_type == 'input_text':
        if step_data is not None and result is not None and not result['input_text_rename']:
          click_step = result
          click_step['action_output'] = step_data['action_output']
          before_screenshot = step_data['before_screenshot']
          step_data['before_screenshot'] = click_step['before_screenshot']

          click_step['before_screenshot'] = before_screenshot
          click_step['before_screenshot_mark'] = step_data['before_screenshot_mark']

          step_data['before_screenshot_mark'] = step_data['before_screenshot']
          step_data['action_output'] = f'Reason: Input {converted_action.text}\n Action:xxx'
          self.history.append(click_step)

    except Exception as e:  # pylint: disable=broad-exception-caught
      print(
          'Some error happened executing the action ',
          converted_action.action_type,
      )
      print(str(e))
      step_data['summary'] = (
          'Some error happened executing the action '
          + converted_action.action_type
      )
      self.history.append(step_data)

      return base_agent.AgentInteractionResult(
          False,
          step_data,
      )

    time.sleep(2.0)
    state = self.get_post_transition_state()
    ui_elements = state.ui_elements

    # Save screenshot only for result visualization.
    step_data['after_screenshot'] = state.pixels.copy()
    step_data['after_element_list'] = ui_elements
    time.sleep(2.0)

    self.history.append(step_data)

    return base_agent.AgentInteractionResult(
        False,
        step_data,
    )


class MarkorCreateNoteAndSms(base_agent.EnvironmentInteractingAgent):
  """Text only autonomous agent for Android."""

  def __init__(
      self,
      env: interface.AsyncEnv,
      llm: infer.LlmWrapper,
      name: str = 'T3A',
  ):
    """Initializes a RandomAgent.

    Args:
      env: The environment.
      llm: The text only LLM.
      name: The agent name.
    """
    super().__init__(env, name)
    self.llm = llm
    self.history = []
    self.additional_guidelines = None

  def reset(self, go_home_on_reset: bool = False):
    super().reset(go_home_on_reset)
    self.env.hide_automation_ui()
    self.history = []

  def set_task_guidelines(self, task_guidelines: list[str]) -> None:
    self.additional_guidelines = task_guidelines

  def steps(self, filename: str, text: str, phone: str):
      file_name, type = filename.split('.')
 
      if type == 'md':
        type_label = 'MarkDown'
        index = 0
      else:
        type_label = 'Plain Text'
        index = 1

      reason = [
        "To start creating a new note in Markor, the first step is to open the Markor app. Since we're on the home screen, using the `open_app` action with app_name \"Markor\" is the appropriate next step.",
        "To create a new note, the next logical step is to interact with the \"Create a new file or folder\" button, which is visible and clickable. This will initiate the process of creating a new note.",
        "Click the \"Title\" input box to activate the editing state",
        f'To set the correct name for the new note, I need to input "{file_name}" into the editable text field for the name, which is currently visible and ready for input.',
        f"To change the note type to {type_label}, I need to interact with the \"Type\" spinner. Clicking on this spinner will likely open a dropdown or selection menu where \"{type_label}\" can be chosen, which aligns with the task's requirement.",
        f"To set the note type to {type_label} as required, we need to select the \"{type_label}\" option from the dropdown. this element corresponds to \"{type_label}\", so clicking on it will achieve this.",
        "To proceed with creating the new note, we need to confirm the entered details (name and type) by clicking the \"OK\" button. This action will finalize the note creation and allow us to input the content next.",
        "To add the required text to the note, I need to switch the note to edit mode. Now click on the note to trigger the edit.",
        f'To add the required text to the note, I need to input \"{text}\" into the editable text field, which is visible and ready for text entry.',
        "The note has been created with the correct name, type, and content. We need to save the note, click \"More\" to find the \"Save\" button.",
        "The \"Save\" button may not be here. Go back to find the \"Save\" button.",
        "The \"Save\" button is visible and clickable, so clicking it will complete creating the note.",
        "To share the note content, I need to access the sharing options. The \"More options\" menu is likely to contain the share functionality, so clicking on it is the next logical step.",
        "To proceed with sharing the note's content, I need to select the \"Share\" option from the opened menu. This will allow access to sharing methods, including SMS, which is required for the task. this element corresponds to \"Share\" and is visible, so clicking it is the logical next step.",
        "To share the note content, we need to select the \"Plain Text\" option from the current submenu. This aligns with the task's requirement to share the entire content by Plain Text, which will likely lead to the next step of choosing the SMS app.",
        "To share the note content via SMS using Simple SMS Messenger, we need to select the \"SMS Messenger\" app from the sharing options. this element corresponds to \"SMS Messenger\", so clicking on it will allow us to proceed with composing the SMS to the specified phone number.",
        f"To share the note via SMS, we need to input the recipient's phone number {phone} into the \"Add Contact or Number…\" editable text field. Using the `input_text` action directly on this field will efficiently enter the number, progressing toward sending the SMS.",
        f"Input {phone}",
        "To proceed with sending the SMS, we need to confirm the entered phone number. this element is the \"confirm\" button (ImageView) next to the recipient number field, which is visible and clickable. Clicking it will likely transition to the conversation screen where the message can be sent.",
        "To complete the task of sharing the note content via SMS, the next step is to send the message. this element is the \"SMS\" send button, which is clickable and visible, so clicking it will send the message to the specified phone number.",
        "All required steps to create the note, populate it, and share via SMS to the specified number have been successfully executed. The message is visible in the chat, confirming completion.",
      ]

      steps = [
          f"""Reason: {reason[0]}
            Action: {{"action_type": "open_app", "app_name": "Markor"}}""",
          f"""Reason: {reason[1]}
            Action: {{"action_type": "click", "index": 1, "element_text": "Create a new file or folder" }}""",
          f"""Reason: {reason[2]}
            Action: {{"action_type": "click", "index": 10, "element_text": "Title" }}""",
          f"""Reason: {reason[3]}
            Action: {{"action_type": "input_text", "text": "{file_name}", "index": 1, "element_text": "Title" }}""",
          f"""Reason: {reason[4]}
            Action: {{"action_type": "click", "index": 4, "element_resource_name": "net.gsantner.markor:id/new_file_dialog__type" }}""",
          f"""Reason: {reason[5]}
            Action: {{"action_type": "click", "index": {index}, "element_text": "{type_label}", "element_resource_name": "android:id/text1" }}""",
          f"""Reason: {reason[6]}
            Action: {{"action_type": "click", "index": 11, "element_text": "OK" }}""",
          f"""Reason: {reason[7]}
            Action: {{"action_type": "click", "index": 8, "element_resource_name": "net.gsantner.markor:id/document__fragment__edit__highlighting_editor" }}""",
          f"""Reason: {reason[8]}
            Action: {{"action_type": "input_text", "text": "{text}", "index": 10, "element_resource_name": "net.gsantner.markor:id/document__fragment__edit__highlighting_editor" }}""",
          f"""Reason: {reason[9]}
            Action: {{"action_type": "click", "index": 6, "element_text": "More"}}""",
          f"""Reason: {reason[10]}
            Action: {{"action_type": "navigate_back"}}""",
          f"""Reason: {reason[11]}
            Action: {{"action_type": "click", "index": 4, "element_text": "Save"}}""",
          f"""Reason: {reason[12]}
            Action: {{"action_type": "click", "index": 6, "element_text": "More"}}""",
          f"""Reason: {reason[13]}
            Action: {{"action_type": "click", "index": 4, "element_text": "Share"}}""",
          f"""Reason: {reason[14]}
            Action: {{"action_type": "click", "index": 4, "element_text": "Plain Text"}}""",
          f"""Reason: {reason[15]}
            Action: {{"action_type": "click", "element_text": "SMS Messenger"}}""",
          f"""Reason: {reason[16]}
            Action: {{"action_type": "click", "element_text": "Add Contact or Number…"}}""",
          f"""Reason: {reason[17]}
            Action: {{"action_type": "input_text", "text": "{phone}", "index": 1, "element_text": "Add Contact or Number…"}}""",
          f"""Reason: {reason[18]}
            Action: {{"action_type": "click", "element_resource_name": "com.simplemobiletools.smsmessenger:id/new_conversation_confirm"}}""",
          f"""Reason: {reason[19]}
            Action: {{"action_type": "click", "element_text": "SMS"}}""",
          f"""Reason: {reason[20]}
            Action: {{"action_type": "status", "goal_status": "complete"}}""",
      ]

      for index,step in enumerate(steps):
        self.step(step, index)

  def step(self, action_output:str, index:int) -> base_agent.AgentInteractionResult:
    step_data = {
        'before_screenshot': None,
        'after_screenshot': None,
        'before_element_list': None,
        'after_element_list': None,
        'action_prompt': None,
        'action_output': None,
        'action_raw_response': None,
        'summary_prompt': None,
        'summary': None,
        'summary_raw_response': None,
    }
    print('----------step ' + str(index + 1))
    time.sleep(1.0)
    state = self.get_post_transition_state()
    time.sleep(1.0)
    logical_screen_size = self.env.logical_screen_size
    ui_elements = state.ui_elements


    step_data['action_output'] = action_output
    time.sleep(1.0)
    reason, action = m3a_utils.parse_reason_action_output(action_output)
    time.sleep(1.0)
    # If the output is not in the right format, add it to step summary which
    # will be passed to next step and return.
    print('Action: ' + action)
    print('Reason: ' + reason)

    try:
      converted_action = json_action.JSONAction(
          **agent_utils.extract_json(action),
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      print('Failed to convert the output to a valid action.')
      print(str(e))
      step_data['summary'] = (
          'Can not parse the output to a valid action. Please make sure to pick'
          ' the action from the list with the correct json format!'
      )
      self.history.append(step_data)

      return base_agent.AgentInteractionResult(
          False,
          step_data,
      )
      
    if converted_action.action_type == 'navigate_back':
      isCreateButtonExist = False
      for element in ui_elements:
        if element.content_description == 'Create a new file or folder':
          isCreateButtonExist = True
          break
      if isCreateButtonExist:
        return
    
    if index == 41:
      isCreateButtonExist = False
      for element in ui_elements:
        if element.content_description == 'Create a new file or folder':
          isCreateButtonExist = True
          break
      if not isCreateButtonExist:
        time.sleep(2.0)
        state = self.get_post_transition_state()
        time.sleep(1.0)
        logical_screen_size = self.env.logical_screen_size
        ui_elements = state.ui_elements
        
  
    # before_element_list = _generate_ui_elements_description_list_full(
    #     ui_elements,
    #     logical_screen_size,
    # )
    # Only save the screenshot for result visualization.
    step_data['before_screenshot'] = state.pixels.copy()
    step_data['before_screenshot_mark'] = state.pixels.copy()
    step_data['before_element_list'] = ui_elements
 

    step_data['direction'] = converted_action.direction
    step_data['keycode'] = converted_action.keycode
    step_data['content'] = converted_action.text
    step_data['action'] = converted_action.action_type

    if converted_action.action_type in ['click', 'long_press', 'input_text']:
      if converted_action.index is not None and converted_action.index >= len(
          ui_elements
      ):
        print('Index out of range.')
        step_data['summary'] = (
            'The parameter index is out of range. Remember the index must be in'
            ' the UI element list!'
        )
        self.history.append(step_data)
        return base_agent.AgentInteractionResult(False, step_data)
      else:
        target_element = None
        if converted_action.index is not None:
          target_element = ui_elements[converted_action.index]
        # if converted_action.element_text == 'SMS Messenger':
        #   print(f'ui_elements>>>>>>>>>>>>>: {ui_elements}')
        _element,_index = get_target_element(converted_action, ui_elements)
        if _element is not None and _index is not None:
          target_element = _element
          converted_action.index = _index 

        if target_element is None:
          print('Target element not found.')
          step_data['summary'] = (
              'The target element is not found. Remember the index must be in'
              ' the UI element list!'
          )
          self.history.append(step_data)
          return base_agent.AgentInteractionResult(False, step_data)
        
        center_x, center_y = target_element.bbox_pixels.center  # (x1, y1, x2, y2)
        click_point = (center_x, center_y)

        if converted_action.click_x is not None and converted_action.click_y is not None:
          click_point = (converted_action.click_x, converted_action.click_y)

        # 记录点击前坐标
        if converted_action.action_type == 'input_text':
          click_point = None

        step_data['start_coords'] = click_point
        step_data['end_coords'] = click_point

        # Add mark for the target ui element, just used for visualization.ƒ
        m3a_utils.add_ui_element_mark(
            step_data['before_screenshot_mark'],
            ui_elements[converted_action.index],
            converted_action.index,
            logical_screen_size,
            adb_utils.get_physical_frame_boundary(self.env.controller),
            adb_utils.get_orientation(self.env.controller),
            click_point,
        )

    if converted_action.action_type == 'open_app':
        step_data['app_name'] = converted_action.app_name

    if converted_action.action_type == 'status':
      if converted_action.goal_status == 'infeasible':
        print('Agent stopped since it thinks mission impossible.')
      step_data['summary'] = 'Agent thinks the request has been completed.'
      self.history.append(step_data)
      step_data['status'] = converted_action.goal_status
      return base_agent.AgentInteractionResult(
          True,
          step_data,
      )

    if converted_action.action_type == 'answer':
      print('Agent answered with: ' + converted_action.text)

    try:
      result = self.env.execute_action(converted_action)
      if converted_action.action_type == 'scroll':
          step_data['start_coords'] = (result[0],result[1])
          step_data['end_coords'] = (result[2],result[3])

      if converted_action.action_type == 'input_text':
        if step_data is not None and result is not None and not result['input_text_rename']:
          click_step = result
          click_step['action_output'] = step_data['action_output']
          before_screenshot = step_data['before_screenshot']
          step_data['before_screenshot'] = click_step['before_screenshot']

          click_step['before_screenshot'] = before_screenshot
          click_step['before_screenshot_mark'] = step_data['before_screenshot_mark']

          step_data['before_screenshot_mark'] = step_data['before_screenshot']
          step_data['action_output'] = f'Reason: Input {converted_action.text}\n Action:xxx'
          self.history.append(click_step)

    except Exception as e:  # pylint: disable=broad-exception-caught
      print(
          'Some error happened executing the action ',
          converted_action.action_type,
      )
      print(str(e))
      step_data['summary'] = (
          'Some error happened executing the action '
          + converted_action.action_type
      )
      self.history.append(step_data)

      return base_agent.AgentInteractionResult(
          False,
          step_data,
      )

    time.sleep(2.0)
    state = self.get_post_transition_state()
    ui_elements = state.ui_elements

    # Save screenshot only for result visualization.
    step_data['after_screenshot'] = state.pixels.copy()
    step_data['after_element_list'] = ui_elements
    time.sleep(2.0)

    self.history.append(step_data)

    return base_agent.AgentInteractionResult(
        True,
        step_data,
    )
