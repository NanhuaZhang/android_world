# android_world/agents/doubao_agent.py

import os
import requests
from android_world.agents import base_agent

class DoubaoReActAgent(base_agent.EnvironmentInteractingAgent):
    def __init__(self):
        self.token = os.getenv("DOUBAO_API_KEY")
        self.model = 'doubao-1-5-ui-tars-250428'
        self.api_url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
        self.messages = []

    def reset(self, task_description):
        self.messages = [{"role":"system","content":task_description}]

    def step(self, observation):
        self.messages.append({"role":"user","content":observation})
        resp = requests.post(self.api_url, headers={
            "Authorization":f"Bearer {self.token}",
            "Content-Type":"application/json"
        }, json={"model":self.model,"messages":self.messages})
        resp.raise_for_status()
        reply = resp.json()["choices"][0]["message"]["content"]
        self.messages.append({"role":"assistant","content":reply})
        return reply.strip()