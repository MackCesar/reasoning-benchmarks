import os, json
from openai import OpenAI
from dataclasses import dataclass
from dotenv import load_dotenv
from .base import Engine, EngineConfig

load_dotenv()

@dataclass
class OpenAIConfig(EngineConfig):
    pass

class OpenAIEngine(Engine):
    def __init__(self, cfg: OpenAIConfig):
        self.config = cfg
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def complete(self,prompt: str, system: str | None = None) -> str:
        msg = []
        if system: msg.append({"role": "system", "content": system})
        msg.append({"role": "user", "content": prompt})
        resp = self.client.chat.completions.create(
            model=self.config.model,
            messages=msg,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return resp.choices[0].message.content