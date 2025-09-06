from dataclasses import dataclass
from ollama import Client
from .base import Engine, EngineConfig

@dataclass
class OllamaConfig(EngineConfig):
    pass

class OllamaEngine(Engine):
    def __init__(self, cfg: OllamaConfig):
        self.cfg = cfg
        self.client = Client()

    def complete(self, prompt: str, system: str | None = None) -> str:
        msgs = []
        if system: msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        r = self.client.chat(model=self.cfg.model, messages=msgs, options={
            "temperature": self.cfg.temperature
        })
        return r["message"]["content"]