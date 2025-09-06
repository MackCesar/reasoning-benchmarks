from dataclasses import dataclass

@dataclass
class EngineConfig:
    model: str
    temperature: float = 0.2
    max_tokens: int = 512

class Engine:
    def complete(self,prompt: str, system: str | None = None) -> str:
        raise NotImplementedError