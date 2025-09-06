from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from .base import Engine, EngineConfig

@dataclass
class HFConfig(EngineConfig):
    max_new_tokens: int = 256
    device_map: str = "auto"

class HFEngine(Engine):
    def __init__(self, cfg: HFConfig):
        self.config = cfg
        tok = AutoTokenizer.from_pretrained("gpt2")
        mdl = AutoModelForCausalLM.from_pretrained("gpt2", device_map=cfg.device_map)
        self.pipe = pipeline("text-generation", model=mdl, tokenizer=tok, device_map=cfg.device_map)

    def complete(self,prompt: str, system: str | None = None) -> str:
        full = (system + "\n" if system else "") + prompt
        out = self.pipe(full, do_sample=self.cfg.temperature > 0,
                        temperature=max(0.1, self.cfg.temperature),
                        max_new_tokens=self.cfg.max_new_tokens)
        text = out[0]["generated_text"]
        return text[len(full):].strip() if text.startswith(full) else text.strip()