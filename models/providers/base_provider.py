from abc import ABC, abstractmethod
from typing import Optional

class Provider(ABC):
    @abstractmethod
    def __init__(self, api_key: Optional[str] = None, model_name: str = "deepseek/deepseek-r1-distill-qwen-7b", seed: int = 42):
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def prompt(self, prompt: str) -> str:
        raise NotImplementedError("This method should be overridden by subclasses.")
