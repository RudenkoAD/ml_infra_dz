from abc import ABC, abstractmethod
from typing import Optional

class Provider(ABC):
    @abstractmethod
    def __init__(self, api_key: Optional[str] = None, model_name: str = "deepseek/deepseek-r1-distill-qwen-7b", seed: int = 42):
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def prompt(self, prompt: str) -> str:
        """
        Send a prompt to the provider's API and return the response.

        Args:
            prompt (str): The prompt to send.

        Returns:
            str: The response from the provider.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
