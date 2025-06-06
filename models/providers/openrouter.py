
import os
from typing import Optional
from openai import OpenAI

from logging import getLogger

from models.providers.base_provider import Provider
log = getLogger(__name__)

class OpenRouterProvider(Provider):
    def __init__(self, api_key: Optional[str] = None, model_name: str = "deepseek/deepseek-r1-distill-qwen-7b"):
        self.api_key = os.getenv("OPENROUTER_API_KEY") if not api_key else api_key
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set.")
        self.model_name = model_name
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key
        )
        log.debug(f"Initialized OpenRouterProvider with model: {self.model_name}")

    def prompt(self, prompt: str) -> str:
        """Send a prompt to the OpenRouter API and return the response."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_completion_tokens=4096
        )
        log.debug(f"OpenRouter response: {response}")
        if response.choices[0].message.content is None:
            raise ValueError("API response is empty or invalid.")
        
        return response.choices[0].message.content.strip()
