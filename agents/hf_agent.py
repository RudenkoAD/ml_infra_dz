import os
from openai import OpenAI
from typing import Optional, Dict, Any
from classes import Action, HistoryEvent, Event
from agents.prompts import PromptManager, Personality
from agents.base_agent import Agent
from logging import getLogger

log = getLogger(__name__)

class HFAgent(Agent):
    def __init__(
        self,
        model_name: str = "openai-community/gpt2",
        temperature: float = 0.5,
        max_new_tokens: int = 4096,
        personality: Personality = Personality.TRUSTING,
        player_id: Optional[str] = None,
    ):
        self.api_token = os.getenv("HUGGINGFACE_API_TOKEN")
        if not self.api_token:
            raise ValueError("HUGGINGFACE_API_TOKEN environment variable is not set.")
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.personality = personality
        self.history: list[Dict[str, Any]] = []
        self.player_id = player_id if player_id else f"hf_agent_{model_name.replace('/', '_')}"
        self.client = OpenAI(
            base_url="https://router.huggingface.co/nscale/v1",
            api_key=self.api_token
        )
    
    def _query(self, prompt: str) -> str:
        """Query the OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_completion_tokens=self.max_new_tokens
        )
        if response.choices[0].message.content is None:
            log.error(response.choices[0].message)
            log.error("API response is empty or invalid.")
            raise ValueError("API response is empty or invalid.")
        return response.choices[0].message.content.strip()
