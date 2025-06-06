from dataclasses import dataclass
from re import A
from openai import OpenAI
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
from classes import Action, HistoryEvent, Event
from agents.prompts import PromptManager, Personality
from logging import getLogger

from models.providers.base_provider import Provider

log = getLogger(__name__)

class LLMAgent:
    player_id: str
    personality: Personality = Personality.TRUSTING

    def __init__(
        self,
        player_id: str,
        provider: Provider,
        personality: Personality = Personality.TRUSTING
    ):
        self.player_id = player_id
        self.provider = provider
        self.personality = personality

    def _extract_message(self, response: str) -> str:
        """Extract the message from the API's response."""
        # Remove prompt if included
        prompt_end = response.find("THOUGHTS:") or response.find("MESSAGE:")
        if prompt_end != -1:
            response = response[prompt_end:]
        
        # Split response into thoughts and message
        parts = response.split("MESSAGE:")
        if len(parts) != 2:
            # Fallback: return first sentence or default
            sentences = response.split('.')
            return sentences[0].strip() if sentences[0].strip() else "I propose we both SPLIT for mutual benefit."
        
        # Extract and return the message
        message = parts[1].strip()
        if len(message) > 100:
            message = message[:97] + "..."
        return message
    
    def _parse_response(self, response: str) -> Action:
        """Parse the API's response to determine the action."""
        # Remove prompt and thoughts if present
        prompt_end = response.find("THOUGHTS:") or response.find("MESSAGE:") or response.find("SPLIT") or response.find("STEAL")
        if prompt_end != -1:
            response = response[prompt_end:]
        
        response = response.lower().strip()
        if "split" in response:
            return Action.SPLIT
        elif "steal" in response:
            return Action.STEAL
        else:
            return Action.SPLIT

    def query(self, prompt: str) -> str:
        """Query the provider with the given prompt."""
        try:
            response = self.provider.prompt(prompt)
            log.debug(f"Provider response: {response}")
            return response
        except Exception as e:
            log.error(f"Error querying provider: {e}")
            raise ValueError("Failed to get a valid response from the provider.")
    
    def get_message(self, communication_history: list[HistoryEvent]) -> str:
        """
        Generate a message to send to the opponent.
        """
        prompt = PromptManager.construct_prompt(
            communication_history=communication_history,
            personality=self.personality,
            player_id=self.player_id,
            is_action=False
        )
        log.debug(f"Prompt: '{prompt}'")
        response = self.query(prompt)
        log.debug(f"Response: '{response}'")
        return self._extract_message(response)
    
    def get_action(self, communication_history: list[HistoryEvent]) -> Action:
        """
        Generate an action based on the game history, opponent's history, and communication.
        """
        prompt = PromptManager.construct_prompt(
            communication_history=communication_history,
            personality=self.personality,
            player_id=self.player_id,
            is_action=True
        )
        log.debug(f"Prompt: '{prompt}'")
        response = self.query(prompt)
        log.debug(f"Response: '{response}'")
        return self._parse_response(response)
