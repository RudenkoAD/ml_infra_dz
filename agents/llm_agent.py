from __future__ import annotations
import random

from classes import Action, GameState
from promptsets.base_promptset import BasePromptSet
from logging import getLogger
from models.providers.base_provider import Provider

log = getLogger(__name__)

class LLMAgent:
    player_id: str
    promptset: BasePromptSet

    def __init__(
        self,
        player_id: str,
        provider: Provider,
        promptset: BasePromptSet
    ):
        self.player_id = player_id
        self.provider = provider
        self.promptset = promptset

    def _extract_message(self, response: str) -> str:
        """Extract the message from the API's response."""
        return response.strip()

    def _parse_response(self, response: str) -> Action:
        """Parse the API's response to determine the action."""
        response = response.strip().lower()
        if "split" in response:
            return Action.SPLIT
        elif "steal" in response:
            return Action.STEAL
        else:
            log.warning(f"Invalid action response: '{response}'")
            raise ValueError(f"Invalid action response: '{response}'. Expected 'SPLIT' or 'STEAL'.")
        

    def query(self, prompt: str) -> str:
        """Query the provider with the given prompt."""
        try:
            response = self.provider.prompt(prompt)
            log.debug(f"Provider response: {response}")
            return response
        except Exception as e:
            log.error(f"Error querying provider: {e}")
            raise ValueError("Failed to get a valid response from the provider.")
    
    def get_message(self, state: GameState) -> str:
        """Generate a message to send to the opponent."""
        while True:
            try:
                # Construct the prompt for communication
                prompt = self.promptset.construct_prompt(
                    player_id=self.player_id,
                    state=state,
                    is_action=False
                )
                response = self.query(prompt)
                message = self._extract_message(response)
            except ValueError as e:
                # retry on error
                log.warning(f"Retrying message query due to error: {e}")
                continue
            return message

    def get_action(self, state: GameState) -> Action:
        """Generate an action based on the game history."""
        while True:
            try:
                prompt = self.promptset.construct_prompt(
                    player_id=self.player_id,
                    state=state,
                    is_action=True
                )
                response = self.query(prompt)
                action = self._parse_response(response)
            except ValueError as e:
                #retry
                log.warning(f"Retrying action query due to error: {e}")
                continue
            return action

    def clone(self) -> LLMAgent:
        """
        Create a clone of the agent with the same player ID and personality.
        """
        return LLMAgent(
            player_id=self.player_id + random.randint(1, 10000).__str__(),
            provider=self.provider,
            promptset=self.promptset
        )
