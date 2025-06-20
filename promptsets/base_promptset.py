from abc import ABC, abstractmethod
from classes import GameState, HistoryEvent

class BasePromptSet(ABC):
    name = "BasePromptSet"
    @staticmethod
    @abstractmethod
    def get_base_prompt(cur_round: int, total_rounds: int) -> str:
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    @staticmethod
    @abstractmethod
    def translate_history_to_prompt(player_id: str, communication_history: list[HistoryEvent]) -> str:
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    @staticmethod
    @abstractmethod
    def construct_communication_prompt(player_id: str, communication_history: list[HistoryEvent]) -> str:
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    @staticmethod
    @abstractmethod
    def construct_action_prompt(player_id: str, communication_history: list[HistoryEvent]) -> str:
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    @staticmethod
    @abstractmethod
    def construct_prompt(player_id: str, state: GameState, is_action: bool = False) -> str:
        raise NotImplementedError("This method should be overridden by subclasses.")
