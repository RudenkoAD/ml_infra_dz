from enum import Enum
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass

class Action(Enum):
    SPLIT = "SPLIT"
    STEAL = "STEAL"

@dataclass
class GameState:
    communication_history: List[Tuple[str, str]]
    current_turn: int
    max_turns: int
    game_id: int
    done: bool
    round_number: int
    total_rounds: int

@dataclass
class RoundResult:
    game_id: int
    round: int
    actions: Tuple[Action, Action]
    rewards: Tuple[float, float]
    result: str
    communication_history: List[Tuple[str, str]]

@dataclass
class GameResult:
    game_id: int
    rounds: List[RoundResult]
    total_rewards: Tuple[float, float]
    communication_history: List[Tuple[str, str]]

class Agent:
    group_id: str = "base_group"
    history: list[Dict[str, Any]] = []
    def get_message(self, opponent_history: Optional[list[Dict[str, Any]]] = None) -> str:
        """
        Generate a message to send to the opponent.
        If using a preset strategy, generate a message based on the strategy.
        """
        return "i am a base agent"
    
    def get_action(self, opponent_history: Optional[list[Dict[str, Any]]] = None, communication_history: Optional[list[Tuple[str, str]]] = None) -> Action:
        """
        Generate an action based on the game history, opponent's history, and communication.
        If using a preset strategy, use that instead of the LLM.
        """
        return Action.SPLIT
    
    def update_history(self, game_result: Dict[str, Any]):
        """Update the agent's game history and group score."""
        pass
