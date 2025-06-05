from enum import Enum
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass

class Action(Enum):
    SPLIT = "SPLIT"
    STEAL = "STEAL"

class Event(Enum):
    ACTION = "ACTION"
    MESSAGE = "MESSAGE"

class Player(Enum):
    AGENT_1 = "AGENT_1"
    AGENT_2 = "AGENT_2"

@dataclass
class HistoryEvent:
    type: Event
    author: Player
    message: Optional[str] = None
    action: Optional[Action] = None
    
@dataclass
class GameState:
    communication_history: list[HistoryEvent]
    current_turn: int
    max_turns: int
    game_id: int
    round_number: int
    total_rounds: int

@dataclass
class RoundResult:
    game_id: int
    round: int
    actions: Tuple[Action, Action]
    rewards: Tuple[float, float]
    result: str
    communication_history: list[HistoryEvent]

@dataclass
class GameResult:
    game_id: int
    rounds: List[RoundResult]
    group_1_id: str
    group_2_id: str
    total_rewards: Tuple[float, float]
    communication_history: list[HistoryEvent]

    
class Agent:
    group_id: str = "base_group"
    history: list[Dict[str, Any]] = []
    player_id: Player
    def get_message(self, communication_history: list[HistoryEvent]) -> str:
        """
        Generate a message to send to the opponent.
        If using a preset strategy, generate a message based on the strategy.
        """
        return "i am a base agent"
    
    def get_action(self, communication_history: list[HistoryEvent]) -> Action:
        """
        Generate an action based on the game history, opponent's history, and communication.
        If using a preset strategy, use that instead of the LLM.
        """
        return Action.SPLIT

