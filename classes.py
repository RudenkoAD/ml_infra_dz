from enum import Enum
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass

class Action(Enum):
    SPLIT = "SPLIT"
    STEAL = "STEAL"

class Event(Enum):
    ACTION = "ACTION"
    MESSAGE = "MESSAGE"

class Personality(Enum):
    """Enum representing different agent personalities."""
    TRUSTING = "trusting"
    SUSPICIOUS = "suspicious"
    LIAR = "liar"

@dataclass
class HistoryEvent:
    type: Event
    author: str  # Player ID of the author
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
    agent_1_id: str
    agent_2_id: str
    total_rewards: Tuple[float, float]
    communication_history: list[HistoryEvent]

