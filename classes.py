from enum import Enum
from typing import List, Tuple, Optional
from dataclasses import dataclass

class Action(Enum):
    SPLIT = "SPLIT"
    STEAL = "STEAL"

class Event(Enum):
    ACTION = "ACTION"
    MESSAGE = "MESSAGE"

class PromptSet(Enum):
    BASE = "BASE"

@dataclass
class HistoryEvent:
    type: Event
    author: str  # Player ID of the author
    message: Optional[str] = None
    action: Optional[Action] = None
    def __str__(self) -> str:
        if self.type == Event.ACTION:
            return f"{self.author} chose {self.action}"
        else:
            return f"{self.author} said {self.message}"
    
@dataclass
class GameState:
    communication_history: list[HistoryEvent]
    current_turn: int
    max_turns: int
    round_number: int
    total_rounds: int

@dataclass
class RoundResult:
    round: int
    actions: Tuple[Action, Action]
    rewards: Tuple[float, float]
    result: str
    communication_history: list[HistoryEvent]

@dataclass
class GameResult:
    rounds: List[RoundResult]
    first_agent_id: str
    second_agent_id: str
    total_rewards: Tuple[float, float]
    communication_history: list[HistoryEvent]

