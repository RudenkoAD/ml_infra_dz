from typing import Optional, List, Tuple
from classes import Action

class BaseStrategy:
    """Base class for all strategies."""
    name = "base"
    def get_action(self, opponent_history: Optional[List[dict]] = None, communication_history: Optional[List[Tuple[str, str]]] = None) -> Action:
        raise NotImplementedError

class AlwaysStealStrategy(BaseStrategy):
    """Strategy that always chooses to steal."""
    name = "always_steal"
    def get_action(self, opponent_history: Optional[List[dict]] = None, communication_history: Optional[List[Tuple[str, str]]] = None) -> Action:
        return Action.STEAL

class AlwaysSplitStrategy(BaseStrategy):
    """Strategy that always chooses to split."""
    name = "always_split"
    def get_action(self, opponent_history: Optional[List[dict]] = None, communication_history: Optional[List[Tuple[str, str]]] = None) -> Action:
        return Action.SPLIT

class TitForTatStrategy(BaseStrategy):
    """Strategy that starts with split and then copies the opponent's previous move."""
    name = "tit_for_tat"
    def get_action(self, opponent_history: Optional[List[dict]] = None, communication_history: Optional[List[Tuple[str, str]]] = None) -> Action:
        if not opponent_history:
            return Action.SPLIT
        
        # Get the opponent's last action
        last_game = opponent_history[-1]
        opponent_action = last_game['actions'][1]  # Second action is opponent's
        
        return opponent_action 
