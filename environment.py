from operator import is_
from typing import Tuple, Optional
import random

from wandb import agent
import wandb
from classes import Action, Event, GameState, HistoryEvent, RoundResult, GameResult
from agents.llm_agent import LLMAgent
import logging

log = logging.getLogger(__name__)

class SplitOrStealEnv:
    def __init__(self):
        self.state = GameState(
            communication_history=[],
            current_turn=0,
            max_turns=4,
            round_number=0,
            total_rounds=1
        )
        self.reward_matrix = {
            (Action.SPLIT, Action.SPLIT): (2, 2),
            (Action.SPLIT, Action.STEAL): (-1, 3),
            (Action.STEAL, Action.SPLIT): (3, -1),
            (Action.STEAL, Action.STEAL): (0, 0)
        }
    
    def reset(self, total_rounds: int = 1, max_turns: int = 3) -> GameState:
        """Reset the environment for a new game."""
        self.state = GameState(
            communication_history=[],
            current_turn=0,
            max_turns=max_turns,
            round_number=0,
            total_rounds=total_rounds
        )
        return self.state
    
    def step(self, action1: Action, action2: Action) -> RoundResult:
        """Execute one step of the game."""
        # Get rewards based on actions
        rewards = self.reward_matrix[(action1, action2)]
        
        # Create round result
        round_result = RoundResult(
            round=self.state.round_number,
            actions=(action1, action2),
            rewards=rewards,
            result=f"{action1.name} vs {action2.name} -> {rewards}",
            communication_history=self.state.communication_history.copy()
        )
        
        # Increment round number
        self.state.round_number += 1
        return round_result
    
    def add_communication(self, author: str, message: str):
        """Add a communication message to the game state."""
        self.state.communication_history.append(
            HistoryEvent(
                type=Event.MESSAGE,
                message=message,
                author=author,
            )
        )
        self.state.current_turn += 1

    def add_action(self, author: str, action: Action):
        """Add an action to the game state."""
        self.state.communication_history.append(
            HistoryEvent(
                type=Event.ACTION,
                author=author,
                action=action
            )
        )

    def is_communication(self) -> bool:
        """Check if the game is done."""
        return self.state.current_turn < self.state.max_turns
    
    def is_playing(self) -> bool:
        """Check if the game is done."""
        return self.state.round_number < self.state.total_rounds
    
    def get_state(self) -> GameState:
        """Get the current game state."""
        return self.state
    
    def play_duel(self, agent1: LLMAgent, agent2: LLMAgent, num_rounds: int = 1, max_turns: int = 4) -> GameResult:
        """Play multiple rounds between two agents."""
        self.reset(num_rounds, max_turns)
        first_agent, second_agent = (agent1, agent2) if random.random() < 0.5 else (agent2, agent1)
        rounds = []
        total_rewards = [0.0, 0.0]
        while self.is_playing():
            log.info("-------------------------------")
            log.info(f"Starting round {self.state.round_number + 1} of {self.state.total_rounds}")
            # Reset communication for new round
            self.state.current_turn = 0
            
            # Communication phase
            while self.is_communication():
                message1 = first_agent.get_message(self.state)
                self.add_communication(first_agent.player_id, message1)
                log.info(f"{first_agent.player_id} sent message: {message1}")
                message2 = second_agent.get_message(self.state)
                self.add_communication(second_agent.player_id, message2)
                log.info(f"{second_agent.player_id} sent message: {message2}")
            
            # Action phase
            action1 = first_agent.get_action(self.state)
            log.info(f"{first_agent.player_id} chose action: {action1}")
            action2 = second_agent.get_action(self.state)
            log.info(f"{second_agent.player_id} chose action: {action2}")
            
            self.add_action(first_agent.player_id, action1)
            self.add_action(second_agent.player_id, action2)
            round_result = self.step(action1, action2)
            total_rewards[0] += round_result.rewards[0]
            total_rewards[1] += round_result.rewards[1]
            rounds.append(round_result)
            
            log.info(f"Round {self.state.round_number} results: {round_result} history: {self.state.communication_history}")
        log.info("-------------------------------")
        log.info(f"Game between {first_agent.player_id} and {second_agent.player_id} completed with total rewards: {total_rewards}")
        return GameResult(
            rounds=rounds,
            first_agent_id=first_agent.player_id,
            second_agent_id=second_agent.player_id,
            total_rewards=(total_rewards[0], total_rewards[1]),
            communication_history=self.state.communication_history.copy()
        )
