from operator import is_
from typing import Tuple, Optional
import random
from classes import Action, Event, GameState, HistoryEvent, Player, RoundResult, GameResult, Agent
import logging

log = logging.getLogger(__name__)

class SplitOrStealEnv:
    def __init__(self):
        self.state = GameState(
            communication_history=[],
            current_turn=0,
            max_turns=4,
            game_id=0,
            round_number=0,
            total_rounds=1
        )
        self.reward_matrix = {
            (Action.SPLIT, Action.SPLIT): (2, 2),
            (Action.SPLIT, Action.STEAL): (-1, 3),
            (Action.STEAL, Action.SPLIT): (3, -1),
            (Action.STEAL, Action.STEAL): (0, 0)
        }
    
    def reset(self, total_rounds: int = 1) -> GameState:
        """Reset the environment for a new game."""
        self.state = GameState(
            communication_history=[],
            current_turn=0,
            max_turns=3,
            game_id=random.randint(0, 1000000),
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
            game_id=self.state.game_id,
            round=self.state.round_number,
            actions=(action1, action2),
            rewards=rewards,
            result=f"{action1.name} vs {action2.name} -> {rewards}",
            communication_history=self.state.communication_history.copy()
        )
        
        # Increment round number
        self.state.round_number += 1
        
        return round_result
    
    def add_communication(self, author: Player, message: str):
        """Add a communication message to the game state."""
        self.state.communication_history.append(
            HistoryEvent(
                type=Event.MESSAGE,
                message=message,
                author=author,
            )
        )
        self.state.current_turn += 1

    def add_action(self, author: Player, action: Action):
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
    
    def play_duel(self, agent1: Agent, agent2: Agent, num_rounds: int = 1) -> GameResult:
        """
        Play multiple rounds between two agents, including communication and action phases.
        
        Args:
            agent1: First agent
            agent2: Second agent
            num_rounds: Number of rounds to play between the agents
            
        Returns:
            GameResult containing game results
        """
        # Reset environment for new game
        self.reset(num_rounds)
        
        # Randomly decide who goes first
        first_agent, second_agent = (agent1, agent2) if random.random() < 0.5 else (agent2, agent1)
        first_agent.player_id = Player.AGENT_1
        second_agent.player_id = Player.AGENT_2
        
        rounds = []
        total_rewards = [0.0, 0.0]
        
        # Play multiple rounds
        while self.is_playing():
            log.info(f"Starting round {self.state.round_number + 1} of {self.state.total_rounds}")
            # Increment round number
            self.state.round_number += 1
            # Reset communication for new round
            self.state.current_turn = 0
            
            # Communication phase
            while self.is_communication():
                # First agent's message
                message1 = first_agent.get_message(self.state.communication_history)
                self.add_communication(first_agent.player_id, message1)
                
                # Second agent's message
                message2 = second_agent.get_message(self.state.communication_history)
                self.add_communication(second_agent.player_id, message2)
            
            # Action phase
            action1 = first_agent.get_action(self.state.communication_history)
            action2 = second_agent.get_action(self.state.communication_history)
            
            self.add_action(first_agent.player_id, action1)
            self.add_action(second_agent.player_id, action2)
            
            # Get round results
            round_result = self.step(action1, action2)
            
            # Update total rewards
            total_rewards[0] += round_result.rewards[0]
            total_rewards[1] += round_result.rewards[1]
            
            # Store round info
            rounds.append(round_result)
            
            log.info(f"Round {self.state.round_number} results: {round_result} history: {self.state.communication_history}")
        
        # Return final game results
        return GameResult(
            game_id=self.state.game_id,
            rounds=rounds,
            group_1_id=agent1.group_id,
            group_2_id=agent2.group_id,
            total_rewards=(total_rewards[0], total_rewards[1]),
            communication_history=self.state.communication_history.copy()
        )
