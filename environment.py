from typing import Tuple, Optional
import random
from classes import Action, GameState, RoundResult, GameResult, Agent

class SplitOrStealEnv:
    def __init__(self):
        self.state = GameState(
            communication_history=[],
            current_turn=0,
            max_turns=3,
            game_id=0,
            done=False,
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
            done=False,
            round_number=0,
            total_rounds=total_rounds
        )
        return self.state
    
    def step(self, action1: Action, action2: Action) -> Tuple[Tuple[float, float], bool, RoundResult]:
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
        
        # Check if game is done
        self.state.done = self.state.round_number >= self.state.total_rounds
        
        return rewards, self.state.done, round_result
    
    def add_communication(self, message1: str, message2: Optional[str] = None) -> None:
        """Add communication messages to the game state."""
        if message2 is None:
            self.state.communication_history.append((message1, ""))
        else:
            if len(self.state.communication_history) > 0 and self.state.communication_history[-1][1] is None:
                self.state.communication_history[-1] = (message1, message2)
            else:
                self.state.communication_history.append((message1, message2))
        
        self.state.current_turn += 1
    
    def is_done(self) -> bool:
        """Check if the game is done."""
        return self.state.done or self.state.current_turn >= self.state.max_turns
    
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
        
        rounds = []
        total_rewards = [0.0, 0.0]
        
        # Play multiple rounds
        while not self.state.done:
            # Reset communication for new round
            self.state.communication_history = []
            self.state.current_turn = 0
            
            # Communication phase
            while not self.is_done():
                # First agent's message
                message1 = first_agent.get_message(second_agent.history)
                self.add_communication(message1)
                
                # Second agent's message
                message2 = second_agent.get_message(first_agent.history)
                self.add_communication(message1, message2)
            
            # Action phase
            action1 = first_agent.get_action(first_agent.history, self.state.communication_history)
            action2 = second_agent.get_action(second_agent.history, self.state.communication_history)
            
            # Get round results
            rewards, done, round_result = self.step(action1, action2)
            
            # Update total rewards
            total_rewards[0] += rewards[0]
            total_rewards[1] += rewards[1]
            
            # Store round info
            rounds.append(round_result)
            
            # Update agent histories
            game_result = {
                'game_id': round_result.game_id,
                'round': round_result.round,
                'opponent': second_agent.group_id,
                'actions': round_result.actions,
                'rewards': round_result.rewards,
                'result': round_result.result,
                'communication_history': round_result.communication_history
            }
            first_agent.update_history(game_result)
            second_agent.update_history(game_result)
        
        # Return final game results
        return GameResult(
            game_id=self.state.game_id,
            rounds=rounds,
            total_rewards=(total_rewards[0], total_rewards[1]),
            communication_history=self.state.communication_history.copy()
        )
