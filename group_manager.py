from typing import List, Dict, Any, Optional, Tuple, Union, Callable
import random
from environment import Action, SplitOrStealEnv
from agents.llm_agent import LLMAgent
from agents.rl_agent import RLAgent
from classes import GameResult, Player
import logging
log = logging.getLogger(__name__)

class AgentGroup:
    def __init__(self, group_id: str, agents: List[Union[LLMAgent, RLAgent]]):
        self.group_id = group_id
        self.agents = agents
        self.total_score = 0.0
    
    def update_score(self, score: float):
        """Update the group's total score."""
        self.total_score += score
    
    def get_random_agent(self) -> Union[LLMAgent, RLAgent]:
        """Get a random agent from the group."""
        return random.choice(self.agents)

class GroupManager:
    def __init__(self):
        self.groups: Dict[str, List] = {}
        self.env = SplitOrStealEnv()
    
    def create_group(self, group_id: str, agents: List):
        """Create a new group of agents."""
        self.groups[group_id] = agents
        log.info(f"Group {group_id} created")
    
    def play_self_play(self, group_id: str, num_games: int = 10, num_rounds: int = 3) -> float:
        """Play games between agents in the same group."""
        if group_id not in self.groups:
            raise ValueError(f"Group {group_id} does not exist")
        
        agents = self.groups[group_id]
        total_score = 0.0
        
        for _ in range(num_games):
            # Randomly select two different agents from the group
            agent1, agent2 = random.sample(agents, 2)
            agent1.player_id = Player.AGENT_1
            agent2.player_id = Player.AGENT_2
            # Play a game with multiple rounds
            game_result = self.env.play_duel(agent1, agent2, num_rounds)
            total_score += game_result.total_rewards[0]  # Use first agent's total reward
            # log the game result including communication history
            agent1.group_score += game_result.total_rewards[0]
            agent2.group_score += game_result.total_rewards[1]
            log.info(f"Game between {group_id}, {group_id}\nGame Result: {game_result.total_rewards}\nCommunication History: {game_result.communication_history}")
        return total_score / num_games
    
    def play_group_vs_group(self, group1_id: str, group2_id: str, num_games: int = 10, num_rounds: int = 3) -> Tuple[float, float]:
        """Play games between agents from different groups."""
        if group1_id not in self.groups or group2_id not in self.groups:
            raise ValueError(f"One or both groups do not exist")
        
        group1_agents = self.groups[group1_id]
        group2_agents = self.groups[group2_id]
        total_score1 = 0.0
        total_score2 = 0.0
        
        for _ in range(num_games):
            # Randomly select one agent from each group
            agent1 = random.choice(group1_agents)
            agent2 = random.choice(group2_agents)
            
            # Play a game with multiple rounds
            game_result = self.env.play_duel(agent1, agent2, num_rounds)
            total_score1 += game_result.total_rewards[0]  # First agent's total reward
            total_score2 += game_result.total_rewards[1]  # Second agent's total reward
            
            log.info(f"Game between {group1_id}, {group2_id}\nGame Result: {game_result.total_rewards}\nCommunication History: {game_result.communication_history}")
        return total_score1 / num_games, total_score2 / num_games
    
    def get_group_rankings(self) -> List[Tuple[str, float]]:
        """Get rankings of all groups based on their average scores."""
        rankings = []
        for group_id, agents in self.groups.items():
            avg_score = sum(agent.group_score for agent in agents) / len(agents)
            rankings.append((group_id, avg_score))
        
        # Sort by score in descending order
        return sorted(rankings, key=lambda x: x[1], reverse=True) 
