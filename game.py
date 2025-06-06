import random
from typing import List, Tuple

from numpy import average
from agents.hf_agent import HFAgent
from agents.llm_agent import LLMAgent
from agents.base_agent import Agent
from environment import SplitOrStealEnv
from classes import Personality

from logging import getLogger
log = getLogger(__name__)

def create_agents(num_agents: int, model_name: str, agent_personalities: list[Personality]) -> List:
    """Create a list of agents."""
    agents = []
    for i in range(num_agents):
        agent = HFAgent(
            model_name=model_name,
            personality=agent_personalities[i],
        )
        agents.append(agent)
    log.info(f"Created {len(agents)} agents with personalities: {[agent.personality for agent in agents]}")
    return agents

def simulate_games(env: SplitOrStealEnv, agents: List[Agent], num_rounds: int = 3) -> List[Tuple[str, float]]:
    """Simulate games between all agents."""
    scores = {agent.player_id: 0.0 for agent in agents}
    for i, agent1 in enumerate(agents):
        for agent2 in agents[i+1:]:
            game_result = env.play_duel(agent1, agent2, num_rounds)
            scores[agent1.player_id] += game_result.total_rewards[0]
            scores[agent2.player_id] += game_result.total_rewards[1]
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def evolve_agents(agents: List, scores: List[Tuple[str, float]], a: int) -> List:
    """Delete the worst-performing agents and duplicate the best-performing agents."""
    # Sort agents by score
    sorted_agents = sorted(agents, key=lambda agent: next(score for group_id, score in scores if group_id == agent.group_id), reverse=True)
    log.info(f"Top agents: {[agent.group_id for agent in sorted_agents[:a]]}")
    # Keep top-performing agents
    top_agents = sorted_agents[:a]
    
    # Duplicate top agents
    new_agents = []
    for agent in top_agents:
        new_agent = agent.clone()
        new_agents.append(new_agent)
    
    average_agents = sorted_agents[a:-a]
    log.info(f"Average agents: {[agent.group_id for agent in average_agents]}")
    log.info(f"Bad agents: {[agent.group_id for agent in sorted_agents[-a:]]}")
    return top_agents + new_agents + average_agents
