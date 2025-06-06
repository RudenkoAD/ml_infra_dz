import random
from typing import List, Tuple
import wandb
from numpy import average
from agents.llm_agent import LLMAgent
from environment import SplitOrStealEnv
from classes import Personality

from logging import getLogger
from models.providers.base_provider import Provider
log = getLogger(__name__)

def create_agents(provider: Provider, agent_personalities: list[Personality], agent_names: list[str]) -> List:
    """Create a list of agents."""
    agents = []
    for i in range(len(agent_names)):
        agent = LLMAgent(
            player_id=agent_names[i],
            provider=provider,
            personality=agent_personalities[i]
        )
        agents.append(agent)
    log.info(f"Created {len(agents)} agents with personalities: {[agent.personality for agent in agents]}")
    return agents

def simulate_games(env: SplitOrStealEnv, agents: List[LLMAgent], num_rounds: int = 3) -> dict[str, float]:
    """Simulate games between all agents."""
    scores = {agent.player_id: 0.0 for agent in agents}
    results = [[0.0 for _ in agents] for _ in agents]
    for i, agent1 in enumerate(agents):
        for j, agent2 in enumerate(agents[i+1:]):
            game_result = env.play_duel(agent1, agent2, num_rounds)
            scores[agent1.player_id] += game_result.total_rewards[0]
            scores[agent2.player_id] += game_result.total_rewards[1]
            results[i][i+1+j] += game_result.total_rewards[0]
            results[i+1+j][i] += game_result.total_rewards[1]
    wandb.log({"game_results": results})
    wandb.log({"round_scores": scores})
    return scores

def evolve_agents(agents: List, scores: dict[str, float], a: int) -> List:
    """Delete the worst-performing agents and duplicate the best-performing agents."""
    sorted_agents = sorted(agents, key=lambda x: scores[x.player_id], reverse=True)
    log.info(f"Agents sorted by score: {[agent.player_id for agent in sorted_agents]}")
    top_agents = sorted_agents[:a]
    new_agents = []
    for agent in top_agents:
        new_agent = agent.clone()
        new_agents.append(new_agent)
    average_agents = sorted_agents[a:-a]
    wandb.log({
        "amount_of_agents": len(agents),
        "amount_of_trusting_agents": len([agent for agent in agents if agent.personality == Personality.TRUSTING]),
        "amount_of_suspicious_agents": len([agent for agent in agents if agent.personality == Personality.SUSPICIOUS]),
        "amount_of_lying_agents": len([agent for agent in agents if agent.personality == Personality.LIAR]),
    })
    return top_agents + new_agents + average_agents
