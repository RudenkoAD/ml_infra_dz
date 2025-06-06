from typing import List, Tuple
import wandb
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
            personality=Personality(agent_personalities[i])
        )
        agents.append(agent)
    log.info(f"Created {len(agents)} agents with personalities: {[agent.personality for agent in agents]}")
    return agents

def simulate_games(env: SplitOrStealEnv, agents: List[LLMAgent], num_rounds: int = 3, max_turns: int = 4) -> dict[str, float]:
    """Simulate games between all agents."""
    scores = {agent.player_id: 0.0 for agent in agents}
    for i, agent1 in enumerate(agents):
        for j, agent2 in enumerate(agents[i+1:]):
            game_result = env.play_duel(agent1, agent2, num_rounds, max_turns=max_turns)
            scores[game_result.first_agent_id] += game_result.total_rewards[0]
            scores[game_result.second_agent_id] += game_result.total_rewards[1]
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
