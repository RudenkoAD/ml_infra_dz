import dataclasses
from typing import List, Tuple
import wandb
from agents.llm_agent import LLMAgent
from environment import SplitOrStealEnv

from logging import getLogger
from models.provider_finder import get_provider
from models.providers.base_provider import Provider
from promptsets.promptset_finder import get_promptset

log = getLogger(__name__)

def create_agents(agents: List[dict], api_keys: dict) -> List[LLMAgent]:
    """Create a list of agents."""
    created_agents: List[LLMAgent] = []
    for i in range(len(agents)):
        provider_name = agents[i]["provider"]
        api_key = api_keys[provider_name]
        if api_key is None:
            raise ValueError(f"api_key for provider {provider_name} not found")
        agent = LLMAgent(
            player_id=agents[i]["name"],
            provider=get_provider(
                provider=provider_name,
                api_key=api_key,
                model_name=agents[i]["model"]
                ),
            promptset=get_promptset(agents[i]["promptset"])
        )
        created_agents.append(agent)
    log.info(f"Created {len(agents)} agents")
    return created_agents

def simulate_games(env: SplitOrStealEnv, agents: List[LLMAgent], num_rounds: int = 3, max_turns: int = 4) -> Tuple[dict[str, float], wandb.Table]:
    """Simulate games between all agents."""
    scores = {agent.player_id: 0.0 for agent in agents}
    table = wandb.Table(["player_1", "player_2", "result", "history"])
    for i, agent1 in enumerate(agents):
        for j, agent2 in enumerate(agents[i+1:]):
            game_result = env.play_duel(agent1, agent2, num_rounds, max_turns=max_turns)
            scores[game_result.first_agent_id] += game_result.total_rewards[0]
            scores[game_result.second_agent_id] += game_result.total_rewards[1]
            table.add_data(
                game_result.first_agent_id, 
                game_result.second_agent_id, 
                f"{game_result.total_rewards[0]} - {game_result.total_rewards[1]}",
                "\n".join([str(event) for event in game_result.communication_history])
                )
    return scores, table

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
    return top_agents + new_agents + average_agents
