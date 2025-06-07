from typing import List, Tuple
from agents.llm_agent import LLMAgent
from classes import GameResult
from environment import SplitOrStealEnv

from logging import getLogger
from models.provider_finder import get_provider
from promptsets.promptset_finder import get_promptset

log = getLogger(__name__)

def create_agents(agents: List[dict[str, str]], api_keys: dict[str, str]) -> List[LLMAgent]:
    """Create a list of agents."""
    created_agents: List[LLMAgent] = []
    for i in range(len(agents)):
        provider_name = agents[i]["provider"]
        api_key = api_keys.get(provider_name)
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

def simulate_games(env: SplitOrStealEnv, agents: List[LLMAgent], num_rounds: int = 3, max_turns: int = 4) -> Tuple[dict[str, float], list[GameResult]]:
    """Simulate games between all agents."""
    scores = {agent.player_id: 0.0 for agent in agents}
    results: list[GameResult] = []
    for i, agent1 in enumerate(agents):
        for agent2 in agents[i+1:]:
            game_result = env.play_duel(agent1, agent2, num_rounds, max_turns=max_turns)
            results.append(game_result)
            scores[game_result.first_agent_id] += game_result.total_rewards[0]
            scores[game_result.second_agent_id] += game_result.total_rewards[1]
    return scores, results

def evolve_agents(agents: List[LLMAgent], scores: dict[str, float], a: int) -> List[LLMAgent]:
    """Delete the worst-performing agents and duplicate the best-performing agents."""
    sorted_agents = sorted(agents, key=lambda x: scores[x.player_id], reverse=True)
    log.info("Evolving agents")
    log.info(f"Agents sorted by score: {[agent.player_id for agent in sorted_agents]}")
    top_agents = sorted_agents[:a]
    new_agents: List[LLMAgent] = []
    for agent in top_agents:
        new_agent = agent.clone()
        new_agents.append(new_agent)
    average_agents = sorted_agents[a:-a]
    return top_agents + new_agents + average_agents
