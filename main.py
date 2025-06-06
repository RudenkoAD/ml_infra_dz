import os
import random
import json
from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb
from tqdm import tqdm
import game
from environment import SplitOrStealEnv
from models.provider_finder import get_provider

import logging
log = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from a JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)

def main():
    # Load environment variables
    load_dotenv()
    
    # Load configuration
    config = load_config("config.json")
    
    #copy config to log folder
    if not os.path.exists(f"experiments/{config['experiment_name']}"):
        os.makedirs(f"experiments/{config['experiment_name']}")
    with open(f"experiments/{config['experiment_name']}/config.json", "w") as f:
        new_config = config.copy()
        new_config["api_key"] = ""
        json.dump(new_config, f, indent=4)
    
    logging.basicConfig(filename=f"experiments/{config["experiment_name"]}/{config["experiment_name"]}.log", level=config.get("log_level", logging.INFO))
    
    
    # Initialize wandb
    wandb.init(
        project="split-or-steal-llm",
        config={
            "model_name": "nilq/mistral-1L-tiny",
            "temperature": 0.7,
            "games_per_round": 10,
            "num_rounds": 10,
            "communication_turns": 3
        }
    )
    
    hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
    if not hf_token:
        raise ValueError("HUGGINGFACE_API_TOKEN environment variable is not set.")
    
    # create provider
    provider = get_provider(
        provider=config["provider"],
        api_key=config["api_key"],
        model_name=config["model_name"]
    )
    
    #create agents
    agents = game.create_agents(
        provider=provider,
        agent_personalities=config["agent_personalities"],
        agent_names=config["agent_names"]
    )
    
    # Initialize environment
    env = SplitOrStealEnv()
    env.reset(total_rounds=config["num_rounds"])
    log.info("Environment initialized with total rounds: %d", config["num_rounds"])

    for _ in tqdm(range(config["num_games"])):
        # Reset environment for each game
        env.reset(total_rounds=config["num_rounds"])
        
        # Simulate games
        result = game.simulate_games(
            env=env,
            agents=agents,
            num_rounds=config["num_rounds"],
            max_turns=config["max_turns"]
        )
        
        # Log results
        wandb.log({"game_results": result})
        
        # Log agent performance
        for agent in agents:
            performance = {
                "player_id": agent.player_id,
                "score": result[agent.player_id],
                "personality": agent.personality.name
            }

        # Evolve agents based on scores
        agents = game.evolve_agents(
            agents=agents,
            scores=result,
            a=config["evolution_factor"]
        )
        
    # Close wandb
    wandb.finish()

if __name__ == "__main__":
    main()
