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

def log_agent_performance(agent_id: str, performance: dict, log_directory: str):
    """Log individual agent performance in its own folder."""
    agent_folder = os.path.join(log_directory, agent_id)
    os.makedirs(agent_folder, exist_ok=True)
    log_file = os.path.join(agent_folder, "performance.log")
    with open(log_file, "a") as f:
        for key, value in performance.items():
            f.write(f"{key}: {value}\n")

def main():
    # Load environment variables
    load_dotenv()
    
    # Load configuration
    config = load_config("config.json")
    
    logging.basicConfig(filename=config["log_file"], level=config.get("log_level", logging.INFO))
    
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

    # play games
    result = game.simulate_games(
        env=env,
        agents=agents,
        num_rounds=config["num_rounds"]
    )
    game.evolve_agents(
        agents=agents,
        scores=result,
        a=config["evolution_factor"]
    )
    
    # Close wandb
    wandb.finish()

if __name__ == "__main__":
    main()
