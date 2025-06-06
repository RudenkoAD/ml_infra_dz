from collections import defaultdict
from copy import deepcopy
import os
import json
import random
from dotenv import load_dotenv
import wandb
from tqdm import tqdm
import game
from environment import SplitOrStealEnv

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
    config_to_log = deepcopy(config)
    for key in config_to_log["api_keys"].keys():
        config_to_log["api_keys"][key] = f"{key}_api_key"
    
    #copy config to log folder
    if not os.path.exists(f"experiments/{config['experiment_name']}"):
        os.makedirs(f"experiments/{config['experiment_name']}")
    with open(f"experiments/{config['experiment_name']}/config.json", "w") as f:
        json.dump(config_to_log, f, indent=4)
    logging.basicConfig(filename=f"experiments/{config['experiment_name']}/{config['experiment_name']}.log", level=config.get("log_level", logging.INFO))
    
    # Initialize wandb
    wandb.init(
        project="split-or-steal-llm",
        config=config_to_log,
        name=config["experiment_name"]
    )
    
    #create agents
    agents = game.create_agents(
        config["agents"],
        api_keys=config["api_keys"]
    )
    
    # Initialize environment
    env = SplitOrStealEnv()
    env.reset(total_rounds=config["num_rounds"])
    log.info("Environment initialized with total rounds: %d", config["num_rounds"])

    for _ in tqdm(range(config["num_games"])):
        # Reset environment for each game
        env.reset(total_rounds=config["num_rounds"])
        
        # Simulate games
        result, table = game.simulate_games(
            env=env,
            agents=agents,
            num_rounds=config["num_rounds"],
            max_turns=config["max_turns"]
        )
        
        # Log results
        dict_to_log = {
        "game_results": result,
        "games_table": table,
        }
        amounts_of_sets = defaultdict(int)
        for agent in agents:
            amounts_of_sets[f"amount_of_agents_with_{agent.promptset.name}"] +=1
        
        dict_to_log.update(amounts_of_sets)
        wandb.log(dict_to_log)

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
