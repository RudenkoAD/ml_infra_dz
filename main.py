import os
import random
from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb
from tqdm import tqdm
from agents.llm_agent import LLMAgent
from group_manager import GroupManager
from environment import SplitOrStealEnv
import logging
logging.basicConfig(filename=f'myapp.log', level=logging.INFO)
log = logging.getLogger(__name__)

def create_group_manager():
    """Create different groups of agents for training and evaluation."""
    group_manager = GroupManager()
    
    # Create shared model instance
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0").to(device)
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    
    # Create LLM agent groups with different personalities
    llm_groups = {
        "llm_trusting": [],    # Agents that tend to trust others
        "llm_suspicious": [],  # Agents that are cautious and skeptical
        "llm_liar": []         # Agents that are willing to deceive
    }
    
    # Initialize LLM agents with different personalities
    for group_id in llm_groups:
        for _ in range(3):  # 3 agents per group
            if group_id == "llm_trusting":
                agent = LLMAgent(
                    model_name="nilq/mistral-1L-tiny",
                    group_id=group_id,
                    temperature=0.7,
                    model=model,
                    tokenizer=tokenizer,
                    personality="trusting"
                )
            elif group_id == "llm_suspicious":
                agent = LLMAgent(
                    model_name="nilq/mistral-1L-tiny",
                    group_id=group_id,
                    temperature=0.7,
                    model=model,
                    tokenizer=tokenizer,
                    personality="suspicious"
                )
            else:  # llm_liar
                agent = LLMAgent(
                    model_name="nilq/mistral-1L-tiny",
                    group_id=group_id,
                    temperature=0.7,
                    model=model,
                    tokenizer=tokenizer,
                    personality="liar"
                )
            llm_groups[group_id].append(agent)
        group_manager.create_group(group_id, llm_groups[group_id])
    
    return group_manager

def play(group_manager: GroupManager, num_games: int, num_rounds: int):
    """Run a single training round."""
    # Play self-play games within each group
    for group_id in tqdm(group_manager.groups, desc="Self-play games"):
        self_play_results = group_manager.play_self_play(group_id, num_games=num_games, num_rounds=num_rounds)
        wandb.log({
            f"{group_id}/self_play_game_score": self_play_results,
        })
        log.info(f"{group_id}, Score: {self_play_results}")
    # Play inter-group games
    groups = list(group_manager.groups.keys())
    for i, group1_id in enumerate(tqdm(groups, desc="Inter-group games")):
        for group2_id in groups[i+1:]:
            score1, score2 = group_manager.play_group_vs_group(
                group1_id, group2_id, num_games=num_games, num_rounds=num_rounds
            )
            wandb.log({
                f"{group1_id}_vs_{group2_id}_score1": score1,
                f"{group1_id}_vs_{group2_id}_score2": score2,
            })
            log.info(f"{group1_id} vs {group2_id}, Score1: {score1}, Score2: {score2}")
            
    # Log group rankings
    rankings = group_manager.get_group_rankings()
    for rank, (group_id, score) in enumerate(rankings):
        wandb.log({
            f"rankings/{group_id}": rank + 1,
            f"scores/{group_id}": score,
        })
        log.info(f"Rankings: {rank + 1}. {group_id}, Score: {score}")

def main():
    # Load environment variables
    load_dotenv()
    
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
    
    # Create agent groups
    group_manager = create_group_manager()
    # Training loop
    play(group_manager, 1, 3)
    
    # Print current rankings
    rankings = group_manager.get_group_rankings()
    log.info("\nCurrent group rankings:")
    for rank, (group_id, score) in enumerate(rankings):
        log.info(f"{rank + 1}. {group_id}: {score:.2f}")
    
    # Close wandb
    wandb.finish()

if __name__ == "__main__":
    main()
