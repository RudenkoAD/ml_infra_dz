import os
import random
from dotenv import load_dotenv
import wandb
from tqdm import tqdm
from agents.llm_agent import LLMAgent
from group_manager import GroupManager
from environment import SplitOrStealEnv

def create_agent_groups():
    """Create different groups of agents for training and evaluation."""
    group_manager = GroupManager()
    
    # Create shared model instance
    llm_model = LLMAgent(
        model_name="nilq/mistral-1L-tiny",
        temperature=0.7,
    )
    llm_model.tokenizer.pad_token = llm_model.tokenizer.eos_token # type: ignore
    llm_model.tokenizer.pad_token_id = llm_model.tokenizer.eos_token_id # type: ignore
    llm_model.model.generation_config.pad_token_id = llm_model.tokenizer.eos_token_id # type: ignore
    
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
                    model=llm_model.model,
                    tokenizer=llm_model.tokenizer,
                    personality="trusting"
                )
            elif group_id == "llm_suspicious":
                agent = LLMAgent(
                    model_name="nilq/mistral-1L-tiny",
                    group_id=group_id,
                    temperature=0.7,
                    model=llm_model.model,
                    tokenizer=llm_model.tokenizer,
                    personality="suspicious"
                )
            else:  # llm_liar
                agent = LLMAgent(
                    model_name="nilq/mistral-1L-tiny",
                    group_id=group_id,
                    temperature=0.7,
                    model=llm_model.model,
                    tokenizer=llm_model.tokenizer,
                    personality="liar"
                )
            llm_groups[group_id].append(agent)
        group_manager.create_group(group_id, llm_groups[group_id])
    
    return group_manager, llm_groups, llm_model

def train_round(group_manager: GroupManager, round_num: int):
    """Run a single training round."""
    # Play self-play games within each group
    for group_id in tqdm(group_manager.groups, desc="Self-play games"):
        self_play_score = group_manager.play_self_play(group_id, num_games=10)
        wandb.log({
            f"{group_id}/self_play_score": self_play_score,
            "round": round_num
        })
    
    # Play inter-group games
    groups = list(group_manager.groups.keys())
    for i, group1_id in enumerate(tqdm(groups, desc="Inter-group games")):
        for group2_id in groups[i+1:]:
            score1, score2 = group_manager.play_group_vs_group(
                group1_id, group2_id, num_games=10
            )
            wandb.log({
                f"{group1_id}_vs_{group2_id}/score1": score1,
                f"{group1_id}_vs_{group2_id}/score2": score2,
                "round": round_num
            })
    
    # Log group rankings
    rankings = group_manager.get_group_rankings()
    for rank, (group_id, score) in enumerate(rankings):
        wandb.log({
            f"rankings/{group_id}": rank + 1,
            f"scores/{group_id}": score,
            "round": round_num
        })

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
    group_manager, llm_groups, llm_model = create_agent_groups()
    
    # Training loop
    num_rounds = 10
    for round_num in tqdm(range(num_rounds), desc="Training rounds"):
        print(f"\nStarting round {round_num + 1}/{num_rounds}")
        train_round(group_manager, round_num)
        
        # Print current rankings
        rankings = group_manager.get_group_rankings()
        print("\nCurrent group rankings:")
        for rank, (group_id, score) in enumerate(rankings):
            print(f"{rank + 1}. {group_id}: {score:.2f}")
    
    # Close wandb
    wandb.finish()

if __name__ == "__main__":
    main() 
