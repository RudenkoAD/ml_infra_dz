from typing import Optional, Dict, Any, Tuple, List
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from classes import Action
import numpy as np

class RLAgent:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        clip_ratio: float = 0.2,
        lora_rank: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        group_id: Optional[str] = None,
        model: Optional[AutoModelForCausalLM] = None,
        tokenizer: Optional[AutoTokenizer] = None
    ):
        self.device = device
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.group_id = group_id
        
        # Initialize LLM
        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Set pad token to eos token without warning
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
            
            # Configure LoRA
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "v_proj"]  # Only adapt attention layers
            )
            
            # Apply LoRA to the model
            self.model = get_peft_model(self.model, peft_config)
        
        # Initialize optimizer (only for LoRA parameters)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate) # type: ignore
        
        # Game history
        self.history: List[Dict[str, Any]] = []
        self.group_score: float = 0.0
        
        # Training buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad) # type: ignore
        total_params = sum(p.numel() for p in self.model.parameters()) # type: ignore
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total)")
    
    def get_message(self, opponent_history: Optional[List[Dict[str, Any]]] = None) -> str:
        """Generate a message using the LLM."""
        prompt = self._construct_communication_prompt(opponent_history)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(self.device) # type: ignore
        outputs = self.model.generate( # type: ignore
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id # type: ignore
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True) # type: ignore
        return self._extract_message(response)
    
    def get_action(self, opponent_history: Optional[List[Dict[str, Any]]] = None, communication_history: Optional[List[Tuple[str, str]]] = None) -> Action:
        """Get action using the LLM."""
        prompt = self._construct_action_prompt(opponent_history, communication_history)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(self.device) # type: ignore
        outputs = self.model(**inputs) # type: ignore
        logits = outputs.logits[:, -1, :]  # Get logits for the last token
        
        # Get logits for "SPLIT" and "STEAL" tokens
        split_token_id = self.tokenizer.encode("SPLIT")[0] # type: ignore
        steal_token_id = self.tokenizer.encode("STEAL")[0] # type: ignore
        action_logits = torch.tensor([logits[0, split_token_id], logits[0, steal_token_id]], device=self.device)
        
        # Sample action
        probs = torch.softmax(action_logits, dim=-1)
        action_idx = torch.multinomial(probs, 1).item()
        action = Action(action_idx)
        
        # Store for training
        self.states.append(prompt)
        self.actions.append(action_idx)
        self.log_probs.append(torch.log(probs[action_idx])) # type: ignore
        
        return action
    
    def update(self, reward: float):
        """Update the LLM using PPO."""
        # Store reward
        self.rewards.append(reward)
        self.group_score += reward  # Update group score
        
        # If we have enough samples, update the model
        if len(self.rewards) >= 10:  # Update every 10 games
            self._update_model()
    
    def _update_model(self):
        """Update the LLM using PPO algorithm."""
        # Convert to tensors
        states = self.states
        actions = torch.tensor(self.actions, device=self.device)
        old_log_probs = torch.stack(self.log_probs)
        rewards = torch.tensor(self.rewards, device=self.device)
        
        # Calculate returns
        returns = self._calculate_returns(rewards)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # PPO update
        for _ in range(10):  # Multiple epochs
            # Get new action probabilities
            new_log_probs = []
            for state in states:
                inputs = self.tokenizer(state, return_tensors="pt", return_token_type_ids=False).to(self.device) # type: ignore
                outputs = self.model(**inputs) # type: ignore
                logits = outputs.logits[:, -1, :]  # Get logits for the last token
                
                # Get logits for "SPLIT" and "STEAL" tokens
                split_token_id = self.tokenizer.encode("SPLIT")[0] # type: ignore
                steal_token_id = self.tokenizer.encode("STEAL")[0] # type: ignore
                action_logits = torch.tensor([logits[0, split_token_id], logits[0, steal_token_id]], device=self.device, requires_grad=True)
                
                probs = torch.softmax(action_logits, dim=-1)
                new_log_probs.append(torch.log(probs[actions[len(new_log_probs)]]))
            
            new_log_probs = torch.stack(new_log_probs)
            
            # Calculate ratio
            ratio = torch.exp(new_log_probs - old_log_probs.detach())
            
            # Calculate PPO loss
            surr1 = ratio * returns
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * returns
            loss = -torch.min(surr1, surr2).mean()
            
            # Update model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Clear buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
    
    def _calculate_returns(self, rewards: torch.Tensor) -> torch.Tensor:
        """Calculate discounted returns."""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, device=self.device)
    
    def _construct_communication_prompt(self, opponent_history: Optional[List[Dict[str, Any]]] = None) -> str:
        """Construct the prompt for communication."""
        prompt = """You are playing a split-or-steal game with the following rules:
- If both players SPLIT: both get 2 points
- If one SPLITS and one STEALS: the STEAL player gets 3 points, the SPLIT player loses 1 point
- If both STEAL: both get 0 points

You can communicate with your opponent before making your decision. Your goal is to maximize your group's total score."""
        
        if opponent_history:
            prompt += "\nPrevious games with this opponent:\n"
            for game in opponent_history[-5:]:
                prompt += f"Game {game['game_id']}: {game['result']}\n"
                if 'communication_history' in game:
                    prompt += "Communication:\n"
                    for msg1, msg2 in game['communication_history']:
                        prompt += f"You: {msg1}\nOpponent: {msg2}\n"
        
        prompt += "\nWhat message would you like to send to your opponent? Keep it brief and strategic:"
        return prompt
    
    def _construct_action_prompt(self, opponent_history: Optional[List[Dict[str, Any]]] = None, communication_history: Optional[List[Tuple[str, str]]] = None) -> str:
        """Construct the prompt for action selection."""
        prompt = """You are playing a split-or-steal game with the following rules:
- If both players SPLIT: both get 2 points
- If one SPLITS and one STEALS: the STEAL player gets 3 points, the SPLIT player loses 1 point
- If both STEAL: both get 0 points

You must choose to either SPLIT or STEAL."""
        
        if opponent_history:
            prompt += "\nPrevious games with this opponent:\n"
            for game in opponent_history[-5:]:
                prompt += f"Game {game['game_id']}: {game['result']}\n"
                if 'communication_history' in game:
                    prompt += "Communication:\n"
                    for msg1, msg2 in game['communication_history']:
                        prompt += f"You: {msg1}\nOpponent: {msg2}\n"
        
        if communication_history:
            prompt += "\nCurrent game communication:\n"
            for msg1, msg2 in communication_history:
                prompt += f"You: {msg1}\nOpponent: {msg2}\n"
        
        prompt += "\nBased on the above information, choose either SPLIT or STEAL:"
        return prompt
    
    def _extract_message(self, response: str) -> str:
        """Extract the message from the LLM's response."""
        message = response.split('.')[0].strip()
        if len(message) > 100:
            message = message[:97] + "..."
        return message
    
    def update_history(self, game_result: Dict[str, Any]):
        """Update the agent's game history."""
        self.history.append(game_result) 
