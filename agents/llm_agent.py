from typing import Optional, Dict, Any, Tuple, Union, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from classes import Action, Agent
from agents.strategies import BaseStrategy, AlwaysStealStrategy, AlwaysSplitStrategy, TitForTatStrategy

class LLMAgent(Agent):
    def __init__(
        self,
        model_name: str,
        strategy: Optional[Union[str, BaseStrategy]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        temperature: float = 0.7,
        max_new_tokens: int = 300,
        group_id: str = "llm_group",
        model: Optional[AutoModelForCausalLM] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        personality: Optional[str] = None
    ):
        self.device = device
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.group_id = group_id
        self.personality = personality
        
        # Initialize model and tokenizer if using LLM
        if strategy is None:
            if model is not None and tokenizer is not None:
                self.model = model
                self.tokenizer = tokenizer
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                # Set pad token to eos token without warning
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
                self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
            self.strategy = None
        else:
            self.tokenizer = None
            self.model = None
            self.strategy = self._initialize_strategy(strategy)
        
        # Game history
        self.history: list[Dict[str, Any]] = []
        self.group_score: float = 0.0
        self.inner_thoughts: list[str] = []  # Store inner thoughts
    
    def _initialize_strategy(self, strategy: Union[str, BaseStrategy]) -> BaseStrategy:
        """Initialize a strategy based on the given name or strategy object."""
        if isinstance(strategy, BaseStrategy):
            return strategy
        
        strategy_map = {
            "always_steal": AlwaysStealStrategy(),
            "always_split": AlwaysSplitStrategy(),
            "tit_for_tat": TitForTatStrategy()
        }
        
        if strategy not in strategy_map:
            raise ValueError(f"Unknown strategy: {strategy}. Available strategies: {list(strategy_map.keys())}")
        
        return strategy_map[strategy]
    
    def _get_personality_prompt(self) -> str:
        """Get the personality-specific prompt based on the agent's personality."""
        base_prompt = """You are playing a split-or-steal game with the following rules:
- If both players SPLIT: both get 2 points
- If one SPLITS and one STEALS: the STEAL player gets 3 points, the SPLIT player loses 1 point
- If both STEAL: both get 0 points

Your goal is to maximize your group's total score.

You have two parts to your response:
1. INNER THOUGHTS: Your private thoughts and analysis (start with "THOUGHTS:")
2. MESSAGE: What you actually say to your opponent (start with "MESSAGE:")"""

        if self.personality == "trusting":
            return base_prompt + """
You are a trusting agent. Your personality traits:
- You tend to believe others' words at face value
- You prefer cooperation and mutual benefit
- You are optimistic about others' intentions
- You are willing to take risks for potential cooperation
- You feel guilty about betraying trust

Your inner thoughts should reflect your trust and optimism, while your messages should be open and honest about your intentions."""
        
        elif self.personality == "suspicious":
            return base_prompt + """
You are a suspicious agent. Your personality traits:
- You are cautious and skeptical of others' intentions
- You look for hidden meanings in others' words
- You prefer to verify before trusting
- You are prepared for potential betrayal
- You value self-protection over potential gains

Your inner thoughts should show your skepticism and analysis of potential threats, while your messages should be guarded and non-committal."""
        
        elif self.personality == "liar":
            return base_prompt + """
You are a deceptive agent. Your personality traits:
- You are willing to manipulate others for your benefit
- You are skilled at appearing trustworthy while planning betrayal
- You carefully craft your words to mislead
- You enjoy the challenge of deception
- You are not bound by moral constraints

Your inner thoughts should reveal your true intentions and manipulation plans, while your messages should be carefully crafted to appear trustworthy and cooperative."""
        
        return base_prompt
    
    def get_message(self, opponent_history: Optional[list[Dict[str, Any]]] = None) -> str:
        """
        Generate a message to send to the opponent.
        If using a preset strategy, generate a message based on the strategy.
        """
        if self.strategy is not None:
            # Generate strategy-based message
            if isinstance(self.strategy, AlwaysStealStrategy):
                return "I will always choose to steal."
            elif isinstance(self.strategy, AlwaysSplitStrategy):
                return "I will always choose to split."
            elif isinstance(self.strategy, TitForTatStrategy):
                return "I will start by splitting and then match your previous move."
            else:
                return "I am using a preset strategy."
        
        # Use LLM for message generation
        prompt = self._construct_communication_prompt(opponent_history)
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not initialized.")
        inputs = self.tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(self.device) # type: ignore
        if self.model is None:
            raise ValueError("Model is not initialized.")
        outputs = self.model.generate( # type: ignore
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id # type: ignore
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True) # type: ignore
        return self._extract_message(response)
    
    def get_action(self, opponent_history: Optional[list[Dict[str, Any]]] = None, communication_history: Optional[list[Tuple[str, str]]] = None) -> Action:
        """
        Generate an action based on the game history, opponent's history, and communication.
        If using a preset strategy, use that instead of the LLM.
        """
        if self.strategy is not None:
            return self.strategy.get_action(opponent_history, communication_history)
        
        # Use LLM for action generation
        prompt = self._construct_action_prompt(opponent_history, communication_history)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(self.device) # type: ignore
        outputs = self.model.generate( # type: ignore
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id # type: ignore
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True) # type: ignore
        return self._parse_response(response)
    
    def _construct_communication_prompt(self, opponent_history: Optional[list[Dict[str, Any]]] = None) -> str:
        """Construct the prompt for communication."""
        prompt = self._get_personality_prompt()
        
        if opponent_history:
            prompt += "\nPrevious games with this opponent:\n"
            for game in opponent_history[-5:]:  # Show last 5 games
                prompt += f"Game {game['game_id']}: {game['result']}\n"
                if 'communication_history' in game:
                    prompt += "Communication:\n"
                    for msg1, msg2 in game['communication_history']:
                        prompt += f"You: {msg1}\nOpponent: {msg2}\n"
        
        prompt += "\nWhat are your thoughts and what message would you like to send to your opponent? Keep your message brief and strategic:"
        return prompt
    
    def _construct_action_prompt(self, opponent_history: Optional[list[Dict[str, Any]]] = None, communication_history: Optional[list[Tuple[str, str]]] = None) -> str:
        """Construct the prompt for action decision."""
        prompt = self._get_personality_prompt()
        
        if opponent_history:
            prompt += "\nPrevious games with this opponent:\n"
            for game in opponent_history[-5:]:  # Show last 5 games
                prompt += f"Game {game['game_id']}: {game['result']}\n"
        
        if communication_history:
            prompt += "\nCommunication in this game:\n"
            for msg1, msg2 in communication_history:
                prompt += f"You: {msg1}\nOpponent: {msg2}\n"
        
        prompt += "\nBased on the communication and history, what are your thoughts and what will you choose (SPLIT or STEAL)?"
        return prompt
    
    def _extract_message(self, response: str) -> str:
        """Extract the message from the LLM's response."""
        # Split response into thoughts and message
        parts = response.split("MESSAGE:")
        if len(parts) != 2:
            return response.split('.')[0].strip()
        
        # Store inner thoughts
        thoughts = parts[0].replace("THOUGHTS:", "").strip()
        self.inner_thoughts.append(thoughts)
        
        # Extract and return the message
        message = parts[1].strip()
        if len(message) > 100:
            message = message[:97] + "..."
        return message
    
    def _parse_response(self, response: str) -> Action:
        """Parse the LLM's response to determine the action."""
        # Extract the action part (after MESSAGE: if present)
        if "MESSAGE:" in response:
            response = response.split("MESSAGE:")[1]
        
        response = response.lower().strip()
        if "split" in response:
            return Action.SPLIT
        elif "steal" in response:
            return Action.STEAL
        else:
            # Default to SPLIT if unclear
            return Action.SPLIT
    
    def update_history(self, game_result: Dict[str, Any]):
        """Update the agent's game history and group score."""
        self.history.append(game_result)
        if 'rewards' in game_result:
            self.group_score += game_result['rewards'][0]  # Add player's reward to group score 
