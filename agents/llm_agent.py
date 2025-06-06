from typing import Optional, Dict, Any
from urllib import response
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from agents.prompts import Personality, PromptManager
from agents.base_agent import Agent
import logging
log = logging.getLogger(__name__)
class LLMAgent(Agent):
    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        temperature: float = 0.7,
        max_new_tokens: int = 1024,
        group_id: str = "llm_group",
        personality: Personality = Personality.TRUSTING,
        model: Optional[AutoModelForCausalLM] = None,
        tokenizer: Optional[AutoTokenizer] = None,
    ):
        self.device = device
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.group_id = group_id
        self.personality = personality
        
        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
    
    def _query(self, prompt) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(self.device) # type: ignore
        outputs = self.model.generate( # type: ignore
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id # type: ignore
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):] # type: ignore
        return response

