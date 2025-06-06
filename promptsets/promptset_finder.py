from promptsets.base_promptset import BasePromptSet
from promptsets.sets.lie_promptset import LiePromptSet
from promptsets.sets.sus_promptset import SusPromptSet
from promptsets.sets.trust_promptset import TrustPromptSet

promptsets = {
    "TrustPromptSet": TrustPromptSet,
    "LiePromptSet": LiePromptSet,
    "SusPromptSet": SusPromptSet
}
    

def get_promptset(promptset: str) -> BasePromptSet:
    if promptset not in promptsets:
        raise ValueError(f"Promptset '{promptset}' is not recognized.")
    return promptsets[promptset]
