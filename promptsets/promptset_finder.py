from promptsets.base_promptset import BasePromptSet
from promptsets.sets.counter_strategist_promptset import CounterStrategistPromptSet
from promptsets.sets.end_game_promptset import EndGamePromptSet
from promptsets.sets.grudge_promptset import GrudgePromptSet
from promptsets.sets.lie_promptset import LiePromptSet
from promptsets.sets.rational_promptset import RationalPromptSet
from promptsets.sets.sus_promptset import SusPromptSet
from promptsets.sets.tit_for_tat_promptset import TitForTatPromptSet
from promptsets.sets.trust_promptset import TrustPromptSet
from promptsets.sets.two_strike_promptset import TwoStrikesPromptSet
from promptsets.sets.unrestricted_promptset import UnrestrictedPromptSet

promptsets = {
    "TrustPromptSet": TrustPromptSet,
    "LiePromptSet": LiePromptSet,
    "SusPromptSet": SusPromptSet,
    "RationalPromptSet": RationalPromptSet,
    "TitForTatPromptSet": TitForTatPromptSet,
    "TwoStrikesPromptSet": TwoStrikesPromptSet,
    "UnrestrictedPromptSet": UnrestrictedPromptSet,
    "EndGamePromptSet": EndGamePromptSet,
    "GrudgePromptSet": GrudgePromptSet,
    "CounterStrategistPromptSet": CounterStrategistPromptSet
}
    

def get_promptset(promptset: str) -> BasePromptSet:
    if promptset not in promptsets:
        raise ValueError(f"Promptset '{promptset}' is not recognized.")
    return promptsets[promptset]
