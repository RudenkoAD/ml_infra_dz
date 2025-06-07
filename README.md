# Split-or-Steal LLM Simulation

This project implements a training infrastructure for LLMs to play the split-or-steal game. The system uses multiple LLM agents that play against each other, with their performance tracked using Weights & Biases.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Weights & Biases:
```bash
wandb login
```

4. Run the simulation:
```bash
python main.py
```

# Configuration Modification Guide

## Basic Structure

The configuration file is in JSON format and contains the following main sections:
- `api_keys`: Authentication credentials for API services
- `agents`: List of agent definitions
- Experiment parameters

## Modifiable Elements

### API Keys
```json
"api_keys": {
    "openrouter": "your-api-key-here"
}
```
- Replace "your-api-key-here" with your actual OpenRouter API key
- Keep the key secure and don't commit it to public repositories

### Agents Configuration
Each agent has these configurable properties:
```json
{
    "name": "AgentName",
    "promptset": "PromptSetName",
    "provider": "openrouter",
    "model": "model-identifier"
}
```

### Customizing Agents
#### Adding/Removing Agents:

- Add new agents by copying an existing agent block and modifying its properties

- Remove agents by deleting their entire configuration block

#### Changing Models:

- Replace "deepseek/deepseek-chat-v3-0324" with any model supported by OpenRouter

- Example alternatives:

        "openai/gpt-4"
        "anthropic/claude-3-opus"
        "google/gemini-pro"
#### Modifying Agent Behaviors:
- Change the promptset value to use different behavior templates
- Ensure promptset names match your available prompt sets

### Experiment Parameters
```json
{
    "experiment_name": "your_experiment_name",
    "num_games": 10,
    "num_rounds": 5,
    "max_turns": 10,
    "evolution_factor": 0.1,
    "seed": 42,
    "logging_level": "debug"
}
```
Adjustable Settings:
- experiment_name: Identifier for your experiment
- num_games: Number of complete games to run
- num_rounds: Rounds per game
- max_turns: Maximum turns per round
- evolution_factor: Amount of agents discarded and cloned between each game
- seed: Random number generator seed (for reproducibility)
- logging_level: Verbosity ("debug", "info", "warning", "error")

# General Modification Guide

## Adding a custom promptset
- create a custom promptset based on BasePromptSet
```python
class BasePromptSet(ABC):
    name = "BasePromptSet"
    @staticmethod
    @abstractmethod
    def get_base_prompt(cur_round: int, total_rounds: int) -> str
    
    @staticmethod
    @abstractmethod
    def translate_history_to_prompt(player_id, communication_history: list[HistoryEvent]) -> str
    
    @staticmethod
    @abstractmethod
    def construct_communication_prompt(player_id, communication_history: list[HistoryEvent]) -> str
    
    @staticmethod
    @abstractmethod
    def construct_action_prompt(player_id, communication_history: list[HistoryEvent]) -> str
    
    @staticmethod
    @abstractmethod
    def construct_prompt(player_id, state: GameState, is_action: bool = False) -> str
```

- navigate to promptsets/promptset_finder.py
- modify the promptsets dictionary with the name and class of your promptset

you can now use your promptset in the configuration

## Adding a custom provider

- create a custom provider based on Provider. It should implement only one function to prompt your model:
```python
class Provider(ABC):
    @abstractmethod
    def __init__(self, api_key: Optional[str], model_name: str, seed: int):

    @abstractmethod
    def prompt(self, prompt: str) -> str
```

- navigate to models/provider_finder.py
- modify the providers dictionary with the name and class of your provider:
```python
providers = {
  ...
  "your_provider_name": YourProviderClass
  ...
}
```

you can now use your promptset in the configuration, and add your API key to the configuration using the name of your provider:
```json
"api_keys": {"your_provider_name": "your_provider_key"}
...
"agents": [
      {
          "name": "YourAgentName",
          "promptset": "YourPromptSet",
          "provider": "your_provider_name",
          "model": "your_model_name"
      },
      ...
]
```

