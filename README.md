# Split-or-Steal LLM Training

This project implements a training infrastructure for LLMs to play the split-or-steal game. The system uses multiple LLM agents that learn to play against each other, with their performance tracked using Weights & Biases.

## Project Structure

```
.
├── agents/
│   └── llm_agent.py      # LLM agent implementation
├── game/
│   └── environment.py    # Game environment
├── train/
│   └── trainer.py        # Training infrastructure
├── main.py              # Main training script
├── requirements.txt     # Project dependencies
└── README.md           # This file
```

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

## Usage

Run the training:
```bash
python main.py
```

## Components

### Game Environment
- Implements the split-or-steal game mechanics
- Handles game state and rewards
- Tracks game results

### LLM Agents
- Uses transformer models to make decisions
- Maintains game history
- Adapts strategy based on opponent behavior

### Training Infrastructure
- Manages multiple agents
- Simulates games between agents
- Tracks performance metrics
- Logs results to Weights & Biases

## Metrics Tracked

- Average reward per game
- Percentage of different game outcomes:
  - Both players split
  - Both players steal
  - Split-Steal combinations
- Agent performance over time

## Customization

You can modify the following parameters in `main.py`:
- Number of agents
- Games per round
- Number of training rounds
- Model selection
- Weights & Biases project name 
