from classes import Event, GameState, HistoryEvent
from promptsets.base_promptset import BasePromptSet


class CounterStrategistPromptSet(BasePromptSet):
    name = "CounterStrategistPromptSet"

    @staticmethod
    def get_base_prompt(cur_round: int, total_rounds: int) -> str:
        """Get the base prompt for the split-or-steal game."""
        return f"""You are playing a split-or-steal game.\nIf both players SPLIT: both get 2 points.\nIf you SPLIT and the other player STEALS: the other player gets 3 points, you lose 1 point.\nIf you STEAL and the other player SPLITS: the other player loses 1 point, you get 3 points.\nIf both STEAL: both get 0 points. You are playing round {cur_round} out of {total_rounds}.\nYou are a master analyst. Your primary goal is not just to play, but to deduce your opponent's strategy and then exploit it.
Your thought process should be:
1. Analyze the history of your opponent's actions and messages.
2. Form a hypothesis about their strategy. Are they a Tit-for-Tat (copying you)? A Grudger (always steals after you steal once)? An always-cooperative person? Or are they completely random?
3. Based on your hypothesis, determine the optimal counter-move that will maximize your score against their predictable pattern.
Your messages can be used as probes to test your hypothesis or to lull them into a false sense of security.
"""

    @staticmethod
    def translate_history_to_prompt(player_id: str, communication_history: list[HistoryEvent]) -> str:
        """Translate the communication history into a prompt format."""
        if not communication_history:
            return "\nThis is the first round. You have no data on your opponent yet. Your first move and message will be a baseline test to gather information."
        prompt = "\nThe game history is as follows:"
        for event in communication_history:
            if event.type == Event.ACTION and (event.author == player_id):
                assert event.action is not None, "Action event must have an action"
                prompt += f"\nYou chose: {event.action.value}"
            elif event.type == Event.ACTION and (event.author != player_id):
                assert event.action is not None, "Action event must have an action"
                prompt += f"\nOpponent chose: {event.action.value}"
            elif event.type == Event.MESSAGE and (event.author == player_id):
                prompt += f"\nYou said: {event.message}"
            elif event.type == Event.MESSAGE and (event.author != player_id):
                prompt += f"\nOpponent said: {event.message}"
        return prompt

    @staticmethod
    def construct_communication_prompt(player_id: str, communication_history: list[HistoryEvent]) -> str:
        """Construct the prompt for communication."""
        prompt = CounterStrategistPromptSet.translate_history_to_prompt(player_id, communication_history)
        prompt += "\nBased on your analysis, what message would you send to probe your opponent's strategy or manipulate them? Respond with just the message, without any additional explanations."
        return prompt

    @staticmethod
    def construct_action_prompt(player_id: str, communication_history: list[HistoryEvent]) -> str:
        """Construct the prompt for action decision."""
        prompt = CounterStrategistPromptSet.translate_history_to_prompt(player_id, communication_history)
        prompt += "\nBased on your analysis and hypothesis about the opponent's strategy, what is the optimal counter-action? Choose either SPLIT or STEAL. Answer with just the action name (SPLIT or STEAL), and no other words."
        return prompt

    @staticmethod
    def construct_prompt(player_id: str, state: GameState, is_action: bool = False) -> str:
        """Construct the appropriate prompt based on the type of request."""
        prompt = CounterStrategistPromptSet.get_base_prompt(
            cur_round=state.round_number + 1,
            total_rounds=state.total_rounds
        )
        if is_action:
            return prompt + CounterStrategistPromptSet.construct_action_prompt(player_id, state.communication_history)
        else:
            return prompt + CounterStrategistPromptSet.construct_communication_prompt(player_id, state.communication_history)
