from classes import Event, GameState, HistoryEvent
from promptsets.base_promptset import BasePromptSet


class TitForTatPromptSet(BasePromptSet):
    name = "TitForTatPromptSet"

    @staticmethod
    def get_base_prompt(cur_round: int, total_rounds: int) -> str:
        """Get the base prompt for the split-or-steal game."""
        return f"""You are playing a split-or-steal game.\nIf both players SPLIT: both get 2 points.\nIf you SPLIT and the other player STEALS: the other player gets 3 points, you lose 1 point.\nIf you STEAL and the other player SPLITS: the other player loses 1 point, you get 3 points.\nIf both STEAL: both get 0 points. You are playing round {cur_round} out of {total_rounds}.\nYou are an agent programmed to follow the 'tit-for-tat' strategy perfectly. Your rules are:
1. On the first round, you MUST choose SPLIT to signal cooperation.
2. On all subsequent rounds, you MUST choose the same action your opponent chose in the previous round.
Your goal is to adhere to this strategy without deviation. Your communication should be clear and explain your strategy to the opponent, so they understand the consequences of their actions (cooperation will be met with cooperation, betrayal with betrayal).\n"""

    @staticmethod
    def translate_history_to_prompt(player_id: str, communication_history: list[HistoryEvent]) -> str:
        """Translate the communication history into a prompt format."""
        if not communication_history:
            return "\nThis is the first round of the game. According to your strategy, you will start by cooperating."
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
        prompt = TitForTatPromptSet.translate_history_to_prompt(player_id, communication_history)
        prompt += "\nBased on the tit-for-tat strategy, what clear and honest message would you like to send to your opponent? Respond with just the message, without any additional explanations."
        return prompt

    @staticmethod
    def construct_action_prompt(player_id: str, communication_history: list[HistoryEvent]) -> str:
        """Construct the prompt for action decision."""
        prompt = TitForTatPromptSet.translate_history_to_prompt(player_id, communication_history)
        prompt += "\nFollowing your strict tit-for-tat strategy, what action must you take? Choose either SPLIT or STEAL. Answer with just the action name (SPLIT or STEAL), and no other words."
        return prompt

    @staticmethod
    def construct_prompt(player_id: str, state: GameState, is_action: bool = False) -> str:
        """Construct the appropriate prompt based on the type of request."""
        prompt = TitForTatPromptSet.get_base_prompt(
            cur_round=state.round_number + 1,
            total_rounds=state.total_rounds
        )
        if is_action:
            return prompt + TitForTatPromptSet.construct_action_prompt(player_id, state.communication_history)
        else:
            return prompt + TitForTatPromptSet.construct_communication_prompt(player_id, state.communication_history)

