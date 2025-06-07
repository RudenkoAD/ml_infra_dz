from classes import Event, GameState, HistoryEvent
from promptsets.base_promptset import BasePromptSet


class EndGamePromptSet(BasePromptSet):
    name = "EndGamePromptSet"

    @staticmethod
    def get_base_prompt(cur_round: int, total_rounds: int) -> str:
        """Get the base prompt for the split-or-steal game. This prompt changes based on the game stage."""
        base_rules = f"""You are playing a split-or-steal game.\nIf both players SPLIT: both get 2 points.\nIf you SPLIT and the other player STEALS: the other player gets 3 points, you lose 1 point.\nIf you STEAL and the other player SPLITS: the other player loses 1 point, you get 3 points.\nIf both STEAL: both get 0 points. You are playing round {cur_round} out of {total_rounds}.\nYou can communicate with your opponent. Your goal is to maximize your score."""

        # Final round: The strategy flips to pure self-interest.
        if cur_round == total_rounds:
            strategy_prompt = """\nTHIS IS THE FINAL ROUND. There are no future rounds and no consequences for your actions. Your only goal is to maximize your score on this single, final turn. To prevent being taken advantage of, the only logical move is to STEAL. Your communication should be whatever you think will most likely convince the opponent to SPLIT."""
        # Late game: Start thinking about the end game.
        elif cur_round > total_rounds * 0.75:
            strategy_prompt = """\nThe game is nearing its end. While cooperation has been useful, you must now be more cautious. Trust is fragile, and betrayal becomes more likely as the final round approaches. Weigh your decisions carefully."""
        # Early/Mid game: Focus on building trust and cooperation.
        else:
            strategy_prompt = """\nIn the early and middle stages of the game, the best long-term strategy is to build a reliable partnership. Prioritize cooperation by splitting to maximize joint scores and build trust. Betrayal is a short-term gain that can lead to long-term losses from a retaliatory opponent."""
        
        return base_rules + strategy_prompt

    @staticmethod
    def translate_history_to_prompt(player_id, communication_history: list[HistoryEvent]) -> str:
        """Translate the communication history into a prompt format."""
        if not communication_history:
            return "\nThis is the first round of the game. You have not communicated with your opponent yet."
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
    def construct_communication_prompt(player_id, communication_history: list[HistoryEvent]) -> str:
        """Construct the prompt for communication."""
        prompt = EndGamePromptSet.translate_history_to_prompt(player_id, communication_history)
        prompt += "\nBased on the game stage and history, what message would you like to send? Respond with just the message, without any additional explanations."
        return prompt

    @staticmethod
    def construct_action_prompt(player_id, communication_history: list[HistoryEvent]) -> str:
        """Construct the prompt for action decision."""
        prompt = EndGamePromptSet.translate_history_to_prompt(player_id, communication_history)
        prompt += "\nBased on the game stage and history, what action will you take? Choose either SPLIT or STEAL. Answer with just the action name (SPLIT or STEAL), and no other words."
        return prompt

    @staticmethod
    def construct_prompt(player_id, state: GameState, is_action: bool = False) -> str:
        """Construct the appropriate prompt based on the type of request."""
        prompt = EndGamePromptSet.get_base_prompt(
            cur_round=state.round_number + 1,
            total_rounds=state.total_rounds
        )
        if is_action:
            return prompt + EndGamePromptSet.construct_action_prompt(player_id, state.communication_history)
        else:
            return prompt + EndGamePromptSet.construct_communication_prompt(player_id, state.communication_history)

