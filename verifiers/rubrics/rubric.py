from abc import ABC
from typing import List, Dict
import logging

from trl.trainer.grpo_trainer import RewardFunc

def equals_reward_func(completions, answer, **kwargs) -> List[float]:
    responses = [c[0]['content'] for c in completions]
    return [1.0 if r == a else 0.0 for r, a in zip(responses, answer)]

class Rubric(ABC):
    def __init__(self, **kwargs):
        self.logger = logging.getLogger(f"verifiers.parsers.{self.__class__.__name__}")
        self.parser = None
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.reward_funcs = []
        self.reward_weights = None

    def get_assistant_messages(self, trajectory: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Helper function to extract assistant messages from a trajectory."""
        return [msg for msg in trajectory if msg['role'] == 'assistant']

    def get_last_answer(self, trajectory: List[Dict[str, str]]) -> str | None:
        """Extract the last answer from a trajectory."""
        for msg in reversed(trajectory):
            if msg['role'] == 'assistant':
                if self.parser is None:
                    raise ValueError("Parser is not set")
                parsed = self.parser.parse(msg['content'])
                if hasattr(parsed, 'answer') and parsed.answer is not None:
                    return parsed.answer
        return None

    def exact_answer_reward_func(self, completions, answer, **kwargs) -> List[float]:
        """Reward function that checks if the final answer matches the expected answer."""
        responses = [self.get_last_answer(c) for c in completions]
        return [1.0 if str(r) == str(a) else 0.0 for r, a in zip(responses, answer)]

    def multi_answer_exact_match_reward_func(self, completions, answer, **kwargs) -> List[float]:
        """Reward function that checks if the final answer matches any of the expected answers."""
        responses = [self.get_last_answer(c) for c in completions]
        results = []
        for r, a in zip(responses, answer):
            if isinstance(a, list):
                # Check if response matches any of the possible answers
                match = any(str(r) == str(possible_answer) for possible_answer in a)
                results.append(1.0 if match else 0.0)
            else:
                # Fallback to regular exact match for non-list answers
                results.append(1.0 if str(r) == str(a) else 0.0)
        return results

    def int_answer_reward_func(self, completions, answer, **kwargs) -> List[float]:
        """Reward function that checks if the final answer is an integer."""
        responses = [self.get_last_answer(c) for c in completions]
        return [1.0 if str(r).isdigit() else 0.0 for r in responses]

    def get_reward_funcs(self) -> List[RewardFunc]:
        return self.reward_funcs

    def get_reward_weights(self) -> List[float] | None:
        return self.reward_weights
