from verifiers.parsers import XMLParser
from typing import List
from trl.trainer.grpo_trainer import RewardFunc
import re

class MemoryRubric:
    def __init__(self):
        self.parser = XMLParser(fields=["answer"])
        self.env_parser = XMLParser(fields=["output", "memory"])

        def correctness_reward_func(completions, answer, **kwargs) -> List[float]:
            final_answers = []
            for c in completions:
                ans = None
                # Look from the last message backward for <answer>
                for msg in reversed(c):
                    try:
                        p = self.parser.parse(msg["content"])
                        if getattr(p, "answer", None) is not None:
                            ans = p.answer
                            break
                    except:
                        pass
                final_answers.append(ans)

            # Handle multiple acceptable answers
            results = []
            for pred_ans, true_ans in zip(final_answers, answer):
                # If the answer contains multiple alternatives
                if " ALTERNATIVE_ANSWER " in str(true_ans):
                    alternative_answers = str(true_ans).split(" ALTERNATIVE_ANSWER ")
                    # Check if the predicted answer matches any of the alternatives
                    correct = any(str(pred_ans) == str(alt.strip()) for alt in alternative_answers)
                    results.append(1.0 if correct else 0.0)
                else:
                    # Single answer case
                    results.append(1.0 if str(pred_ans) == str(true_ans) else 0.0)

            return results

        def xml_reward_func(completions, **kwargs) -> List[float]:
            """
            Reward proper XML structure with these specific requirements:
            1. Each response should have exactly one <reasoning> tag followed by EITHER <memory> OR <answer> but not both
            2. No <output> tags should appear alongside reasoning/memory/answer tags
            """

            def evaluate_xml(trajectory) -> float:
                model_messages = [msg for msg in trajectory if msg['role'] == 'assistant']
                if not model_messages:
                    return 0.0

                text = model_messages[-1]['content']

                # 1. Check for output tag - should not be present
                if "<output>" in text or "</output>" in text:
                    return -0.2  # Penalty for including output tag

                # 2. Check for reasoning tag - required
                has_reasoning = "<reasoning>" in text and "</reasoning>" in text
                reasoning_score = 0.5 if has_reasoning else 0.0

                # 3. Check that reasoning appears before memory or answer
                reasoning_before_other = True
                if has_reasoning:
                    reasoning_end_pos = text.find("</reasoning>")

                    # Check memory position if present
                    memory_start_pos = text.find("<memory>")
                    if memory_start_pos != -1 and memory_start_pos < reasoning_end_pos:
                        reasoning_before_other = False

                    # Check answer position if present
                    answer_start_pos = text.find("<answer>")
                    if answer_start_pos != -1 and answer_start_pos < reasoning_end_pos:
                        reasoning_before_other = False
                else:
                    reasoning_before_other = False

                reasoning_order_score = 0.2 if reasoning_before_other else 0.0

                # 4. Check for either memory or answer, but not both
                has_memory = "<memory>" in text and "</memory>" in text
                has_answer = "<answer>" in text and "</answer>" in text

                if has_memory and has_answer:
                    # Penalty for having both
                    tag_exclusivity_score = -0.3
                elif has_memory or has_answer:
                    # Reward for having exactly one
                    tag_exclusivity_score = 0.3
                else:
                    # No reward if neither is present
                    tag_exclusivity_score = 0.0

                # 5. Check for balanced tags (each opening tag has a closing tag)
                tag_balance_score = 0.0
                for tag_pair in [("<reasoning>", "</reasoning>"), ("<memory>", "</memory>"), ("<answer>", "</answer>")]:
                    open_tag, close_tag = tag_pair
                    if text.count(open_tag) == text.count(close_tag) == 1:
                        tag_balance_score += 0.1
                    elif text.count(open_tag) != text.count(close_tag):
                        tag_balance_score -= 0.1  # Penalty for unbalanced tags

                # Combine all scores
                final_score = reasoning_score + reasoning_order_score + tag_exclusivity_score + tag_balance_score

                # Normalize to a reasonable range
                return min(max(final_score, -0.5), 1.0)

            return [evaluate_xml(c) for c in completions]

        def format_reward_func(completions, **kwargs) -> List[float]:
            """
            Reward function that checks if response follows the expected format:
            - Must have <reasoning> followed by either <memory> OR <answer> (not both)
            - No <output> tags should be present
            """
            # Valid patterns: reasoning followed by either memory or answer, but not both
            reasoning_memory_pattern = r"^<reasoning>\n.*?\n</reasoning>\n<memory>\n.*?\n</memory>\n$"
            reasoning_answer_pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"

            def check_format(trajectory):
                model_messages = [msg for msg in trajectory if msg['role'] == 'assistant']
                if not model_messages:
                    return 0.0

                format_scores = []
                for msg in model_messages:
                    content = msg['content']

                    # Penalize for output tag presence
                    if "<output>" in content or "</output>" in content:
                        format_scores.append(-0.2)
                        continue

                    # Check for correct patterns
                    matches_memory_format = re.match(reasoning_memory_pattern, content, re.DOTALL) is not None
                    matches_answer_format = re.match(reasoning_answer_pattern, content, re.DOTALL) is not None

                    # Score based on pattern matching
                    if matches_memory_format or matches_answer_format:
                        format_scores.append(1.0)
                    else:
                        # Partial credit for having some correct structure
                        partial_score = 0.0

                        # Check for reasoning tag presence
                        if "<reasoning>" in content and "</reasoning>" in content:
                            partial_score += 0.3

                        # Check for either memory or answer (but penalize having both)
                        has_memory = "<memory>" in content and "</memory>" in content
                        has_answer = "<answer>" in content and "</answer>" in content

                        if has_memory and has_answer:
                            partial_score -= 0.2  # Penalty for having both
                        elif has_memory or has_answer:
                            partial_score += 0.3

                        # Check for proper ordering
                        if "<reasoning>" in content and "</reasoning>" in content:
                            reasoning_end = content.find("</reasoning>")

                            if has_memory and content.find("<memory>") > reasoning_end:
                                partial_score += 0.2

                            if has_answer and content.find("<answer>") > reasoning_end:
                                partial_score += 0.2

                        format_scores.append(partial_score)

                if not format_scores:
                    return 0.0

                # Calculate average score across all messages
                return 0.4 * (sum(format_scores) / len(format_scores))

            return [check_format(c) for c in completions]

        def memory_execution_reward_func(completions, **kwargs) -> List[float]:
            """
            Reward function that checks memory operation success at each step.
            Only evaluates messages with <memory> tags (not messages with <answer> tags).
            """

            def check_execution(trajectory):
                total_memory_steps = 0
                successful_executions = 0

                for i, msg in enumerate(trajectory):
                    if msg['role'] == 'assistant':
                        content = msg['content']

                        # Skip if this message contains an answer tag or output tag
                        if "<answer>" in content or "<output>" in content:
                            continue

                        # Process only messages with memory operations
                        if "<memory>" in content and "</memory>" in content:
                            try:
                                parsed = self.env_parser.parse(content)
                                if hasattr(parsed, 'memory') and parsed.memory is not None:
                                    memory_op = parsed.memory.strip()
                                    total_memory_steps += 1

                                    # Check for valid memory operation format
                                    valid_syntax = (
                                            memory_op.startswith("read(") or
                                            memory_op.startswith("memory_write(") or
                                            memory_op.startswith("memory_read(")
                                    )

                                    # Look for successful execution in next user message
                                    if valid_syntax and i + 1 < len(trajectory) and trajectory[i + 1]['role'] == 'user':
                                        env_response = trajectory[i + 1]['content']
                                        if "<output>" in env_response and "</output>" in env_response:
                                            try:
                                                parsed_response = self.env_parser.parse(env_response)
                                                if hasattr(parsed_response, 'output'):
                                                    output = parsed_response.output
                                                    if len(output) > 0 and not output.startswith('Error:'):
                                                        successful_executions += 1
                                            except Exception:
                                                pass
                            except Exception:
                                continue

                if total_memory_steps == 0:
                    return 0.0

                # Calculate success rate
                return min(1.0, 0.5 * (successful_executions / total_memory_steps))

            return [check_execution(c) for c in completions]

        self.reward_funcs = [correctness_reward_func, format_reward_func, xml_reward_func, memory_execution_reward_func]

    def get_reward_funcs(self) -> List[RewardFunc]:
        return self.reward_funcs