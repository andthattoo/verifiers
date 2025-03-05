from typing import List, Dict

from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric

from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import torch.nn.functional as F

class ToolRubric(Rubric):
    def __init__(self,
                 parser: XMLParser = XMLParser(fields=["reasoning", ("tool", "answer")]),
                 env_parser: XMLParser = XMLParser(fields=["result"])):
        self.parser = parser
        self.env_parser = env_parser
        self.reward_funcs = [
            self.multi_answer_semantic_reward_func,
            self.tool_execution_reward_func,
            self.parser.get_format_reward_func(),
            self.parser.get_xml_reward_func(),
        ]

        # Load model and tokenizer
        model_name = "answerdotai/ModernBERT-Large-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            self.model = AutoModelForMaskedLM.from_pretrained(model_name, attn_implementation="flash_attention_2")
        else:
            self.model = AutoModelForMaskedLM.from_pretrained(model_name)

        self.model.to(self.device)

    def tool_execution_reward_func(self, completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
        """
        Reward function that checks tool execution success.

        Uses XMLParser to identify proper tool calls.
        """
        def check_execution(trajectory):
            tool_attempts = 0
            successful_executions = 0
            
            # Find assistant messages with tools and their responses
            for i, msg in enumerate(trajectory):
                if msg['role'] == 'assistant':
                    # Use parser to check for tool tag
                    parsed = self.parser.parse(msg['content'])
                    if hasattr(parsed, 'tool') and parsed.tool is not None:
                        # Found a properly formatted tool message
                        if i + 1 < len(trajectory) and trajectory[i + 1]['role'] == 'user':
                            tool_attempts += 1
                            # Check response with env_parser
                            parsed_response = self.env_parser.parse(trajectory[i + 1]['content'])
                            if hasattr(parsed_response, 'result') and parsed_response.result is not None and not parsed_response.result.startswith("Error:"):
                                successful_executions += 1
            
            # Calculate reward
            if tool_attempts == 0:
                return 0.0
            return 0.2 * (successful_executions / tool_attempts)
        
        return [check_execution(c) for c in completions]

    def multi_answer_semantic_reward_func(self, completions, answer, **kwargs) -> List[float]:
        """Reward function that checks if the final answer matches any of the expected answers."""
        responses = [self.get_last_answer(c) for c in completions]
        results = []
        for r, a in zip(responses, answer):
            if isinstance(a, list):
                best_score = 0.0
                for ans in a:
                    flag, metadata = self._evaluate_answer(r, ans, self.model, self.tokenizer, self.device)
                    if metadata["bert_confidence"] > best_score:
                        best_score = metadata["bert_confidence"]
                    if flag:
                        break
                results.append(best_score)
            else:
                # Fallback to regular exact match for non-list answers
                flag, metadata = self._evaluate_answer(r,a, self.model, self.tokenizer, self.device)
                results.append(metadata["bert_confidence"] if flag else 0.0)
        return results

    def _evaluate_answer(self, model_answer, reference_answer, bert_model, tokenizer, device):
        """
        Evaluates if a model's verbose answer matches a concise reference answer
        using a combination of string matching and BERT embeddings.

        Args:
            model_answer (str): The verbose answer from the model
            reference_answer (str): The concise reference answer
            bert_model: The BERT model
            tokenizer: The tokenizer for the BERT model
            device: The device to run on ('cuda' or 'cpu')

        Returns:
            bool: Whether the answer is correct
            dict: Additional information about the evaluation
        """
        results = {}

        prompt = f"""Is this statement true or false?
        The answer "{model_answer}" correctly contains the key information "{reference_answer}".
        Answer: [unused0] [MASK]"""

        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Get model prediction
        with torch.no_grad():
            outputs = bert_model(**inputs)

        # Find the masked token position
        mask_idx = (inputs.input_ids == tokenizer.mask_token_id).nonzero()[0, 1]

        # Get logits for the masked position
        logits = outputs.logits[0, mask_idx]

        # Get the top predicted tokens
        top_k = 5
        top_values, top_indices = torch.topk(logits, top_k)
        top_tokens = [tokenizer.decode(idx) for idx in top_indices]

        # Check if positive words are in the top predictions
        positive_words = ["true", "yes", "correct", "right"]
        negative_words = ["false", "no", "incorrect", "wrong"]

        # Count positive vs negative predictions
        pos_count = sum(1 for token in top_tokens if any(pos in token.lower() for pos in positive_words))
        neg_count = sum(1 for token in top_tokens if any(neg in token.lower() for neg in negative_words))

        # Get confidence score based on logit values
        true_confidence = F.softmax(top_values, dim=0)[0].item() if "true" in top_tokens[0].lower() else 0

        results["bert_top_tokens"] = top_tokens
        results["bert_confidence"] = true_confidence
        results["bert_match"] = pos_count > neg_count

        # Final decision
        return results["bert_match"], results