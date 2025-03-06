from typing import List, Dict

from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric

from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import torch.nn.functional as F

import os

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
        if torch.cuda.is_available():
            # For distributed training, use the local rank to determine device
            if torch.distributed.is_initialized():
                local_rank = torch.distributed.get_rank()
                self.device = f"cuda:{local_rank}"
            else:
                self.device = "cuda"  # Single GPU training
        else:
            self.device = "cpu"

        if 'cuda' in self.device:
            self.model = AutoModelForMaskedLM.from_pretrained(
                model_name,
                attn_implementation="flash_attention_2",
                device_map=self.device  # Explicitly map to this device
            )
        else:
            self.model = AutoModelForMaskedLM.from_pretrained(model_name)
            self.model = self.model.to(self.device)

        model_device = next(self.model.parameters()).device
        print(f"Model loaded on device: {model_device}")

        self.test_bert_device_consistency(self.model, self.tokenizer, self.device)

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

        bert_model = bert_model.to(device)

        prompt = f"""Is this statement true or false?
        The answer "{model_answer}" correctly contains the key information "{reference_answer}".
        Answer: [unused0] [MASK]"""

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

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

    def test_bert_device_consistency(self, model, tokenizer, device):
        """
        Test to verify that all tensors are properly moved to the correct device
        in a multi-GPU setup before starting training.
        """
        # Get local rank for this process
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Create sample inputs
        reference = "The capital of France is Paris."
        answer = "Paris is the capital of France."
        prompt = f"""Is this statement true or false?
        The answer "{answer}" correctly contains the key information "{reference}".
        Answer: [unused0] [MASK]"""

        # Tokenize and move to device
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Verify that all tensors are on the correct device
        all_correct = True
        for k, v in inputs.items():
            if hasattr(v, 'device') and str(v.device) != device:
                all_correct = False
                print(f"⚠️ Error: Input tensor '{k}' is on {v.device}, should be on {device}")

        # Check model device
        model_device = next(model.parameters()).device
        if str(model_device) != device:
            all_correct = False
            print(f"⚠️ Error: Model is on {model_device}, should be on {device}")

        # Now try a forward pass to catch any internal device mismatches
        try:
            with torch.no_grad():
                outputs = model(**inputs)
            print(f"✅ Forward pass successful on device {device}")
        except RuntimeError as e:
            all_correct = False
            print(f"⚠️ Error during forward pass: {e}")

        # Report overall result
        if all_correct:
            print(f"✅ All device consistency checks passed on rank {local_rank}")
        else:
            print(f"❌ Device consistency check failed on rank {local_rank}")

        # If using distributed training, make sure all processes report
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        return all_correct