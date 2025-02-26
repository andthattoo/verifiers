from typing import List, Dict, Any, Optional
from datasets import Dataset
from trl.trainer.grpo_trainer import RewardFunc
from verifiers.envs.multistep_env import MultiStepEnv
from verifiers.parsers import XMLParser
from verifiers.utils import preprocess_dataset
from verifiers.rubrics import MemoryRubric
from verifiers.prompts import MEMORY_PROMPT, MEMORY_FEW_SHOT

class MemoryToolEnv(MultiStepEnv):
    """
    Demonstrates 3 tools:
      - read(file_path, position, max_bytes)
      - memory_write(memoir)
      - memory_read()

    Limits how many times each can be called per episode.
    Completes when <answer> is found in the assistant’s last message.
    """

    def __init__(self,
                 dataset: str = "longbench",
                 system_prompt: str = MEMORY_PROMPT,
                    few_shot: Optional[List[Dict[str, str]]] = MEMORY_FEW_SHOT[0],
                 N: int = 25,       # Max read calls
                 M: int = 15,       # Max memory_write calls
                 P: int = 15 + 1,      # Max memory_read calls
                 MB: int = 256,     # Max bytes to read
                 MM: int = 1024,    # Max memory size
        ):
        super().__init__(system_prompt=system_prompt, few_shot=few_shot)

        self.files = {}
        self.memories= {}
        self.counter = {}
        self.dataset = preprocess_dataset(
            dataset_name=dataset,
            split="test",
            system_prompt=system_prompt,
            few_shot=few_shot
        )
        for entry in self.dataset:
            self.files[entry["file_path"]] = entry["context"]
            self.memories[entry["file_path"]] = ""
            self.counter[entry["file_path"]] = {"read":0, "memwrite":0, "memread":0}
        self.dataset = self.dataset.remove_columns(["context", "file_path"])

        print("Total files: ", len(self.files))

        self.rubric = MemoryRubric()

        self.n_read = 0
        self.n_write = 0
        self.n_memread = 0
        self.N = N
        self.M = M
        self.P = P
        self.MB = MB
        self.MM = MM
        self.llm_parser = XMLParser(fields=["reasoning", ("memory", "args"), "answer"])
        self.env_parser = XMLParser(fields=["output"])

    def get_dataset(self, **kwargs: Any) -> Dataset:
        return self.dataset

    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        return self.rubric.get_reward_funcs()

    def is_completed(self, messages: List[Dict[str, str]], **kwargs: Any) -> bool:
        # If last assistant message has <answer>, we're done
        try:
            parsed = self.llm_parser.parse(messages[-1]["content"])
            return getattr(parsed, "answer", None) is not None
        except:
            return False

    def env_response(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, str]:
        # Parse the last assistant message for a <tool> call
        try:
            parsed = self.llm_parser.parse(messages[-1]["content"])
            memory_content = getattr(parsed, "memory", None)

            if memory_content:
                # Parse the function call within memory content
                import re
                match = re.match(r'(\w+)\s*\((.*)\)', memory_content.strip())

                if match:
                    tool_name = match.group(1)  # Extract tool name (e.g., "read")
                    args_str = match.group(2)  # Extract arguments as string

                    return {"role": "user", "content": self.env_parser.format(output=self.handle_tool(tool_name, args_str))}
        except Exception as e:
            pass
        return {"role": "user", "content": "No valid tool call found."}

    @staticmethod
    def parse_args(args_str):
        arg_pairs = [p.split("=") for p in args_str.split(",") if "=" in p]
        args = {}

        for k, v in arg_pairs:
            key = k.strip()
            value = v.strip()

            # Handle string types (file_path or memoir)
            if key in ['file_path', 'memoir']:
                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
            else:
                # For all other arguments, attempt to convert to int
                try:
                    value = int(value)
                except ValueError:
                    # If conversion fails, keep as string
                    pass

            args[key] = value

        return args

    def handle_tool(self, tool_name: str, args_str: str) -> str:
        """
        Expects arguments in a simple 'key=value' format, e.g. position=0,max_bytes=10
        """
        # Parse args
        args = self.parse_args(args_str)

        file_path = args.get("file_path", "")
        if file_path not in self.files:
            return "Error: Invalid file_path."

        if tool_name == "read":
            if self.counter[file_path]["read"] >= self.N:
                return "Error: read() calls exceeded."
            self.counter[file_path]["read"] += 1
            return self.handle_read(args)
        elif tool_name == "memory_write":
            if self.counter[file_path]["memwrite"] >= self.M:
                return "Error: memory_write() calls exceeded."
            self.counter[file_path]["memwrite"] += 1
            return self.handle_memory_write(args)
        elif tool_name == "memory_read":
            if self.counter[file_path]["memread"] >= self.P:
                return "Error: memory_read() calls exceeded."
            self.counter[file_path]["memread"] += 1
            return self.handle_memory_read(args)
        else:
            return f"Error: Unknown tool '{tool_name}'."

    def handle_read(self, args: Dict[str, str]) -> str:
        try:
            fp = str(args.get("file_path"))
            pos = int(args.get("position", 0))
            mb = int(args.get("max_bytes", self.MB))
            if mb > self.MB:
                return f"Error: max_bytes exceeds limit Should be lower than {self.MB}."
            if mb < 0:
                return "Error: max_bytes can't be negative."
            if pos < 0:
                return "Error: position can't be negative."
            return self.files[fp][pos:pos+mb]
        except:
            return "Error: Invalid read arguments."

    def handle_memory_write(self, args: Dict[str, str]) -> str:
        # e.g. 'memoir=some text'
        memoir = args.get("memoir", "")
        file_path = args.get("file_path", "")
        self.memories[file_path] += memoir
        # Limit memory size
        if len(self.memories[file_path]) > self.MM:
            self.memories[file_path] = self.memories[file_path][-self.MM:]
        return f"Wrote {len(memoir)} chars to memory."

    def handle_memory_read(self, args: Dict[str, str]) -> str:
        file_path = args.get("file_path", "")
        return self.memories[file_path]