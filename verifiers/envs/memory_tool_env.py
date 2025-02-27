import json
import re
from typing import List, Dict, Any, Optional, Callable
from datasets import Dataset
from trl.trainer.grpo_trainer import RewardFunc

from verifiers.envs.multistep_env import MultiStepEnv
from verifiers.envs.tool_env import ToolEnv
from verifiers.parsers import XMLParser
from verifiers.utils import preprocess_dataset
from verifiers.rubrics import MemoryRubric
from verifiers.prompts import MEMORY_TOOL_PROMPT, MEMORY_TOOL_FEW_SHOT
from verifiers.tools.memory_tools import read, memory_write, memory_read

class MemoryToolEnv(ToolEnv):
    """
    A tool-based environment that provides memory operations using standard ToolEnv infrastructure.
    Supports three tools:
      - read(file_path, position, max_bytes)
      - memory_write(file_path, memoir)
      - memory_read(file_path)
    
    Limits how many times each can be called per episode.
    Completes when <answer> is found in the assistant's last message.
    """

    def __init__(self,
                 dataset: str = "longbench",
                 system_prompt: str = MEMORY_TOOL_PROMPT,
                 few_shot: Optional[List[Dict[str, str]]] = MEMORY_TOOL_FEW_SHOT[0],
                 N: int = 25,       # Max read calls
                 M: int = 15,       # Max memory_write calls
                 P: int = 15 + 1,   # Max memory_read calls
                 MB: int = 256,     # Max bytes to read
                 MM: int = 1024,    # Max memory size
                 max_steps: int = 50):
        
        self.files = {}
        self.memories = {}
        self.counter = {}
        self.N = N
        self.M = M
        self.P = P
        self.MB = MB
        self.MM = MM
        
        # Define custom tool implementations
        self._read = self._create_read_tool()
        self._memory_write = self._create_memory_write_tool()
        self._memory_read = self._create_memory_read_tool()
        
        # Initialize the ToolEnv with our custom tools
        super().__init__(
            dataset=dataset,
            tools=[self._read, self._memory_write, self._memory_read], 
            system_prompt=system_prompt,
            few_shot=few_shot,
            max_steps=max_steps
        )
        
        # Load dataset and initialize memory structures
        self._setup_dataset(dataset, system_prompt, few_shot)
        
        # Replace the default ToolRubric with our MemoryRubric
        self.rubric = MemoryRubric()
        
        # Override the parsers to match the memory environment expectations
        self.llm_parser = XMLParser(fields=["reasoning", ("tool", "memory", "answer")])
        self.env_parser = XMLParser(fields=["result"])
        
    def _setup_dataset(self, dataset: str, system_prompt: str, few_shot: Optional[List[Dict[str, str]]]):
        """Load and preprocess the dataset, setting up memory structures for each entry."""
        self.dataset = preprocess_dataset(
            dataset_name=dataset,
            split="test",
            system_prompt=system_prompt,
            few_shot=few_shot
        )
        
        for entry in self.dataset:
            self.files[entry["file_path"]] = entry["context"]
            self.memories[entry["file_path"]] = ""
            self.counter[entry["file_path"]] = {"read": 0, "memory_write": 0, "memory_read": 0}
        
        self.dataset = self.dataset.remove_columns(["context", "file_path"])
        print("Total files: ", len(self.files))
    
    def _create_read_tool(self) -> Callable:
        """Create a custom implementation of the read tool that uses our file storage."""
        def custom_read(file_path: str, position: int = 0, max_bytes: int = 256) -> str:
            if file_path not in self.files:
                return "Error: Invalid file_path."
                
            if self.counter[file_path]["read"] >= self.N:
                return "Error: read() calls exceeded."
            self.counter[file_path]["read"] += 1
                
            try:
                if max_bytes > self.MB:
                    return f"Error: max_bytes exceeds limit. Should be lower than {self.MB}."
                if max_bytes < 0:
                    return "Error: max_bytes can't be negative."
                if position < 0:
                    return "Error: position can't be negative."
                    
                return self.files[file_path][position:position+max_bytes]
            except:
                return "Error: Invalid read arguments."
                
        # Replace the function with our implementation but keep its signature and docstring
        custom_read.__name__ = read.__name__
        custom_read.__doc__ = read.__doc__
        custom_read.__annotations__ = read.__annotations__
        custom_read.__defaults__ = read.__defaults__
        
        return custom_read
    
    def _create_memory_write_tool(self) -> Callable:
        """Create a custom implementation of the memory_write tool that uses our memory storage."""
        def custom_memory_write(file_path: str, memoir: str) -> str:
            if file_path not in self.files:
                return "Error: Invalid file_path."
                
            if self.counter[file_path]["memory_write"] >= self.M:
                return "Error: memory_write() calls exceeded."
            self.counter[file_path]["memory_write"] += 1
                
            self.memories[file_path] += memoir
            # Limit memory size
            if len(self.memories[file_path]) > self.MM:
                self.memories[file_path] = self.memories[file_path][-self.MM:]
                
            return f"Wrote {len(memoir)} chars to memory."
                
        # Replace the function with our implementation but keep its signature and docstring
        custom_memory_write.__name__ = memory_write.__name__
        custom_memory_write.__doc__ = memory_write.__doc__
        custom_memory_write.__annotations__ = memory_write.__annotations__
        custom_memory_write.__defaults__ = memory_write.__defaults__
        
        return custom_memory_write
    
    def _create_memory_read_tool(self) -> Callable:
        """Create a custom implementation of the memory_read tool that uses our memory storage."""
        def custom_memory_read(file_path: str) -> str:
            if file_path not in self.files:
                return "Error: Invalid file_path."
                
            if self.counter[file_path]["memory_read"] >= self.P:
                return "Error: memory_read() calls exceeded."
            self.counter[file_path]["memory_read"] += 1
                
            return self.memories[file_path]
                
        # Replace the function with our implementation but keep its signature and docstring
        custom_memory_read.__name__ = memory_read.__name__
        custom_memory_read.__doc__ = memory_read.__doc__
        custom_memory_read.__annotations__ = memory_read.__annotations__
        custom_memory_read.__defaults__ = memory_read.__defaults__
        
        return custom_memory_read
        
    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        """Return the memory-specific reward functions."""
        return self.rubric.get_reward_funcs()
        
    def is_completed(self, messages: List[Dict[str, str]], **kwargs: Any) -> bool:
        """
        Override completion check to look for <answer> tags or maximum steps.
        """
        # First check the parent class's step count logic
        step_count = self._get_step_count(messages)
        if step_count >= self.max_steps:
            return True
            
        # Then check for answer tag
        try:
            parsed = self.llm_parser.parse(messages[-1]["content"])
            return getattr(parsed, "answer", None) is not None
        except:
            return False
            
    def env_response(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, str]:
        """
        Process tool calls from the assistant's messages and execute the appropriate tool.
        This overrides both the MultiStepEnv abstract method and the ToolEnv implementation.
        """
        try:
            parsed = self.llm_parser.parse(messages[-1]["content"])
            
            # Handle memory tag for backward compatibility
            if hasattr(parsed, 'memory') and parsed.memory is not None:
                # Parse the function call from memory content
                import re
                match = re.match(r'(\w+)\s*\((.*)\)', parsed.memory.strip())
                
                if match:
                    tool_name = match.group(1)
                    args_str = match.group(2)
                    
                    # Convert to JSON format for tool execution
                    args_dict = {}
                    for key_val in args_str.split(','):
                        if '=' in key_val:
                            key, val = key_val.split('=', 1)
                            key = key.strip()
                            val = val.strip()
                            
                            # Handle string values
                            if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                                val = val[1:-1]
                            # Handle numeric values
                            elif val.isdigit():
                                val = int(val)
                                
                            args_dict[key] = val
                    
                    # Format as JSON and execute
                    tool_json = {
                        "name": tool_name,
                        "args": args_dict
                    }
                    
                    result = self.execute_tool(json.dumps(tool_json))
                    return {"role": "user", "content": self.env_parser.format(result=result)}
            
            # Handle normal tool tag execution from ToolEnv
            elif hasattr(parsed, 'tool') and parsed.tool is not None:
                result = self.execute_tool(parsed.tool)
                return {"role": "user", "content": self.env_parser.format(result=result)}
                
        except Exception as e:
            return {"role": "user", "content": f"Error: {str(e)}. Please use the correct format for tool calls."}
            
        return {"role": "user", "content": "Error: No valid tool call found. Please use <tool> or <memory> tags with the correct format."}