import json
import re
from typing import List, Dict, Any, Optional, Callable, Union
from datasets import Dataset
from trl.trainer.grpo_trainer import RewardFunc

from verifiers.envs.multistep_env import MultiStepEnv
from verifiers.envs.tool_env import ToolEnv
from verifiers.parsers import XMLParser
from verifiers.utils import preprocess_dataset
from verifiers.rubrics import MemoryRubric, ToolRubric
from verifiers.prompts import MEMORY_TOOL_PROMPT, MEMORY_TOOL_FEW_SHOT, CONVERSATION_MEMORY_PROMPT, CONVERSATION_MEMORY_FEW_SHOT, DEFAULT_TOOL_PROMPT_TEMPLATE
from verifiers.tools.memory_tools import write, search, update

class MemoryToolEnv(ToolEnv):
    """
    A tool-based environment that provides graph-based memory operations using standard ToolEnv infrastructure.
    Supports four tools:
      - read(file_path, position, max_bytes) - only in file mode
      - memory_write(metadata, connections)
      - memory_search(query, top_k, bfs_depth, bm25_candidates)
      - memory_update(node_id, metadata, add_connections, remove_connections)
    
    Can operate in two modes:
      1. File processing mode: Read from files and store important information in memory
      2. Conversation mode: Track user preferences and conversation history in memory
    
    Limits how many times each can be called per episode.
    Completes when <answer> is found in the assistant's last message.
    """

    def __init__(self,
                 dataset: str = "memory",
                 system_prompt: str = MEMORY_TOOL_PROMPT,
                 few_shot: Optional[List[Dict[str, str]]] = MEMORY_TOOL_FEW_SHOT[0],
                 conversation_mode: bool = False,
                 N: int = 25,       # Max read calls
                 M: int = 15,       # Max memory_write calls
                 P: int = 15,       # Max memory_search calls
                 U: int = 10,       # Max memory_update calls
                 MB: int = 256,     # Max bytes to read
                 max_steps: int = 50):
        
        # For conversation mode, override the default prompt and few shot examples
        if conversation_mode:
            system_prompt = CONVERSATION_MEMORY_PROMPT
            few_shot = CONVERSATION_MEMORY_FEW_SHOT[0]
        
        self.files = {}
        self.counter = {}
        self.N = N
        self.M = M
        self.P = P
        self.U = U
        self.MB = MB
        self.conversation_mode = conversation_mode
        
        # Define custom tool implementations
        self._read = self._create_read_tool()
        self._memory_write = self._create_memory_write_tool()
        self._memory_search = self._create_memory_search_tool()
        self._memory_update = self._create_memory_update_tool()
        
        # Initialize tools based on mode
        if conversation_mode:
            tools = [self._memory_write, self._memory_search, self._memory_update]
        else:
            tools = [self._read, self._memory_write, self._memory_search, self._memory_update]
        
        # Initialize the ToolEnv with our custom tools
        super().__init__(
            dataset=dataset,
            tools=tools, 
            system_prompt=system_prompt,
            few_shot=few_shot,
            max_steps=max_steps
        )
        
        # Load dataset and initialize memory structures
        self._setup_dataset(dataset, system_prompt, few_shot)
        
        # Replace the default ToolRubric with our MemoryRubric
        self.rubric = ToolRubric()
        
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
        
        if self.conversation_mode:
            # In conversation mode, we only need a single counter for all interactions
            self.conversation_id = "conversation"
            self.counter[self.conversation_id] = {
                "memory_write": 0, 
                "memory_search": 0, 
                "memory_update": 0
            }
        else:
            # In file mode, set up counters for each file
            for entry in self.dataset:
                self.files[entry["file_path"]] = entry["context"]
                self.counter[entry["file_path"]] = {
                    "read": 0, 
                    "memory_write": 0, 
                    "memory_search": 0, 
                    "memory_update": 0
                }
            
            # Only remove columns in file mode
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
                
        # Replace the function with our implementation but keep its signature
        custom_read.__name__ = "read"
        custom_read.__doc__ = "Read from a file at a given position with a maximum number of bytes."
        
        return custom_read
    
    def _create_memory_write_tool(self) -> Callable:
        """Create a custom implementation of the memory_write tool that uses our graph-based memory."""
        def custom_memory_write(metadata: Union[str, Dict[str, Any]], 
                              connections: Optional[List[str]] = None) -> str:
            current_file = self._get_current_file()
            if not current_file:
                return "Error: No active file context."
                
            if self.counter[current_file]["memory_write"] >= self.M:
                return "Error: memory_write() calls exceeded."
            self.counter[current_file]["memory_write"] += 1
                
            try:
                node_id = write(metadata, connections)
                return f"Success: Node written with ID: {node_id}"
            except Exception as e:
                return f"Error: Failed to write to memory: {str(e)}"
                
        # Set function properties
        custom_memory_write.__name__ = "memory_write"
        custom_memory_write.__doc__ = "Write metadata and optional connections to the memory graph."
        
        return custom_memory_write
    
    def _create_memory_search_tool(self) -> Callable:
        """Create a custom implementation of the memory_search tool."""
        def custom_memory_search(query: str, top_k: int = 5, 
                              bfs_depth: int = 1, bm25_candidates: int = 10) -> str:
            current_file = self._get_current_file()
            if not current_file:
                return "Error: No active file context."
                
            if self.counter[current_file]["memory_search"] >= self.P:
                return "Error: memory_search() calls exceeded."
            self.counter[current_file]["memory_search"] += 1
                
            try:
                results = search(query, top_k, bfs_depth, bm25_candidates)
                return json.dumps(results, indent=2)
            except Exception as e:
                return f"Error: Failed to search memory: {str(e)}"
                
        # Set function properties
        custom_memory_search.__name__ = "memory_search"
        custom_memory_search.__doc__ = "Search the memory graph using a query with BM25 + semantic search."
        
        return custom_memory_search
    
    def _create_memory_update_tool(self) -> Callable:
        """Create a custom implementation of the memory_update tool."""
        def custom_memory_update(node_id: str, 
                               metadata: Optional[Union[str, Dict[str, Any]]] = None,
                               add_connections: Optional[List[str]] = None,
                               remove_connections: Optional[List[str]] = None) -> str:
            current_file = self._get_current_file()
            if not current_file:
                return "Error: No active file context."
                
            if self.counter[current_file]["memory_update"] >= self.U:
                return "Error: memory_update() calls exceeded."
            self.counter[current_file]["memory_update"] += 1
                
            try:
                success = update(node_id, metadata, add_connections, remove_connections)
                if success:
                    return f"Success: Node {node_id} updated."
                else:
                    return f"Error: Node {node_id} not found."
            except Exception as e:
                return f"Error: Failed to update memory: {str(e)}"
                
        # Set function properties
        custom_memory_update.__name__ = "memory_update"
        custom_memory_update.__doc__ = "Update an existing node in the memory graph."
        
        return custom_memory_update
        
    def _get_current_file(self) -> Optional[str]:
        """Helper to get the current file or conversation ID."""
        if self.conversation_mode:
            return self.conversation_id
            
        if hasattr(self, 'current_episode') and self.current_episode:
            for file_path in self.files:
                if file_path in self.counter:
                    return file_path
        return None
        
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