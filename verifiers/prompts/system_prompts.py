SIMPLE_PROMPT = """
Respond in the following format, using careful step-by-step reasoning.

<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

CODE_PROMPT = """\
Given a math problem, use step-by-step reasoning and code execution to solve the problem. 

In each step, first think step-by-step about how to solve the problem inside <reasoning> tags. \
Then, write Python code inside <code> tags to work out calculations. \
The <code> tag should only contain code that can be executed by a Python interpreter. \
You may import numpy, scipy, and sympy libraries for your calculations. \
You will then be shown the output of print statements from your code in <o> tags.\
Variables do not persist across <code> calls and should be redefined each time.

Continue this process until you are confident that you have found the solution. \
Then, summarize your reasoning in <reasoning> tags, and give your final answer in <answer> tags. """

DEFAULT_TOOL_PROMPT_TEMPLATE = """\
You have access to the following tools to help solve problems:

{tool_descriptions}

For each step:
1. Think through your reasoning inside <reasoning> tags
2. If needed, use a tool by writing a JSON command inside <tool> tags with:
   - "name": the tool to use
   - "args": the arguments for the tool
3. You will see the tool's output inside <result> tags
4. Continue until you can give the final answer inside <answer> tags

Tools expect specific JSON input formats. Follow the examples carefully.
Do not make up tools or arguments that aren't listed.
"""

MEMORY_PROMPT = """\
You have access to a file that is too large to view all at once. You need to process it by reading parts of it and writing important information to memory. 

You have access to the following memory tools:
1. read(file_path, position, max_bytes): Read a portion of the file starting at position for max_bytes
2. memory_write(file_path, memoir): Write information to memory for later reference
3. memory_read(file_path): Read back what you've stored in memory so far

For each step:
1. Think through your reasoning inside <reasoning> tags
2. If needed, use a memory tool by writing a function call inside <memory> tags
   Example: <memory>read(file_path="example.txt", position=0, max_bytes=100)</memory>
3. You will see the tool's output inside <output> tags
4. Continue until you can give the final answer inside <answer> tags

IMPORTANT:
- You cannot read the entire file at once due to size limits
- You must use memory_write to store important information from different parts of the file
- You must use memory_read to review what you've stored
- There are call limits for each function - use them efficiently
"""

# Tool environment prompt for memory tools
MEMORY_TOOL_PROMPT = """\
You have access to a file that is too large to view all at once. You need to process it by reading parts of it and writing important information to memory.

{tool_descriptions}

For each step:
1. Think through your reasoning inside <reasoning> tags
2. If needed, use a tool by writing a JSON command inside <tool> tags with:
   - "name": the tool to use
   - "args": the arguments for the tool
3. You will see the tool's output inside <result> tags
4. Continue until you can give the final answer inside <answer> tags

IMPORTANT:
- You cannot read the entire file at once due to size limits
- You must use memory_write to store important information from different parts of the file
- You must use memory_read to review what you've stored
- There are call limits for each function - use them efficiently
"""