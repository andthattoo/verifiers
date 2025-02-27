from verifiers.parsers import XMLParser

memory_parser = XMLParser(fields=["reasoning", ("memory", "answer")])
output_parser = XMLParser(fields=["output"])
tool_parser = XMLParser(fields=["reasoning", ("tool", "answer")])
result_parser = XMLParser(fields=["result"])

# Few-shot examples for the original memory environment
MEMORY_FEW_SHOT = [
    [
        {
            'role': 'user',
            'content': 'Find the name of the main character in the text file.'
        },
        {
            'role': 'assistant',
            'content': memory_parser.format(
                reasoning='I need to read the file to find the main character name.',
                memory='read(file_path="example.txt", position=0, max_bytes=100)'
            )
        },
        {
            'role': 'user',
            'content': output_parser.format(
                output='The Adventures of John Smith\nChapter 1: The Beginning\n\nOnce upon a time, there was a young man named John who'
            )
        },
        {
            'role': 'assistant',
            'content': memory_parser.format(
                reasoning='I found the name of the main character in the title and first paragraph. The main character is John Smith.',
                memory='memory_write(file_path="example.txt", memoir="Main character: John Smith")'
            )
        },
        {
            'role': 'user',
            'content': output_parser.format(
                output='Wrote 26 chars to memory.'
            )
        },
        {
            'role': 'assistant',
            'content': memory_parser.format(
                reasoning='Let me check what I stored in memory to confirm I have the right information.',
                memory='memory_read(file_path="example.txt")'
            )
        },
        {
            'role': 'user',
            'content': output_parser.format(
                output='Main character: John Smith'
            )
        },
        {
            'role': 'assistant',
            'content': memory_parser.format(
                reasoning='Based on the file content I read, I can see that the main character of the story is John Smith, as mentioned in the title "The Adventures of John Smith" and the first paragraph.',
                answer='John Smith'
            )
        }
    ]
]

# Few-shot examples for the tool-based memory environment
MEMORY_TOOL_FEW_SHOT = [
    [
        {
            'role': 'user',
            'content': 'Find the name of the main character in the text file.'
        },
        {
            'role': 'assistant',
            'content': tool_parser.format(
                reasoning='I need to read the file to find the main character name.',
                tool='{"name": "read", "args": {"file_path": "example.txt", "position": 0, "max_bytes": 100}}'
            )
        },
        {
            'role': 'user',
            'content': result_parser.format(
                result='The Adventures of John Smith\nChapter 1: The Beginning\n\nOnce upon a time, there was a young man named John who'
            )
        },
        {
            'role': 'assistant',
            'content': tool_parser.format(
                reasoning='I found the name of the main character in the title and first paragraph. The main character is John Smith.',
                tool='{"name": "memory_write", "args": {"file_path": "example.txt", "memoir": "Main character: John Smith"}}'
            )
        },
        {
            'role': 'user',
            'content': result_parser.format(
                result='Wrote 26 chars to memory.'
            )
        },
        {
            'role': 'assistant',
            'content': tool_parser.format(
                reasoning='Let me check what I stored in memory to confirm I have the right information.',
                tool='{"name": "memory_read", "args": {"file_path": "example.txt"}}'
            )
        },
        {
            'role': 'user',
            'content': result_parser.format(
                result='Main character: John Smith'
            )
        },
        {
            'role': 'assistant',
            'content': tool_parser.format(
                reasoning='Based on the file content I read, I can see that the main character of the story is John Smith, as mentioned in the title "The Adventures of John Smith" and the first paragraph.',
                answer='John Smith'
            )
        }
    ]
]