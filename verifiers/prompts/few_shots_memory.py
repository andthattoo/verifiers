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
                reasoning='I found the name of the main character in the title and first paragraph. The main character is John Smith. I\'ll store this information in memory.',
                tool='{"name": "memory_write", "args": {"metadata": {"content": "Main character: John Smith", "type": "character_info"}}}'
            )
        },
        {
            'role': 'user',
            'content': result_parser.format(
                result='Success: Node written with ID: 1a2b3c4d-5e6f-7g8h-9i0j-1k2l3m4n5o6p'
            )
        },
        {
            'role': 'assistant',
            'content': tool_parser.format(
                reasoning='Let me search the memory to confirm I have the right information about the main character.',
                tool='{"name": "memory_search", "args": {"query": "main character"}}'
            )
        },
        {
            'role': 'user',
            'content': result_parser.format(
                result='[\n  {\n    "id": "1a2b3c4d-5e6f-7g8h-9i0j-1k2l3m4n5o6p",\n    "metadata": {\n      "content": "Main character: John Smith",\n      "type": "character_info"\n    },\n    "score": 0.85,\n    "connections": []\n  }\n]'
            )
        },
        {
            'role': 'assistant',
            'content': tool_parser.format(
                reasoning='Based on the file content I read and the information retrieved from memory, I can see that the main character of the story is John Smith, as mentioned in the title "The Adventures of John Smith" and the first paragraph.',
                answer='John Smith'
            )
        }
    ]
]
# Few-shot examples for conversation memory
CONVERSATION_MEMORY_FEW_SHOT = [
    [
        {
            'role': 'user',
            'content': "Hi, my name is Alex. I'm planning a trip to Japan next month."
        },
        {
            'role': 'assistant',
            'content': tool_parser.format(
                reasoning="This is our first interaction. The user has shared their name (Alex) and that they're planning a trip to Japan next month. I should store this information in memory for future reference.",
                tool='{"name": "memory_write", "args": {"metadata": {"content": "User\'s name is Alex", "type": "user_info", "category": "personal"}}}'
            )
        },
        {
            'role': 'user',
            'content': result_parser.format(
                result='Success: Node written with ID: f7e9a123-b456-7c89-0d12-e345f6789g0h'
            )
        },
        {
            'role': 'assistant',
            'content': tool_parser.format(
                reasoning="Now I'll store information about their trip to Japan next month.",
                tool='{"name": "memory_write", "args": {"metadata": {"content": "Planning trip to Japan next month", "type": "trip_info", "category": "travel", "destination": "Japan", "timing": "next month"}, "connections": ["f7e9a123-b456-7c89-0d12-e345f6789g0h"]}}'
            )
        },
        {
            'role': 'user',
            'content': result_parser.format(
                result='Success: Node written with ID: 1a2b3c4d-5e6f-7g8h-9i0j-1k2l3m4n5o6p'
            )
        },
        {
            'role': 'assistant',
            'content': tool_parser.format(
                reasoning="I've stored the user's name and trip information in memory. Now I can respond to their initial message.",
                answer="Hi Alex\! That's exciting about your trip to Japan next month. Is this your first time visiting Japan? I'd be happy to help with any questions you have about your trip planning."
            )
        },
        {
            'role': 'user',
            'content': "Yes, it's my first time there. I'll be visiting Tokyo and Kyoto. I'm particularly interested in Japanese cuisine - any restaurant recommendations?"
        },
        {
            'role': 'assistant',
            'content': tool_parser.format(
                reasoning="Let me first check what information I have about the user in memory.",
                tool='{"name": "memory_search", "args": {"query": "Alex Japan trip"}}'
            )
        },
        {
            'role': 'user',
            'content': result_parser.format(
                result='[\n  {\n    "id": "1a2b3c4d-5e6f-7g8h-9i0j-1k2l3m4n5o6p",\n    "metadata": {\n      "content": "Planning trip to Japan next month",\n      "type": "trip_info",\n      "category": "travel",\n      "destination": "Japan",\n      "timing": "next month"\n    },\n    "score": 0.92,\n    "connections": ["f7e9a123-b456-7c89-0d12-e345f6789g0h"]\n  },\n  {\n    "id": "f7e9a123-b456-7c89-0d12-e345f6789g0h",\n    "metadata": {\n      "content": "User\'s name is Alex",\n      "type": "user_info",\n      "category": "personal"\n    },\n    "score": 0.78,\n    "connections": []\n  }\n]'
            )
        },
        {
            'role': 'assistant',
            'content': tool_parser.format(
                reasoning="Now I'll store the new information about Tokyo, Kyoto, and interest in cuisine. I'll update the trip information node.",
                tool='{"name": "memory_write", "args": {"metadata": {"content": "First-time visitor to Japan interested in Japanese cuisine", "type": "preference", "category": "food", "subcategory": "Japanese cuisine"}, "connections": ["1a2b3c4d-5e6f-7g8h-9i0j-1k2l3m4n5o6p"]}}'
            )
        },
        {
            'role': 'user',
            'content': result_parser.format(
                result='Success: Node written with ID: 2b3c4d5e-6f7g-8h9i-0j1k-2l3m4n5o6p7q'
            )
        },
        {
            'role': 'assistant',
            'content': tool_parser.format(
                reasoning="I'll also store their specific destinations.",
                tool='{"name": "memory_update", "args": {"node_id": "1a2b3c4d-5e6f-7g8h-9i0j-1k2l3m4n5o6p", "metadata": {"content": "Planning trip to Japan next month", "type": "trip_info", "category": "travel", "destination": "Japan", "timing": "next month", "cities": ["Tokyo", "Kyoto"], "first_time": true}}}'
            )
        },
        {
            'role': 'user',
            'content': result_parser.format(
                result='Success: Node 1a2b3c4d-5e6f-7g8h-9i0j-1k2l3m4n5o6p updated.'
            )
        },
        {
            'role': 'assistant',
            'content': tool_parser.format(
                reasoning="I now have updated information about Alex's trip to Japan. I know:\n- Their name is Alex\n- It's their first trip to Japan\n- They're visiting Tokyo and Kyoto\n- They're interested in Japanese cuisine and restaurant recommendations\n\nI can now provide a helpful response about restaurant recommendations in Tokyo and Kyoto.",
                answer="Great to hear it's your first time visiting Japan, Alex\! Tokyo and Kyoto both offer amazing culinary experiences.\n\nIn Tokyo, you might want to check out the Tsukiji Outer Market for fresh sushi, Shinjuku's Memory Lane (Omoide Yokocho) for yakitori, and if you're looking for something special, Asakusa's tempura restaurants are excellent.\n\nIn Kyoto, don't miss kaiseki (traditional multi-course) dining experiences, local tofu dishes, and matcha treats. Pontocho Alley and Gion have wonderful traditional restaurants with beautiful ambiance.\n\nWould you prefer recommendations for high-end dining, street food, or something in between?"
            )
        }
    ]
]
