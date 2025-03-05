from verifiers.parsers import XMLParser

math_parser = XMLParser(fields=["reasoning", "answer"])
MATH_FEW_SHOT = [
    [
        {'role': 'user', 'content': 'What is the largest single-digit prime number?'},
        {'role': 'assistant', 'content': math_parser.format(
            reasoning='The largest single-digit prime number is 7.',
            answer='7'
        )}
    ]
]

DOUBLECHECK_FEW_SHOT = [
    [
        {'role': 'user', 'content': 'What is the largest single-digit prime number?'},
        {'role': 'assistant', 'content': math_parser.format(
            reasoning='The largest single-digit prime number is 7.',
            answer='7'
        )},
        {'role': 'user', 'content': 'Are you sure?'},
        {'role': 'assistant', 'content': math_parser.format(
            reasoning='The only larger single-digit numbers are 8 and 9, which are not prime. So yes, the answer is 7.',
            answer='7'
        )}
    ]
]

code_parser = XMLParser(fields=["reasoning", ("code", "answer")])
output_parser = XMLParser(fields=["output"])
CODE_FEW_SHOT = [
    [
        {
            'role': 'user',
            'content': 'What is sum of the first 100 positive even integers?'
        },
        {
            'role': 'assistant',
            'content': code_parser.format(
                reasoning='Let\'s compute the sum of the first 100 positive even integers.',
                code='print(sum(range(2, 102, 2)))'
            )
        },
        {
            'role': 'user', 
            'content': output_parser.format(output='2550')
        },
        {
            'role': 'assistant',
            'content': code_parser.format(reasoning='The answer is 2550.', answer='2550')
        },
        {
            'role': 'user',
            'content': 'What is the sum of the first 100 natural numbers, minus the largest prime number less than 100?'
        },
        {
            'role': 'assistant',
            'content': code_parser.format(
                reasoning='The sum of the first n natural numbers is given by the formula n(n+1)/2.',
                code='print(100*101/2)'
            )
        },
        {
            'role': 'user',
            'content': output_parser.format(output='5050')
        },
        {
            'role': 'assistant',
            'content': code_parser.format(
                reasoning='The sum of the first 100 natural numbers is 5050. Now we need to subtract the largest prime number less than 100.',
                code='print(5050 - 97)'
            )
        },
        {
            'role': 'user',
            'content': output_parser.format(output='4953')
        },
        {
            'role': 'assistant',
            'content': code_parser.format(
                reasoning='The largest prime number less than 100 is 97. Subtracting this from 5050 gives 4953.',
                answer='4953'
            )
        }
    ]
]

tool_parser = XMLParser(fields=["reasoning", ("tool", "answer")])
result_parser = XMLParser(fields=["result"])

TOOL_FEW_SHOT = [
    [
        {
            'role': 'user',
            'content': 'What is the current working directory?'
        },
        {
            'role': 'assistant',
            'content': tool_parser.format(
                reasoning='Let\'s use the pwd command to find out the current working directory.',
                tool='pwd'
            )
        },
        {
            'role': 'user',
            'content': result_parser.format(result='/Users/user/project')
        },
        {
            'role': 'assistant',
            'content': tool_parser.format(
                reasoning='The current working directory is /Users/user/project.',
                answer='/Users/user/project'
            )
        },
        {
            'role': 'user',
            'content': 'How many Python files are in the current directory and its subdirectories?'
        },
        {
            'role': 'assistant',
            'content': tool_parser.format(
                reasoning='Let\'s use the find command to count Python files.',
                tool='find . -name "*.py" | wc -l'
            )
        },
        {
            'role': 'user',
            'content': result_parser.format(result='42')
        },
        {
            'role': 'assistant',
            'content': tool_parser.format(
                reasoning='There are 42 Python files in the current directory and its subdirectories.',
                answer='42'
            )
        }
    ]
]

COMMONSENSE_FEW_SHOT = [
    [
        {
            'role': 'user',
            'content': 'Which would be louder: a mouse or an elephant?'
        },
        {
            'role': 'assistant',
            'content': tool_parser.format(
                reasoning='Let\'s compare the volume levels of a mouse and an elephant.',
                tool='compare mouse elephant volume'
            )
        },
        {
            'role': 'user',
            'content': result_parser.format(result='''{
  "difference": -4,
  "mouse_volume": 1,
  "elephant_volume": 5
}''')
        },
        {
            'role': 'assistant',
            'content': tool_parser.format(
                reasoning='Based on the comparison, an elephant has a volume level of 5 while a mouse has a volume level of 1 (on a scale of 1-5). The difference of -4 indicates the elephant is much louder.',
                answer='An elephant would be louder than a mouse.'
            )
        },
        {
            'role': 'user',
            'content': 'What properties does a car have?'
        },
        {
            'role': 'assistant',
            'content': tool_parser.format(
                reasoning='Let\'s look up the properties of a car.',
                tool='get_related car'
            )
        },
        {
            'role': 'user',
            'content': result_parser.format(result='''{
  "properties": ["metallic", "fast", "loud", "heavy"],
  "category": "vehicle"
}''')
        },
        {
            'role': 'assistant',
            'content': tool_parser.format(
                reasoning='A car has several key properties: it is metallic, fast, loud, and heavy. It belongs to the category of vehicles.',
                answer='A car is a vehicle that is metallic, fast, loud, and heavy.'
            )
        }
    ]
]

SEARCH_FEW_SHOT = [
    [
        {'role': 'user', 'content': 'What programming languages were used to build Facebook?'},
        {'role': 'assistant', 'content': tool_parser.format(
            reasoning="To answer what programming languages were used to build Facebook, I need to search for information about Facebook's technology stack and development history.",
            tool='{"name": "search", "args": {"query": "programming languages used to build Facebook", "num_results": 2}}'
        )},
        {'role': 'user', 'content': result_parser.format(
            result="What programming language is Facebook written in? - Quora\nhttps://www.quora.com/What-programming-language-is-Facebook-written-in\nPhillip Nelson. Former CTO, Python Developer, Consultant. Originally Answered: What programming language is Facebook coded in? PHP is the main language used by Facebook. Specifically the HipHop Virtual Machine (hhvm) which is an open source virtual machine for executing programs written in PHP.\n\nWhat Programming Languages Does Facebook Use? - Code Platoon\nhttps://www.codeplatoon.org/what-programming-languages-does-facebook-use/\nThe platform was initially written in PHP, a server-side scripting language. As the site grew, Facebook developed its own version of PHP called Hack, which was designed to maintain compatibility with PHP while adding static typing features to improve developer productivity."
        )},
        {'role': 'assistant', 'content': tool_parser.format(
            reasoning="I've found that Facebook was initially built using PHP, and later developed their own version of PHP called Hack. Let me search for more information about their full technology stack.",
            tool='{"name": "search", "args": {"query": "Facebook technology stack JavaScript React", "num_results": 1}}'
        )},
        {'role': 'user', 'content': result_parser.format(
            result="Facebook Tech Stack: The Technology Behind the Social Media Giant\nhttps://theiotmagazine.com/facebook-tech-stack-the-technology-behind-the-social-media-giant-a9982f2778c4\nProgramming Languages: Facebook uses a variety of programming languages, with PHP and Hack being primary server-side languages. JavaScript is used for client-side development. The company also utilizes Python for data analysis and C++ for backend services that require high performance. Facebook created React, a JavaScript library for building user interfaces, which is now widely used across the web."
        )},
        {'role': 'assistant', 'content': tool_parser.format(
            reasoning="Based on my research, I can now provide a comprehensive answer about the programming languages used to build Facebook. The information shows that Facebook was initially built with PHP, and they later developed their own PHP variant called Hack for server-side programming. On the client side, they use JavaScript, particularly with their own React library. They also use Python for data analysis and C++ for performance-critical backend services. This gives me enough information to provide a concise answer listing the primary programming languages used in Facebook's development.",
            answer="PHP, Hack, JavaScript, Python, C++"
        )}
    ],
    [
        {'role': 'user', 'content': 'Who invented the airplane?'},
        {'role': 'assistant', 'content': tool_parser.format(
            reasoning="I need to search for who invented the airplane.",
            tool='{"name": "search", "args": {"query": "who invented the airplane", "num_results": 1}}'
        )},
        {'role': 'user', 'content': result_parser.format(
            result="Wright brothers - Wikipedia\nhttps://en.wikipedia.org/wiki/Wright_brothers\nThe Wright brothers, Orville and Wilbur Wright, are credited with inventing, building, and flying the world's first successful airplane."
        )},
        {'role': 'assistant', 'content': tool_parser.format(
            reasoning="The result confirms the airplane was invented by the Wright brothers.",
            answer="The Wright brothers."
        )}
    ]
]

SEARCH_FEW_SHOT_QA = [
    [
        {'role': 'user', 'content': 'Are both Elon Musk and Jack Ma founders of technology companies?'},
        {'role': 'assistant', 'content': tool_parser.format(
            reasoning="To determine if both Elon Musk and Jack Ma are founders of technology companies, I need to research each individual and their business backgrounds. I'll start by searching for information about Elon Musk.",
            tool='{"name": "search", "args": {"query": "Elon Musk founder companies", "num_results": 1}}'
        )},
        {'role': 'user', 'content': result_parser.format(
            result="Elon Musk - Wikipedia\nhttps://en.wikipedia.org/wiki/Elon_Musk\nElon Reeve Musk is a business magnate and investor. He is the founder, chairman, CEO, and CTO of SpaceX; angel investor, CEO, product architect and former chairman of Tesla, Inc.; owner, chairman and CTO of X Corp.; founder of the Boring Company and xAI; co-founder of Neuralink and OpenAI; and president of the Musk Foundation."
        )},
        {'role': 'assistant', 'content': tool_parser.format(
            reasoning="The search confirms that Elon Musk is a founder of several technology companies including SpaceX, and co-founder of Neuralink and OpenAI. Now I need to find information about Jack Ma.",
            tool='{"name": "search", "args": {"query": "Jack Ma founder Alibaba", "num_results": 1}}'
        )},
        {'role': 'user', 'content': result_parser.format(
            result="Jack Ma - Wikipedia\nhttps://en.wikipedia.org/wiki/Jack_Ma\nJack Ma, or Ma Yun, is a Chinese business magnate, investor and philanthropist. He is the co-founder and former executive chairman of Alibaba Group, a multinational technology conglomerate. In addition, Ma is also the co-founder of Yunfeng Capital, a Chinese private equity firm."
        )},
        {'role': 'assistant', 'content': tool_parser.format(
            reasoning="Based on my research, I can now answer the question with confidence. The search results show that Elon Musk is the founder of SpaceX, The Boring Company, and xAI, as well as a co-founder of Neuralink and OpenAI. These are all technology companies. Jack Ma is the co-founder of Alibaba Group, which is explicitly described as a 'multinational technology conglomerate.' He also co-founded Yunfeng Capital, though that's a private equity firm rather than a technology company directly. Both individuals have founded multiple ventures, but what's important for the question is that they both have founded at least one technology company. Elon Musk has founded several technology companies (SpaceX, Neuralink, OpenAI), and Jack Ma has founded at least one major technology company (Alibaba Group). Therefore, the answer is yes.",
            answer="Yes."
        )}
        ]
    ]
CALCULATOR_FEW_SHOT = [
    [
        {
            'role': 'user',
            'content': 'Janet has 3 times as many marbles as Tom. If Tom has 12 marbles, how many marbles do they have together?'
        },
        {
            'role': 'assistant',
            'content': tool_parser.format(
                reasoning="Let's solve this step by step:\n1. First, calculate how many marbles Janet has (3 times Tom's)",
                tool='{"name": "calculator", "args": {"expression": "3 * 12"}}'
            )
        },
        {
            'role': 'user',
            'content': result_parser.format(result="36")
        },
        {
            'role': 'assistant',
            'content': tool_parser.format(
                reasoning="2. Now we know:\n   - Tom has 12 marbles\n   - Janet has 36 marbles\n   Let's add them together",
                tool='{"name": "calculator", "args": {"expression": "36 + 12"}}'
            )
        },
        {
            'role': 'user',
            'content': result_parser.format(result="48")
        },
        {
            'role': 'assistant',
            'content': tool_parser.format(
                reasoning="Together they have 48 marbles:\n- Tom: 12 marbles\n- Janet: 36 marbles (3 times Tom's)\n- Total: 48 marbles",
                answer="48 marbles"
            )
        }
    ],
    [
        {
            'role': 'user',
            'content': 'Samantha is baking cookies. Each batch requires 2.5 cups of flour. If she has 10 cups of flour, how many complete batches can she make?'
        },
        {
            'role': 'assistant',
            'content': tool_parser.format(
                reasoning="To find how many complete batches Samantha can make, I need to divide the total amount of flour by the amount needed per batch.",
                tool='{"name": "calculator", "args": {"expression": "10 / 2.5"}}'
            )
        },
        {
            'role': 'user',
            'content': result_parser.format(result="4.0")
        },
        {
            'role': 'assistant',
            'content': tool_parser.format(
                reasoning="Samantha has 10 cups of flour and each batch requires 2.5 cups of flour.\n10 ÷ 2.5 = 4\nSo Samantha can make 4 complete batches of cookies with her 10 cups of flour.",
                answer="4 batches"
            )
        }
    ]
]