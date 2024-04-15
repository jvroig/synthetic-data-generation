import os
import subprocess
import textwrap

###Machine specific settings
path_to_main  = "/home/debbie/Dev/llama.cpp"
n_threads     = "6"
# model         = "/mnt/fuji/gguf/c4ai-command-r-v01-Q4_K_M.gguf"
model         = "/mnt/fuji/gguf/mistral/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"

executable        = "main"
max_input_tokens  = "1024"
max_output_tokens = "512"


# tags = ['quality']
# products = ['Sofa']
tags = ['quality',
              'performance',
              'value',
              'design',
              'none',
              'non-review',
              'performance, quality',
              'quality, value',
              'design, value',
              'design, performance',
              'design, quality',
              ]
products = ['Sofa',
            'Smart TV',
            'Android Phone',
            'Coffee Maker',
            'Screwdriver']

for tag in tags:
    for product in products:
        prompt = f"""
            ### INSTRUCTIONS ###
            You're tasked with generating synthetic datasets for fine-tuning a language model on customer reviews for a product. Each review should be categorized based on its content, indicating zero, one, or multiple tags/categories:

            Quality: Evaluate if the review primarily focuses on craftsmanship, durability, or overall build quality. Look for mentions of materials, construction, defects, or flaws encountered.

            Performance: Determine if the review emphasizes functionality, efficiency, or how well the product performs its intended tasks. Pay attention to mentions of speed, accuracy, features, and any performance issues experienced.

            Value: Identify reviews that highlight the product's worth relative to its price. Consider references to cost-effectiveness, affordability, comparisons with similar products, and whether the product delivers promised benefits for the price paid.

            Design: Look for reviews discussing aesthetic appeal, ergonomics, user interface, or overall design aspects. Note any compliments or criticisms regarding appearance, ease of use, and user experience design.

            Non-review: Recognize reviews indicating the product hasn't been tried or whose content is solely "I bought this," "bought this at sale," or "bought this for my wife" with no additional information or customer sentiment expressed.

            A review might receive multiple tags/categories if its content covers multiple aspects of the product experience.

            Your task is to generate diverse synthetic datasets adhering to these guidelines, providing nuanced insights into customer feedback on product quality, performance, value, and design.\n
            """

        prompt += f"""
            ### INPUT ####
            Create three sample reviews for a product: {product}.
            The tag should be: {tag}.

            ### OUTPUT FORMAT ####
            Answer in csv format, where the first column is the product name, the second column is the review text and the third column contains the tags.
            Do not include any other extra information beyond that.

            ### LLM ANSWER ###"""

        prompt = textwrap.dedent(prompt)

        command = [f"{path_to_main}/{executable}", 
                "-m", model, 
                "-t", n_threads, 
                "-n", max_output_tokens,
                "-c", max_input_tokens,
                "--log-disable",
                "--prompt", prompt,
                "-ngl", '20',
                ]

        # print(command)

        with open("out.txt", "a") as stdout_file, open("error.txt", "w") as stderr_file:
            process = subprocess.Popen(
                command,
                stderr=stderr_file,  # raw perf data go to our stderr_file (llama.cpp pipes perf data to stderr),
                text=True
            )
            process.wait()
            stdout_file.flush()

print("\nDone!\n")