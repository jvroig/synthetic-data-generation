import os
import subprocess
import textwrap

###Machine specific settings
path_to_main  = "/Users/jvroig/Dev/llama.cpp"
n_threads     = "4"
model         = "/Users/Shared/llama-2-7b-chat/llama-2-7b-chat-q4_K.gguf"

executable        = "main"
max_input_tokens  = "512"
max_output_tokens = "512"


sentiments = ['strongly positive',
              'slightly positive',
              'strongly negative',
              'slightly negative',
              'neutral/indifferent',
              'mixed']
variants = ['normal',
            'typo',
            'no punctuation wall of text',
            'super mad all caps',
            'sarcastic review']
#FIXME: Where do we put the trolling / exaggeration reviews?

for sentiment in sentiments:
    for variant in variants:
        prompt = f"""
            Create a product review for a fictitious product. 
            The product can be a furniture or household appliance. 
            The review sentiment must be {sentiment}.
            The reviews must be a minimum of 3 sentences, with a maximum of 5 sentences.
            Answer in JSON format, where the first key is the "product_name", and the second key is "review".
        """

        if variant == "normal":
            pass
        elif variant == "typo":
            prompt += f"""
            Additionally, the review text should include a lot of common typographical errors.
            """
        elif variant == "no punctuation wall of text":
            prompt += f"""
            Additionally, the review text should have no punctuation and just be a wall of text.
            """
        elif variant == "super mad all caps":
            if sentiment == "strongly negative":
                prompt += f"""
                Additionally, the review text be in all caps and shows very strong emotions, as if the user submitting the review was super mad.
                """
            else:
                break
        elif variant == "sarcastic review":
            if sentiment == "strongly negative" or sentiment == "slightly negative":
                prompt += f"""
                Additionally, the review should be sarcastic.
                """
            else:
                break

        prompt += f"""
            Do not include any other extra information beyond that.
            Product Review:
        """


        prompt = textwrap.dedent(prompt)


        command = [f"{path_to_main}/{executable}", 
                "-m", model, 
                "-t", n_threads, 
                "-n", max_output_tokens,
                "-c", max_input_tokens,
                "--log-disable",
                "--prompt", prompt,]

        print(command)

        with open("out.txt", "a") as stdout_file, open("error.txt", "w") as stderr_file:
            process = subprocess.Popen(
                command,
                stderr=stderr_file,  # raw perf data go to our stderr_file (llama.cpp pipes perf data to stderr),
                text=True
            )
            process.wait()

print("Done!")