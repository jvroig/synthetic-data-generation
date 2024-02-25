import os
import json
from datetime import datetime
import subprocess
import textwrap
from typing import Dict, List
from faker import Faker

import boto3
from botocore.config import Config
my_config = Config(region_name = 'us-east-2')

#####STARDATES
def gregorian_to_stardate(year, month, day):
    """
    Convert Gregorian date to Star Trek stardate.
    Formula adapted from the Star Trek: The Next Generation Technical Manual.
    """
    start_year = 2323  # Reference year for stardates
    stardate = 1000 + (year - start_year) + ((month - 1) / 12.0) + (day / 365.0)
    return round(stardate, 2)

def get_current_stardate():
    """
    Get the current date and convert it into Star Trek stardate.
    """
    current_date = datetime.now()
    year = current_date.year
    month = current_date.month
    day = current_date.day
    return gregorian_to_stardate(year, month, day)

current_stardate = get_current_stardate()


def query_endpoint(payload):
    client = boto3.client("sagemaker-runtime", config=my_config)
    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload).encode("utf-8"),
        CustomAttributes="accept_eula=true",
    )
    response = response["Body"].read().decode("utf8")
    response = json.loads(response)
    return response

def query_mixtral(payload):
    client = boto3.client("runtime.sagemaker")
    response = client.invoke_endpoint(
        EndpointName=endpoint_name, 
        ContentType="application/json", 
        Body=json.dumps(payload).encode("utf-8")
    )
    response = response["Body"].read().decode("utf8")
    response = json.loads(response)
    return response

def format_instructions(instructions: List[Dict[str, str]]) -> List[str]:
    """Format instructions where conversation roles must alternate user/assistant/user/assistant/..."""
    prompt: List[str] = []
    for user, answer in zip(instructions[::2], instructions[1::2]):
        prompt.extend(["<s>", "[INST] ", (user["content"]).strip(), " [/INST] ", (answer["content"]).strip(), "</s>"])
    prompt.extend(["<s>", "[INST] ", (instructions[-1]["content"]).strip(), " [/INST] "])
    return "".join(prompt)



####Settings
max_input_tokens  = "512"
max_output_tokens = "512"

# sentiments = ['strongly positive',
#               'slightly positive',
#               'strongly negative',
#               'slightly negative',
#               'neutral or indifferent',
#               'mixed']

sentiments = [
    'strongly positive',
    'mostly positive',
    'slightly positive',
    'mixed but leans positive',
    'strongly negative',
    'mostly negative',
    'slightly negative',
    'mixed but leans negative',
]
negative_sentiments = [
    'strongly negative',
    'mostly negative',
    'slightly negative',
    'mixed but leans negative',
]

review_length = [
    '1',
    '2',
    '3',
    '10',
]

variants = ['normal',
            'typo',
            'sarcastic']

personas = [
    'a tech enthusiast',
    'a housewife',
    'a social media influencer',
    'an asshole',
    'a Silicon Valley tech bro',
    'a professional corporate worker',
    'a teenager'
]


temperature = 0.8
runs = 3
offset = 1 #default to zero for new experiments

####Output log settings
output_basedir = "results/output/"
error_basedir  = "results/perf/"
log_basedir    = "results/log/"
log_filename   = "run_" + datetime.now().strftime('%Y-%m-%d_%Hh%Mm')
os.makedirs(os.path.dirname(log_basedir), exist_ok=True)


#FIXME: Where do we put the trolling / exaggeration reviews?
#NOTE:
#   These variants were removed from inference loop, will instead be 
#   local post-processing of `normal` output:
#       - 'no punctuation wall of text',
#       - 'super mad all caps'

####Sagemaker
#model_family = "llama-2"
#model = "Llama-2-70B-chat"
#endpoint_name = "jumpstart-dft-meta-textgeneration-llama-2-70b-f"
model_family = "mistral"
model = "Mixtral-8x7b-instruct"
endpoint_name = "jumpstart-dft-hf-llm-mixtral-8x7b-instruct"

####Experiment proper
products = [
    'Android Phone',
    'Sofa',
    'Screwdriver',
    'Microwave',
    'Smart TV',
    'GPU',
    'Electric Kettle',
    'Queen-Sized Bed',
    'Coffee Table',
    'Laptop',
]

fake = Faker()
for run in range(1+offset, runs + 1):
    #Create a fake product name
    company = fake.company()
    prod_index = (run - 1) % len(products)
    product = products[prod_index]
    for sentiment in sentiments:
        for num_sentences in review_length:
            for persona in personas:
                for variant in variants:

                    opener = ''
                    closer = ''

                    prompt = f"""
                        Create a product review for this product: {company} {product}.              
                        The review sentiment must be {sentiment}.
                        The review must be {num_sentences} sentences long.
                        The review must be written from the perspective of {persona}.
                        Answer in JSON format, where the first key is "product_name", and the second key is "review".
                        """

                    if variant == "normal":
                        pass
                    if variant == "star trek":
                        #NOTE: opener and closer can be used to expand "persona" prompting in the future
                        opener = f"""
                        «Command, we need you to plot a course through this
                        turbulence and locate the source of the anomaly. Use all
                        available data and your expertise to guide us through
                        this challenging situation.»
                        """
                        closer = f"""
                        Captain’s Log, Stardate {current_stardate}: We have
                        successfully plotted a course through the turbulence
                        and are now approaching the source of the anomaly.
                        """
                    elif variant == "typo":
                        prompt += f"""
                        Additionally, the review text should include a few common typographical errors.
                        """
                    elif variant == "sarcastic":
                        if sentiment in negative_sentiments:
                            prompt += f"""
                            Additionally, the review should be sarcastic.
                            """
                        else:
                            break

                    prompt += f"""
                        Do not include any other extra information beyond that.
                        Product Review:"""
                    
                    opener = textwrap.dedent(opener)
                    closer = textwrap.dedent(closer)
                    prompt = textwrap.dedent(prompt)
                    prompt = opener + prompt + closer

                    stdout = []
                    stdout.append("*******************")
                    stdout.append(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    stdout.append(f"Now running: {model}")
                    stdout.append(f"Temperature: {temperature}")
                    stdout.append(f"Run: {str(run)}")
                    stdout.append(f"Sentiment: {sentiment}")
                    stdout.append(f"Sentences: {num_sentences}")
                    stdout.append(f"Persona: {persona}")
                    stdout.append(f"Variant: {variant}")
                    
                    print(stdout[0])
                    print(stdout[1])
                    print(stdout[2])
                    print(stdout[3])
                    print(stdout[4])
                    print(stdout[5])
                    print(stdout[6])
                    print(stdout[7])
                    print(stdout[8])

                    should_execute = True
                    if persona == 'an asshole' and sentiment not in negative_sentiments:
                        should_execute = False

                    if should_execute:
                        if model_family == "llama-2":
                            payload = {
                                "inputs": [[{"role": "user", "content": prompt}]], 
                                "parameters": {"max_new_tokens": int(max_output_tokens), "top_p": 0.9, "temperature": float(temperature)}
                            }
                        elif model_family == "mistral":
                            instructions = [{"role": "user", "content": prompt}]
                            prompt = format_instructions(instructions)
                            payload = {
                                "inputs": prompt, 
                                "parameters": {"max_new_tokens": int(max_output_tokens), "top_p": 0.9, "temperature": float(temperature)}
                            }                                
                        else:
                            print("Invalid LLM family given!")
                            exit()

                        result = query_endpoint(payload)[0]

                        if model_family == "llama-2":
                            answer = f"{result['generation']['content']}"
                        elif model_family == "mistral":
                            answer = f"{result['generated_text']}"
                        
                        # senti_folder = sentiment.replace(" ", "_")
                        # variant_folder = variant.replace(" ", "_")
                        # persona_folder = persona.replace(" ", "_")
                        # sentence_folder = "s" + str(num_sentences)
                        # output_dir = output_basedir + model + "/" + senti_folder + "/" + sentence_folder + "/" + persona_folder + "/" + variant_folder + "/"
                        # error_dir  = error_basedir + model + "/" + senti_folder + "/" + sentence_folder + "/" + persona_folder + "/" + variant_folder + "/"
                        # out_file =  f"run{str(run)}.txt"

                        output_dir = output_basedir + model + "/"
                        error_dir  = error_basedir + model + "/"
                        os.makedirs(os.path.dirname(output_dir), exist_ok=True)        
                        os.makedirs(os.path.dirname(error_dir), exist_ok=True)        

                        senti_path   = sentiment.replace(" ", "-")
                        persona_path = persona.replace(" ", "-")
                        variant_path = variant.replace(" ", "-")
                        out_file = f"{senti_path}_{persona_path}_{variant_path}_s{str(num_sentences)}_run{str(run)}.txt" 

                        with open(output_dir+out_file, "w", encoding="utf-8") as stdout_file:
                            stdout_file.write("Prompt:" + "\n")
                            stdout_file.write(prompt + "\n")
                            stdout_file.write("Response:" + "\n")
                            stdout_file.write(answer + "\n")

                        stdout.append("Done!")
                        stdout.append("-------------------")
                        print(stdout[9])
                        print(stdout[10])


                        with open(log_basedir+log_filename, "a", encoding="utf-8") as log_file:
                            log_file.write(stdout[0] + "\n")
                            log_file.write(stdout[1] + "\n")
                            log_file.write(stdout[2] + "\n")
                            log_file.write(stdout[3] + "\n")
                            log_file.write(stdout[4] + "\n")
                            log_file.write(stdout[5] + "\n")
                            log_file.write(stdout[6] + "\n")
                            log_file.write(stdout[7] + "\n")
                            log_file.write(stdout[8] + "\n")
                            log_file.write(stdout[9] + "\n")
                            log_file.write(stdout[10] + "\n")

print("Done!")