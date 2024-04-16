import os
import csv
import json

def parse_reviews(text_file):
    reviews = []
    with open(text_file, 'r', encoding='utf-8') as file:
        json_started = False
        json_content = ""
        for line in file:
            if line.strip() == "```json" and not json_started:
                json_started = True
                continue
            elif line.strip() == "```" and json_started:
                json_started = False
                break
            elif json_started:
                json_content += line
        try:
            reviews = json.loads(json_content)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON in file '{text_file}': {e}")
    return reviews

def process_directory(directory):
    all_reviews = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            reviews = parse_reviews(file_path)
            all_reviews.extend(reviews)
    return all_reviews

def save_to_csv(reviews, csv_file):
    with open(csv_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["product_name", "review", "tags"])
        writer.writeheader()
        for review in reviews:
            writer.writerow(review)

def main(input_directory, output_csv):
    reviews = process_directory(input_directory)
    save_to_csv(reviews, output_csv)

if __name__ == "__main__":
    main("/home/debbie/Dev/synthetic-data-generation/results/output/qwen1_5-32b-chat-q4_k_m.gguf/", "output.csv")  # Replace "input_directory" with your input directory path
