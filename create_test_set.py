import os
import csv
import random
import argparse
from preprocess import preprocess_data

def create_test_set(input_file, topics, test_size, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    relevant_folder = os.path.join(output_folder, 'relevant')
    irrelevant_folder = os.path.join(output_folder, 'irrelevant')
    
    if not os.path.exists(relevant_folder):
        os.makedirs(relevant_folder)
    
    if not os.path.exists(irrelevant_folder):
        os.makedirs(irrelevant_folder)
    
    relevant_tweets = []
    irrelevant_tweets = []
    
    with open(input_file, 'r', encoding='latin-1') as file:
        reader = csv.reader(file)
        for row in reader:
            text = row[5].lower()
            if any(topic.lower() in text for topic in topics):
                relevant_tweets.append(row)
            else:
                irrelevant_tweets.append(row)
    
    relevant_sample_size = min(test_size // 2, len(relevant_tweets))
    irrelevant_sample_size = min(test_size // 2, len(irrelevant_tweets))
    
    if relevant_sample_size < test_size // 2 or irrelevant_sample_size < test_size // 2:
        print(f"Insufficient tweets found. Found {relevant_sample_size} relevant and {irrelevant_sample_size} irrelevant tweets.")
    
    relevant_sample = random.sample(relevant_tweets, relevant_sample_size)
    irrelevant_sample = random.sample(irrelevant_tweets, irrelevant_sample_size)
    
    relevant_texts = [tweet[5] for tweet in relevant_sample]
    irrelevant_texts = [tweet[5] for tweet in irrelevant_sample]
    
    cleaned_relevant_texts = preprocess_data(relevant_texts)
    cleaned_irrelevant_texts = preprocess_data(irrelevant_texts)
    
    for tweet, cleaned_text in zip(relevant_sample, cleaned_relevant_texts):
        tweet_id = tweet[1]  
        with open(os.path.join(relevant_folder, f'{tweet_id}.txt'), 'w', encoding='utf-8') as file:
            file.write(cleaned_text)
    
    for tweet, cleaned_text in zip(irrelevant_sample, cleaned_irrelevant_texts):
        tweet_id = tweet[1] 
        with open(os.path.join(irrelevant_folder, f'{tweet_id}.txt'), 'w', encoding='utf-8') as file:
            file.write(cleaned_text)

    print(f"Created test set with {len(relevant_sample)} relevant and {len(irrelevant_sample)} irrelevant tweets.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a test set for tweet classification.')
    parser.add_argument('topics', nargs='+', type=str, help='List of topics to filter tweets by')
    parser.add_argument('test_size', type=int, help='Total number of tweets in the test set')
    parser.add_argument('output_folder', type=str, help='Output folder name (e.g., ./data/test)')
    
    args = parser.parse_args()
    
    input_file = './archive/training.1600000.processed.noemoticon.csv'
    
    create_test_set(input_file, args.topics, args.test_size, args.output_folder)
