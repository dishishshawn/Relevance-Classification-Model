import os
import csv
import random
import argparse
from preprocess import preprocess_data

def filter_tweets(input_file, topic, relevant_count, irrelevant_count, output_folder):
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
            text = row[5]
            if topic.lower() in text.lower():
                relevant_tweets.append(row)
            else:
                irrelevant_tweets.append(row)
    
    relevant_sample = random.sample(relevant_tweets, min(relevant_count, len(relevant_tweets)))
    irrelevant_sample = random.sample(irrelevant_tweets, min(irrelevant_count, len(irrelevant_tweets)))
    
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

    print(f"Exported {len(relevant_sample)} relevant and {len(irrelevant_sample)} irrelevant tweets.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter tweets based on topic and export as text files.')
    parser.add_argument('topic', type=str, help='Topic to filter tweets by')
    parser.add_argument('relevant_count', type=int, help='Number of relevant tweets to export')
    parser.add_argument('irrelevant_count', type=int, help='Number of irrelevant tweets to export')
    parser.add_argument('output_folder', type=str, help='Output folder name (e.g., ./data/50.50)')
    
    args = parser.parse_args()
    
    input_file = './archive/training.1600000.processed.noemoticon.csv'
    
    filter_tweets(input_file, args.topic, args.relevant_count, args.irrelevant_count, args.output_folder)