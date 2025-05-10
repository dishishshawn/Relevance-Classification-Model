import os
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from preprocess import preprocess_data

def load_tweets(csv_file, num_samples=1000):
    df = pd.read_csv(csv_file, 
                     encoding='ISO-8859-1',
                     names=['target', 'id', 'date', 'flag', 'user', 'text'])
    df = df.sample(n=num_samples, random_state=42)
    texts = df['text'].tolist()
    labels = df['target'].tolist()
    return texts, labels

def find_relevant_samples(model, sequences, texts, threshold=0.5, min_samples_to_check=5000):
    """Keep searching until finding a post with high confidence"""
    predictions = model.predict(sequences)
    relevance_scores = predictions.flatten()
    
    # Sort all by relevance score
    sorted_indices = np.argsort(relevance_scores)[::-1]
    
    # Find any samples above threshold
    relevant_indices = sorted_indices[relevance_scores[sorted_indices] >= threshold]
    
    if len(relevant_indices) > 0:
        # Return most relevant that meets threshold
        best_idx = relevant_indices[0]
        return [best_idx], len(sorted_indices)
    else:
        # If no samples meet threshold, return None to continue searching
        return None, len(sorted_indices)
    

def train_sequential_feedback(csv_file, num_samples=5000, threshold=0.5):  # Increased num_samples
    # Setup paths for sequential-specific data
    data_folder = './data/sequential_feedback'  # Changed from ./data/feedback
    model_path = f'{data_folder}/sequential_model.h5'
    tokenizer_path = f'{data_folder}/tokenizer.pkl'
    labeled_data_path = f'{data_folder}/labeled_data.csv'  # Different file for sequential feedback
    
    # Initial relevant example
    initial_text = "I love watching football! The game was amazing, great match today!"
    initial_label = 1
    
    # Create data directory if it doesn't exist
    os.makedirs(data_folder, exist_ok=True)
    
    # Load or create model and tokenizer
    if os.path.exists(model_path) and os.path.exists(tokenizer_path):
        model = load_model(model_path)
        tokenizer = joblib.load(tokenizer_path)
    else:
        # Initialize tokenizer and model
        tokenizer = Tokenizer(num_words=10000)  # Increased vocabulary size
        model = Sequential([
            Embedding(10000, 128, input_length=100),  # Adjusted for larger vocabulary
            LSTM(128, dropout=0.3, recurrent_dropout=0.3),  # Increased dropout
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train on initial example
        tokenizer.fit_on_texts([initial_text])
        initial_sequence = tokenizer.texts_to_sequences([initial_text])
        X_initial = pad_sequences(initial_sequence, maxlen=100)
        model.fit(X_initial, np.array([initial_label]), epochs=3, verbose=0)  # Increased epochs
    
    # Load tweets
    texts, _ = load_tweets(csv_file, num_samples)
    
    # Load existing labeled data
    if os.path.exists(labeled_data_path):
        labeled_data = pd.read_csv(labeled_data_path)
        texts = [text for text in texts if text not in labeled_data['text'].tolist()]
    else:
        labeled_data = pd.DataFrame(columns=['text', 'label'])
    
    # Preprocess and tokenize texts
    cleaned_texts = preprocess_data(texts)
    tokenizer.fit_on_texts(cleaned_texts)
    sequences = tokenizer.texts_to_sequences(cleaned_texts)
    X_pool = pad_sequences(sequences, maxlen=100)
    
    # Feedback loop
    while True:
        if len(texts) == 0:
            print("\nAll samples have been processed.")
            break
        
        print("\nSearching for relevant posts...")
        print(f"Checking posts for relevance (minimum confidence: {threshold:.0%})...")
        
        # Keep searching until finding a relevant post
        relevant_indices = None
        while relevant_indices is None and len(texts) > 0:
            # Get predictions
            X_pool = pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=100)
            relevant_indices, num_checked = find_relevant_samples(model, X_pool, texts, threshold=threshold)
            
            if relevant_indices is None:
                print(f"No posts found meeting confidence threshold. Continuing search...")
                continue
        
        if relevant_indices is None:
            print("No more posts to evaluate.")
            break
            
        best_idx = relevant_indices[0]
        pred_prob = model.predict(X_pool[best_idx:best_idx+1])[0][0]
        
        # Show prediction
        print(f"\nPost: {texts[best_idx]}")
        print(f"Confidence of relevance: {pred_prob:.2%}")
        print(f"Checked {num_checked} posts to find this one.")
        feedback = input("Relevant? (y/n/stop): ").lower()
        
        if feedback == 'stop':
            break
        
        if feedback in ['y', 'n']:
            # Process feedback
            new_label = 1 if feedback == 'y' else 0
            X_train = X_pool[best_idx:best_idx+1]
            y_train = np.array([new_label])
            
            # Update model with more epochs
            model.fit(X_train, y_train, epochs=5, verbose=0)  # Increased epochs for feedback
            
            # Save to labeled data
            new_data = pd.DataFrame({'text': [texts[best_idx]], 'label': [new_label]})
            labeled_data = pd.concat([labeled_data, new_data], ignore_index=True)
            
            # Remove processed text
            texts.pop(best_idx)
            X_pool = np.delete(X_pool, best_idx, axis=0)
            
            # Save progress
            os.makedirs(data_folder, exist_ok=True)
            labeled_data.to_csv(labeled_data_path, index=False)
            model.save(model_path)
            joblib.dump(tokenizer, tokenizer_path)
            
            # Print stats
            relevant_count = len(labeled_data[labeled_data['label'] == 1])
            irrelevant_count = len(labeled_data[labeled_data['label'] == 0])
            print(f"Current stats - Relevant: {relevant_count}, Not Relevant: {irrelevant_count}")

if __name__ == "__main__":
    csv_file = 'archive/training.1600000.processed.noemoticon.csv'
    train_sequential_feedback(csv_file, num_samples=1000, threshold=0.5)
