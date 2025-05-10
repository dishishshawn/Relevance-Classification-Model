import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, TFBertForSequenceClassification
from preprocess import preprocess_data

def load_test_data(test_folder):
    texts = []
    labels = []
    for label, subfolder in enumerate(['irrelevant', 'relevant']):
        folder_path = os.path.join(test_folder, subfolder)
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read().strip()
                    texts.append(preprocess_data([text])[0])  # Preprocess each text
                    labels.append(label)
    return texts, labels

def evaluate_sequential_model(test_texts, test_labels, model_path, tokenizer_path, results_csv):
    # Load the sequential model and tokenizer
    model = load_model(model_path)
    tokenizer = joblib.load(tokenizer_path)

    # Get feedback count from model if available
    feedback_count = getattr(model, 'feedback_count', 0)

    # Preprocess and tokenize the test data
    cleaned_texts = preprocess_data(test_texts)
    sequences = tokenizer.texts_to_sequences(cleaned_texts)
    X_test = pad_sequences(sequences, maxlen=100)

    # Predict and evaluate
    predictions = model.predict(X_test)
    predicted_labels = (predictions.flatten() > 0.5).astype(int)
    accuracy = accuracy_score(test_labels, predicted_labels)
    report = classification_report(test_labels, predicted_labels, target_names=["Not Relevant", "Relevant"])
    
    print(f"\nSequential Model Evaluation (after {feedback_count} iterations):")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

    # Save results with model type
    results = pd.DataFrame({
        "model": ["Sequential"],
        "feedback_count": [feedback_count],
        "accuracy": [accuracy],
        "timestamp": [pd.Timestamp.now()]
    })
    
    if os.path.exists(results_csv):
        existing_results = pd.read_csv(results_csv)
        updated_results = pd.concat([existing_results, results], ignore_index=True)
    else:
        updated_results = results
    
    # Sort by feedback count for better visualization
    updated_results = updated_results.sort_values('feedback_count')
    updated_results.to_csv(results_csv, index=False)

def evaluate_bag_of_words_model(test_texts, test_labels, model_path, vectorizer_path, results_csv):
    # Load the bag-of-words model and vectorizer
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    # Get feedback count from model
    feedback_count = getattr(model, 'feedback_count', 0)

    # Preprocess and vectorize the test data
    cleaned_texts = preprocess_data(test_texts)
    X_test = vectorizer.transform(cleaned_texts)

    # Predict and evaluate
    predicted_labels = model.predict(X_test)
    accuracy = accuracy_score(test_labels, predicted_labels)
    report = classification_report(test_labels, predicted_labels, target_names=["Not Relevant", "Relevant"])
    
    print(f"\nBag-of-Words Model Evaluation (after {feedback_count} iterations):")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

    # Save results with model type
    results = pd.DataFrame({
        "model": ["Bag-of-Words"],
        "feedback_count": [feedback_count],
        "accuracy": [accuracy],
        "timestamp": [pd.Timestamp.now()]
    })
    
    if os.path.exists(results_csv):
        existing_results = pd.read_csv(results_csv)
        updated_results = pd.concat([existing_results, results], ignore_index=True)
    else:
        updated_results = results
    
    # Sort by feedback count for better visualization
    updated_results = updated_results.sort_values('feedback_count')
    updated_results.to_csv(results_csv, index=False)

def evaluate_bert_model(test_texts, test_labels, model_path, tokenizer_path, results_csv):
    """Evaluate BERT model with enhanced metrics"""
    # Load the BERT model and tokenizer
    model = TFBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    # Get feedback count from labeled data
    labeled_data_path = os.path.join(os.path.dirname(model_path), "labeled_data.csv")
    if os.path.exists(labeled_data_path):
        labeled_data = pd.read_csv(labeled_data_path)
        feedback_count = len(labeled_data)
    else:
        feedback_count = 0

    # Preprocess and encode the test data
    encoded_data = tokenizer(
        test_texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="tf"
    )

    # Get predictions
    predictions = model.predict(dict(encoded_data))
    
    # Use probability threshold for classification
    probs = tf.nn.softmax(predictions.logits, axis=1).numpy()
    predicted_labels = (probs[:, 1] > 0.5).astype(int)
    
    # Calculate metrics with more detail
    accuracy = accuracy_score(test_labels, predicted_labels)
    report = classification_report(
        test_labels, 
        predicted_labels,
        target_names=["Not Relevant", "Relevant"],
        zero_division=0
    )
    
    print(f"\nALERT Model Evaluation (after {feedback_count} iterations):")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    
    # Save detailed results
    results = pd.DataFrame({
        "model": ["ALERT"],
        "feedback_count": [feedback_count],
        "accuracy": [accuracy],
        "timestamp": [pd.Timestamp.now()],
        "threshold": [0.5]
    })
    
    if os.path.exists(results_csv):
        existing_results = pd.read_csv(results_csv)
        updated_results = pd.concat([existing_results, results], ignore_index=True)
    else:
        updated_results = results
    
    # Sort by feedback count for better visualization
    updated_results = updated_results.sort_values(['model', 'feedback_count'])
    updated_results.to_csv(results_csv, index=False)

if __name__ == "__main__":
    # Update paths to include BERT model
    bert_model_path = "./data/feedback/bert_model"
    bert_tokenizer_path = "./data/feedback/bert_tokenizer"
    sequential_model_path = "./data/sequential_feedback/sequential_model.h5"
    sequential_tokenizer_path = "./data/sequential_feedback/tokenizer.pkl"
    bag_of_words_model_path = "./data/feedback/feedback_model.pkl"
    bag_of_words_vectorizer_path = "./data/feedback/feedback_vectorizer.pkl"
    test_data_folder = "./data/test"
    results_csv = "./data/model_comparison_results.csv"
    
    # Create data directory if it doesn't exist
    os.makedirs("./data", exist_ok=True)

    # Load test data
    if not os.path.exists(test_data_folder):
        print(f"Test data folder not found at {test_data_folder}. Please provide a valid test dataset.")
        exit()

    test_texts, test_labels = load_test_data(test_data_folder)

    # Evaluate all models
    if os.path.exists(bert_model_path) and os.path.exists(bert_tokenizer_path):
        evaluate_bert_model(test_texts, test_labels, bert_model_path, bert_tokenizer_path, results_csv)
    else:
        print("BERT model or tokenizer not found. Please train the ALERT model with active feedback first.")

    # Evaluate the sequential model
    if os.path.exists(sequential_model_path) and os.path.exists(sequential_tokenizer_path):
        evaluate_sequential_model(test_texts, test_labels, sequential_model_path, sequential_tokenizer_path, results_csv)
    else:
        print("Sequential model or tokenizer not found. Please train the sequential model with active feedback.")

    # Evaluate the bag-of-words model
    if os.path.exists(bag_of_words_model_path) and os.path.exists(bag_of_words_vectorizer_path):
        evaluate_bag_of_words_model(test_texts, test_labels, bag_of_words_model_path, bag_of_words_vectorizer_path, results_csv)
    else:
        print("Bag-of-words model or vectorizer not found. Please train the bag-of-words model with active feedback.")
