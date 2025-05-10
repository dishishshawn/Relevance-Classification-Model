import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

# Load data from folders
def load_data_from_folders(folder_path):
    texts = []
    labels = []

    irrelevant_path = os.path.join(folder_path, "irrelevant")
    relevant_path = os.path.join(folder_path, "relevant")

    # Load irrelevant posts (label 0)
    if os.path.exists(irrelevant_path):
        for file_name in os.listdir(irrelevant_path):
            file_path = os.path.join(irrelevant_path, file_name)
            if file_name.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as file:
                    texts.append(file.read().strip())
                    labels.append(0)

    # Load relevant posts (label 1)
    if os.path.exists(relevant_path):
        for file_name in os.listdir(relevant_path):
            file_path = os.path.join(relevant_path, file_name)
            if file_name.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as file:
                    texts.append(file.read().strip())
                    labels.append(1)

    return texts, labels

# Tokenize texts
def tokenize_texts(tokenizer, texts, max_length=128):
    if not texts:
        raise ValueError("No texts provided for tokenization.")
    return tokenizer(
        texts,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="tf"
    )

# Train model
def train_model(model, tokenizer, labeled_data):
    texts = labeled_data['text'].tolist()
    labels = labeled_data['label'].tolist()

    # Calculate class weights
    n_samples = len(labels)
    n_relevant = sum(labels)
    n_irrelevant = n_samples - n_relevant

    class_weight = {
        0: n_samples / (2 * n_irrelevant),
        1: n_samples / (2 * n_relevant)
    }

    # Prepare dataset
    inputs = tokenize_texts(tokenizer, texts)
    dataset = tf.data.Dataset.from_tensor_slices((
        dict(inputs),
        labels
    )).shuffle(1000).batch(16)

    # Train
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.fit(
        dataset,
        epochs=1,
        class_weight=class_weight
    )

# Feedback loop
def feedback_loop(model, tokenizer, csv_file, labeled_data_path, results_csv, test_texts, test_labels, batch_size=32):
    """
    Feedback loop to iteratively improve the model by finding relevant posts with minimal inputs.
    """
    try:
        # Load tweets from the CSV file
        df = pd.read_csv(csv_file, encoding='ISO-8859-1', names=['target', 'id', 'date', 'flag', 'user', 'text'])
        texts = df['text'].tolist()

        # Load existing labeled data
        if os.path.exists(labeled_data_path):
            labeled_data = pd.read_csv(labeled_data_path)
            texts = [text for text in texts if text not in labeled_data['text'].tolist()]
        else:
            labeled_data = pd.DataFrame(columns=['text', 'label'])

        # Track class balance
        class_counts = {'relevant': 0, 'irrelevant': 0}

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenize_texts(tokenizer, batch_texts)
            predictions = model.predict(dict(inputs)).logits
            confidences = tf.nn.softmax(predictions, axis=1).numpy()

            # Calculate uncertainty scores
            uncertainty = 1 - np.max(confidences, axis=1)  # Higher uncertainty = lower confidence
            relevance_prob = confidences[:, 1]  # Probability of being relevant

            # Combine uncertainty and relevance for scoring
            scores = uncertainty + 0.5 * relevance_prob  # Adjust weight for relevance if needed

            # Select the most informative example (highest score)
            top_idx = np.argmax(scores)
            text = batch_texts[top_idx]
            confidence = confidences[top_idx]

            print(f"\nPost: {text}")
            print(f"Confidence: {confidence[1]:.2f} relevant, {confidence[0]:.2f} irrelevant")
            print(f"Current balance - Relevant: {class_counts['relevant']}, Irrelevant: {class_counts['irrelevant']}")

            # Get feedback from the user
            feedback = input("Is this relevant? (y/n/stop): ").lower()
            if feedback == 'stop':
                return

            if feedback in ['y', 'n']:
                new_label = 1 if feedback == 'y' else 0
                class_counts['relevant' if new_label == 1 else 'irrelevant'] += 1

                # Add new data to labeled dataset
                new_data = pd.DataFrame({'text': [text], 'label': [new_label]})
                labeled_data = pd.concat([labeled_data, new_data], ignore_index=True)
                labeled_data.to_csv(labeled_data_path, index=False)

                # Retrain the model every 10 examples
                if len(labeled_data) % 10 == 0:
                    print("\nRetraining model on updated dataset...")
                    train_model(model, tokenizer, labeled_data)

                # Evaluate the model every 20 examples
                if len(labeled_data) % 20 == 0:
                    print("\nEvaluating model on test dataset...")
                    test_inputs = tokenize_texts(tokenizer, test_texts)
                    test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_inputs), test_labels)).batch(16)
                    loss, accuracy = model.evaluate(test_dataset, verbose=0)
                    print(f"Test Accuracy after {len(labeled_data)} feedback iterations: {accuracy:.4f}")

                    # Save results to CSV
                    results = pd.DataFrame({
                        "feedback_count": [len(labeled_data)],
                        "accuracy": [accuracy],
                        "timestamp": [pd.Timestamp.now()]
                    })

                    if os.path.exists(results_csv):
                        existing_results = pd.read_csv(results_csv)
                        updated_results = pd.concat([existing_results, results], ignore_index=True)
                    else:
                        updated_results = results

                    updated_results.to_csv(results_csv, index=False)
    except KeyboardInterrupt:
        print("\nFeedback loop interrupted. Saving progress...")
        labeled_data.to_csv(labeled_data_path, index=False)
        print("Progress saved. Exiting.")

# Main function
def main():
    labeled_data_path = "data/feedback/labeled_data.csv"
    results_csv = "data/feedback/bert_feedback_results.csv"
    model_path = "data/feedback/bert_model"
    tokenizer_path = "data/feedback/bert_tokenizer"
    csv_file = "archive/training.1600000.processed.noemoticon.csv"
    test_data_path = "data/test/"

    os.makedirs(os.path.dirname(labeled_data_path), exist_ok=True)  # Ensure the directory exists

    # Load or initialize the model and tokenizer
    if os.path.exists(model_path) and os.path.exists(tokenizer_path):
        print("Loading saved model and tokenizer...")
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        model = TFBertForSequenceClassification.from_pretrained(model_path)
    else:
        print("No saved model found. Fine-tuning a new model...")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

        # Load labeled data
        if os.path.exists(labeled_data_path):
            labeled_data = pd.read_csv(labeled_data_path)
        else:
            labeled_data = pd.DataFrame(columns=['text', 'label'])

        if labeled_data.empty:
            print("No labeled data available. Please add some initial labeled examples.")
            return

        train_model(model, tokenizer, labeled_data)

        # Save the model and tokenizer
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(tokenizer_path)

    # Load test data
    test_texts, test_labels = load_data_from_folders(test_data_path)
    if len(test_texts) == 0 or len(test_labels) == 0:
        print("No test data available. Please provide a valid test dataset.")
        return

    # Evaluate on a subset of the test dataset (e.g., 100 samples)
    subset_size = min(100, len(test_texts))
    test_texts_subset = test_texts[:subset_size]
    test_labels_subset = test_labels[:subset_size]

    test_inputs = tokenize_texts(tokenizer, test_texts_subset)
    test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_inputs), test_labels_subset)).batch(16)
    loss, accuracy = model.evaluate(test_dataset, verbose=0)

    # Start the feedback loop
    feedback_loop(model, tokenizer, csv_file, labeled_data_path, results_csv, test_texts, test_labels)

if __name__ == "__main__":
    main()