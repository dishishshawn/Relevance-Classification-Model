import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from modAL import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from preprocess import preprocess_data

def load_tweets(csv_file, num_samples=1000):
    df = pd.read_csv(csv_file)
    df = df.sample(n=num_samples, random_state=42)
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    return texts, labels

def train_bag_of_words_feedback(csv_file, initial_text, initial_label, num_samples=5000):  # Increased num_samples
    # Initial training data
    initial_texts = [initial_text]
    initial_labels = [initial_label]

    # Load additional data from CSV file
    texts, labels = load_tweets(csv_file, num_samples=num_samples)
    
    # Combine initial data with loaded data
    texts = initial_texts + texts
    labels = initial_labels + labels
    
    # Preprocess data
    cleaned_texts = preprocess_data(texts)
    
    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=5000)  # Increased max_features
    X = vectorizer.fit_transform(cleaned_texts)
    y = labels
    
    # Use LogisticRegression for better performance
    learner = ActiveLearner(
        estimator=LogisticRegression(max_iter=1000, solver='liblinear'),  # Tuned hyperparameters
        query_strategy=uncertainty_sampling,
        X_training=X, y_training=y
    )
    
    # Feedback loop
    while True:
        # Recommend new posts
        unlabeled_texts, _ = load_tweets(csv_file, num_samples=10)
        cleaned_unlabeled_texts = preprocess_data(unlabeled_texts)
        X_unlabeled = vectorizer.transform(cleaned_unlabeled_texts)
        
        # Query the most uncertain samples
        query_idx, query_instance = learner.query(X_unlabeled)
        
        # Get user feedback
        print("Recommended post:")
        print(unlabeled_texts[query_idx[0]])
        feedback = input("Is this post relevant? (y/n): ")
        if feedback.lower() == 'y':
            new_label = 1
        else:
            new_label = 0
        
        # Teach the model with the new label
        learner.teach(X_unlabeled[query_idx], [new_label])
        # Retrain with more iterations
        learner.estimator.max_iter += 100  # Incrementally increase iterations
        
        # Option to stop the feedback loop
        stop = input("Do you want to stop? (y/n): ")
        if stop.lower() == 'y':
            break
    
    # Save model and vectorizer
    data_folder = './data/feedback'
    model_filename = os.path.join(data_folder, 'feedback_model.pkl')
    vectorizer_filename = os.path.join(data_folder, 'feedback_vectorizer.pkl')
    joblib.dump(learner, model_filename)
    joblib.dump(vectorizer, vectorizer_filename)
    
    # Evaluate model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = learner.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    
    print(f"Model trained and saved as {model_filename}")
    print(f"Vectorizer saved as {vectorizer_filename}")
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(matrix)
    
    return accuracy, report, matrix

if __name__ == "__main__":
    csv_file = '/home/dishishshawn/Desktop/Relevance Classification/archive/training.1600000.processed.noemoticon.csv'
    initial_text = "Post about football"
    initial_label = 1  # 1 = relevant
    
    train_bag_of_words_feedback(csv_file, initial_text, initial_label, num_samples=1000)
