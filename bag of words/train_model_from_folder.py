import os
import argparse
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from load_data import load_data
from preprocess import preprocess_data

def train_model(data_folder):
    # Load data
    relevant_folder = os.path.join(data_folder, 'relevant')
    irrelevant_folder = os.path.join(data_folder, 'irrelevant')
    
    texts, labels = load_data(relevant_folder, irrelevant_folder)
    
    # Preprocess data
    cleaned_texts = preprocess_data(texts)
    
    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(cleaned_texts)
    y = labels
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LogisticRegression(class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Save model and vectorizer
    model_filename = os.path.join(data_folder, f"{os.path.basename(data_folder)}_model.pkl")
    vectorizer_filename = os.path.join(data_folder, f"{os.path.basename(data_folder)}_vectorizer.pkl")
    joblib.dump(model, model_filename)
    joblib.dump(vectorizer, vectorizer_filename)
    
    # Evaluate model
    y_pred = model.predict(X_test)
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
    parser = argparse.ArgumentParser(description='Train a text classification model based on a folder in ./data')
    parser.add_argument('data_folder', type=str, help='Folder name in ./data (e.g., 25.25)')
    
    args = parser.parse_args()
    
    data_folder = os.path.join('./data', args.data_folder)
    
    if not os.path.exists(data_folder):
        print(f"Error: Folder {data_folder} does not exist.")
    else:
        train_model(data_folder)
