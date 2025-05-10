import os
import argparse
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocess import preprocess_data
from load_data import load_data

def test_model(data_folder, model_file, vectorizer_file):
    # Load data
    relevant_folder = os.path.join(data_folder, 'relevant')
    irrelevant_folder = os.path.join(data_folder, 'irrelevant')
    
    texts, labels = load_data(relevant_folder, irrelevant_folder)
    
    # Preprocess data
    cleaned_texts = preprocess_data(texts)
    
    # Load model and vectorizer
    model = joblib.load(model_file)
    vectorizer = joblib.load(vectorizer_file)
    
    # Vectorize text
    X = vectorizer.transform(cleaned_texts)
    y = labels
    
    # Predict
    y_pred = model.predict(X)
    
    # Evaluate model
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, output_dict=True)
    matrix = confusion_matrix(y, y_pred)
    
    print(f"Model: {model_file}")
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(matrix)
    
    return accuracy, report, matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test a text classification model based on a folder in ./data')
    parser.add_argument('data_folder', type=str, help='Folder name in ./data (e.g., test)')
    parser.add_argument('model_file', type=str, help='Path to the model file (e.g., 25.25_model.pkl)')
    parser.add_argument('vectorizer_file', type=str, help='Path to the vectorizer file (e.g., 25.25_vectorizer.pkl)')
    
    args = parser.parse_args()
    
    data_folder = os.path.join('./data', args.data_folder)
    
    if not os.path.exists(data_folder):
        print(f"Error: Folder {data_folder} does not exist.")
    elif not os.path.exists(args.model_file):
        print(f"Error: Model file {args.model_file} does not exist.")
    elif not os.path.exists(args.vectorizer_file):
        print(f"Error: Vectorizer file {args.vectorizer_file} does not exist.")
    else:
        test_model(data_folder, args.model_file, args.vectorizer_file)
