from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from load_data import load_data
from preprocess import preprocess_data

# Load and preprocess the data
relevant_folder = './data/relevant'
irrelevant_folder = './data/irrelevant'
texts, labels = load_data(relevant_folder, irrelevant_folder)

cleaned_texts = preprocess_data(texts)

# Vectorize the text
vectorizer = TfidfVectorizer(max_features=1000) # Only consider the top 1000 words
x = vectorizer.fit_transform(cleaned_texts)
y = labels

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(x_train, y_train)

# Make predictions
y_pred = model.predict(x_test)

# Evlaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))