import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Clean the text by removing URLS, non-alphabetic characters, and stopwords
def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Removing URLS
    text = re.sub(r'[^a-zA-Z]', ' ', text) # Removing non-alphabetic
    text = text.lower() # Convert to lowercase to ensure uniformity because model is case-sensitive
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words]) # Removing stopwords
    return text

# Preprocess the data
def preprocess_data(texts):
    cleaned_texts = [clean_text(text) for text in texts]
    return cleaned_texts