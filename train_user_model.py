import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, make_scorer
from transformers import pipeline
from load_data import load_data
from preprocess import preprocess_data

# Synthetic data generator setup
generator = pipeline('text-generation', model='gpt2')

def generate_synthetic_data(prompt, num_samples=50):
    """Generate synthetic text samples using GPT-2"""
    return [generator(prompt, max_length=50, truncation=True)[0]['generated_text']
            for _ in range(num_samples)]

def prepare_dataset(folder_name, n_relevant, n_irrelevant, use_synthetic):
    """Load and augment dataset with synthetic data if needed"""
    # Build folder paths
    relevant_folder = os.path.join(folder_name, "relevant")
    irrelevant_folder = os.path.join(folder_name, "irrelevant")
    
    # Load real data using updated load_data function
    try:
        texts, labels = load_data(relevant_folder, irrelevant_folder)
    except FileNotFoundError:
        print(f"Error: Missing 'relevant' or 'irrelevant' folder in {folder_name}")
        return [], []
    
    # Split into relevant/irrelevant using labels
    real_rel = [text for text, label in zip(texts, labels) if label == 1]
    real_irrel = [text for text, label in zip(texts, labels) if label == 0]
    
    # Generate synthetic data if enabled
    synthetic_rel, synthetic_irrel = [], []
    if use_synthetic:
        if len(real_rel) < n_relevant:
            needed = n_relevant - len(real_rel)
            synthetic_rel = generate_synthetic_data("Sustainable urban farming involves", needed)
        if len(real_irrel) < n_irrelevant:
            needed = n_irrelevant - len(real_irrel)
            synthetic_irrel = generate_synthetic_data("Cryptocurrency trading strategies", needed)
    
    # Combine and resample
    all_rel = real_rel + synthetic_rel
    all_irrel = real_irrel + synthetic_irrel
    
    rel_resampled = resample(all_rel, replace=True, n_samples=n_relevant, random_state=42) if all_rel else []
    irrel_resampled = resample(all_irrel, replace=True, n_samples=n_irrelevant, random_state=42) if all_irrel else []
    
    return rel_resampled, irrel_resampled


def train_and_evaluate(relevant_texts, irrelevant_texts):
    """Train model and return performance metrics"""
    texts = relevant_texts + irrelevant_texts
    labels = [1]*len(relevant_texts) + [0]*len(irrelevant_texts)
    
    if not texts:
        return np.nan, np.nan
    
    cleaned = preprocess_data(texts)
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(cleaned)
    y = np.array(labels)
    
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    
    model = LogisticRegression(class_weight='balanced')
    
    # Cross-validated metrics
    accuracy = cross_val_score(model, X, y, cv=3, scoring='accuracy').mean()
    f1 = cross_val_score(model, X, y, cv=3, scoring=make_scorer(f1_score)).mean()
    
    return accuracy, f1

def run_experiment(folder_name, use_synthetic, sample_ratios):
    """Run full experiment with multiple sample ratios"""
    results = []
    
    for rel, irrel in sample_ratios:
        print(f"\nTraining with {rel}R/{irrel}I samples...")
        relevant, irrelevant = prepare_dataset(folder_name, rel, irrel, use_synthetic)
        accuracy, f1 = train_and_evaluate(relevant, irrelevant)
        
        results.append({
            'relevant_samples': rel,
            'irrelevant_samples': irrel,
            'total_samples': rel + irrel,
            'class_ratio': rel/(rel+irrel),
            'accuracy': accuracy,
            'f1_score': f1
        })
    
    return pd.DataFrame(results)

def visualize_results(df, folder_name):
    """Generate and save performance visualizations"""
    plt.figure(figsize=(12, 6))
    
    # Accuracy vs Class Ratio
    plt.subplot(1, 2, 1)
    plt.scatter(df['class_ratio'], df['accuracy'], c=df['total_samples'], cmap='viridis')
    plt.xlabel('Proportion of Relevant Samples')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Class Balance')
    plt.colorbar(label='Total Samples')
    
    # F1 Score vs Sample Size
    plt.subplot(1, 2, 2)
    plt.scatter(df['total_samples'], df['f1_score'], c=df['class_ratio'], cmap='coolwarm')
    plt.xlabel('Total Samples')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Dataset Size')
    plt.colorbar(label='Class Ratio')
    
    plt.tight_layout()
    plt.savefig(f'performance_{folder_name}.png')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run class balance experiment')
    parser.add_argument('folder', type=str, help='Data folder name (e.g., 10.10)')
    parser.add_argument('synthetic', type=int, choices=[0, 1], 
                       help='Use synthetic data (0=No, 1=Yes)')
    parser.add_argument('--ratios', nargs='+', type=int,
                       help='Space-separated relevant:irrelevant ratios (e.g., 10 90 50 50)',
                       default=[10,10, 20,80, 50,50, 80,20])
    args = parser.parse_args()
    
    # Parse ratios into pairs
    ratio_pairs = [(args.ratios[i], args.ratios[i+1]) 
                   for i in range(0, len(args.ratios), 2)]
    
    # Run experiment
    results_df = run_experiment(args.folder, args.synthetic, ratio_pairs)
    print("\nExperiment Results:")
    print(results_df)
    
    # Save and visualize
    results_df.to_csv(f'results_{args.folder}.csv', index=False)
    visualize_results(results_df, args.folder)