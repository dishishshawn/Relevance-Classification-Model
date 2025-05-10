import os
import subprocess
import pandas as pd

def run_tests(test_data_folder, start_count, increment, max_count):
    results = []
    current_count = start_count
    while current_count <= max_count:
        folder_name = f'{current_count}.{current_count}'
        model_file = os.path.join('./data', folder_name, f'{folder_name}_model.pkl')
        vectorizer_file = os.path.join('./data', folder_name, f'{folder_name}_vectorizer.pkl')
        
        if not os.path.exists(model_file) or not os.path.exists(vectorizer_file):
            print(f"Error: Model or vectorizer file for {folder_name} does not exist.")
            break
        
        command = [
            'python', 'test_model.py', test_data_folder, model_file, vectorizer_file
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error running test for {folder_name}: {result.stderr}")
            break
        
        # Collect the evaluation statistics
        output = result.stdout
        accuracy_line = [line for line in output.split('\n') if "Accuracy:" in line][0]
        accuracy = float(accuracy_line.split(":")[1].strip())
        report_start = output.index("Classification Report:") + len("Classification Report:")
        report_end = output.index("Confusion Matrix:")
        report = output[report_start:report_end].strip()
        
        results.append({
            'model': folder_name,
            'accuracy': accuracy,
            'classification_report': report
        })
        
        current_count += increment
    
    # Export results to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv('model_evaluation_results.csv', index=False)
    print("Evaluation results saved to model_evaluation_results.csv")

if __name__ == "__main__":
    test_data_folder = 'test'  # Replace with your test data folder
    start_count = 25
    increment = 25
    max_count = 1000
    
    run_tests(test_data_folder, start_count, increment, max_count)
