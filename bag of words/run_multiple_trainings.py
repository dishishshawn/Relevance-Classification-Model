import os
import subprocess

def run_training(start_count, increment, max_count):
    current_count = start_count
    while current_count <= max_count:
        folder_name = f'{current_count}.{current_count}'
        data_folder = os.path.join('./data', folder_name)
        
        if not os.path.exists(data_folder):
            print(f"Error: Folder {data_folder} does not exist.")
            break
        
        command = [
            'python', 'train_model_from_folder.py', folder_name
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error running training for {folder_name}: {result.stderr}")
            break
        
        # Print a simple line of statistics
        output = result.stdout
        print(f"Statistics for {folder_name}:")
        print(output.split("Classification Report:")[1].split("Confusion Matrix:")[0].strip())
        
        current_count += increment

if __name__ == "__main__":
    start_count = 125
    increment = 25
    max_count = 1000
    
    run_training(start_count, increment, max_count)
