import os
import subprocess

def run_filter(topic, start_count, increment, max_count, input_file):
    current_count = start_count
    while current_count <= max_count:
        folder_name = f'./data/{current_count}.{current_count}'
        command = [
            'python', 'download_and_filter.py', topic, str(current_count), str(current_count), folder_name
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error running filter for {current_count}.{current_count}: {result.stderr}")
            break
        
        # Check if the correct number of tweets were exported
        output = result.stdout
        if f"Exported {current_count} relevant and {current_count} irrelevant tweets." not in output:
            print(f"Stopped at {current_count}.{current_count} due to insufficient tweets.")
            break
        
        current_count += increment

if __name__ == "__main__":
    topic = 'football'  # Replace w/ topic
    start_count = 125
    increment = 25
    max_count = 1000
    input_file = './archive/training.1600000.processed.noemoticon.csv'
    
    run_filter(topic, start_count, increment, max_count, input_file)
