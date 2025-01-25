import os
import glob
import re
from datetime import datetime
import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def parse_timestamp(line):
    # Extract timestamp from line and convert to datetime object
    match = re.match(r'\[ (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \]', line)
    if match:
        return datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')
    return None

def clean_numeric(value):
    # Remove commas from numeric strings
    return value.replace(',', '') if value else ''

def parse_training_line(line):
    # Extract all relevant metrics from a training line
    timestamp = parse_timestamp(line)
    
    # Extract other metrics using regex
    epoch = re.search(r'Epoch \[(\d+)\]', line)
    step = re.search(r'Step \[(\d+(?:,\d+)*) / (\d+(?:,\d+)*)\]', line)
    batch = re.search(r'Batch \[(\d+(?:,\d+)*) / (\d+(?:,\d+)*)\]', line)
    lr = re.search(r'Lr: \[([^\]]+)\]', line)
    loss = re.search(r'Avg Loss \[([^\]]+)\]', line)
    rank_corr = re.search(r'Rank Corr\.: \[([^\]]+)\]', line)
    examples = re.search(r'Examples: (\d+(?:,\d+)*)', line)
    # Updated patterns to handle comma-separated numbers
    time_ms = re.search(r'(\d+(?:,\d+)*\.\d+) ms', line)
    total_s = re.search(r'(?:0*)(\d+(?:,\d+)*\.\d+) s total', line)
    
    # Get the base values
    total_time = float(clean_numeric(total_s.group(1))) if total_s else 0.0
    epoch_num = int(epoch.group(1)) if epoch else 0
    
    # Add 1000 seconds for each complete cycle (when counter resets)
    total_time += epoch_num * 1000
    
    return {
        'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S') if timestamp else '',
        'epoch': clean_numeric(epoch.group(1)) if epoch else '',
        'step': clean_numeric(step.group(1)) if step else '',
        'total_steps': clean_numeric(step.group(2)) if step else '',
        'batch': clean_numeric(batch.group(1)) if batch else '',
        'total_batches': clean_numeric(batch.group(2)) if batch else '',
        'learning_rate': lr.group(1) if lr else '',
        'avg_loss': loss.group(1) if loss else '',
        'rank_correlation': rank_corr.group(1).rstrip('%') if rank_corr else '',
        'examples': clean_numeric(examples.group(1)) if examples else '',
        'time_ms': clean_numeric(time_ms.group(1)) if time_ms else '',
        'total_time_s': str(total_time)
    }

def plot_training_metrics(csv_file="training_metrics.csv"):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Convert columns to numeric
    df['total_time_s'] = pd.to_numeric(df['total_time_s'])
    df['avg_loss'] = pd.to_numeric(df['avg_loss'])
    df['rank_correlation'] = pd.to_numeric(df['rank_correlation'])
    df['epoch'] = pd.to_numeric(df['epoch'])
    df['step'] = pd.to_numeric(df['step'])
    df['total_steps'] = pd.to_numeric(df['total_steps'])
    
    # Calculate absolute step count including epoch transitions
    df['absolute_step'] = df['step'] + (df['epoch'] * df['total_steps'])
    
    # Create figure and axis objects with a single subplot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot average loss on primary y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Average Loss', color=color)
    ax1.plot(df['total_time_s'], df['avg_loss'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Create second y-axis for rank correlation
    ax2 = ax1.twinx()
    
    # Plot rank correlation on secondary y-axis
    color = 'tab:orange'
    ax2.set_ylabel('Rank Correlation (%)', color=color)
    ax2.plot(df['total_time_s'], df['rank_correlation'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Create second x-axis for steps
    ax3 = ax1.twiny()
    
    # Set the steps axis limits and labels
    ax3.set_xlim(ax1.get_xlim())
    max_time = df['total_time_s'].max()
    ax3.set_xticks(np.linspace(0, max_time, 5))
    
    # Calculate step labels based on absolute steps
    step_labels = np.linspace(0, df['absolute_step'].max(), 5)
    ax3.set_xticklabels([f'Step {int(s):,}' for s in step_labels])
    ax3.set_xlabel('Training Progress (Steps)')
    
    # Add title and adjust layout
    plt.title('Training Metrics over Time')
    fig.tight_layout()
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, ['Loss', 'Rank Correlation'], loc='upper right')
    
    # Save plot
    plt.savefig('training_metrics.png')
    print("Training metrics plot has been saved as training_metrics.png")

def extract_training_lines(file_path):
    # Pattern to match lines with date/time and training metrics
    # Updated to only match lines that have Step and Batch info
    pattern = r'\[ \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \] Epoch \[\d+\] Step \[.*?\] Batch \[.*?\].*'
    
    with open(file_path, 'r') as f:
        content = f.read()
        matches = re.findall(pattern, content)
        return matches

def find_rank_files(base_dir="rank-files", output_file="training_metrics.csv"):
    # Use glob to recursively find all rank_0.txt files
    pattern = os.path.join(base_dir, "**", "rank_0.txt")
    rank_files = glob.glob(pattern, recursive=True)
    
    # Collect all lines from all files
    all_lines = []
    for file_path in rank_files:
        training_lines = extract_training_lines(file_path)
        all_lines.extend(training_lines)
    
    # Sort lines by timestamp
    sorted_lines = sorted(all_lines, key=parse_timestamp)
    
    # Parse lines and write to CSV
    if sorted_lines:
        # Get field names from first parsed line
        fields = list(parse_training_line(sorted_lines[0]).keys())
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            
            for line in sorted_lines:
                parsed_data = parse_training_line(line)
                writer.writerow(parsed_data)
        
        print(f"Training metrics have been written to {output_file}")
        
        # Generate plot
        plot_training_metrics(output_file)

if __name__ == "__main__":
    find_rank_files() 