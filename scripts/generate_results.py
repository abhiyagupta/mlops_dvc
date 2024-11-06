import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from tabulate import tabulate
from glob import glob
import os

def load_yaml(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)

def find_latest_runs():
    # Look for runs in outputs directory instead of logs
    base_dir = Path("outputs")
    if not base_dir.exists():
        raise FileNotFoundError("outputs directory not found")
    
    # Find all experiment directories
    exp_dirs = list(base_dir.glob("*/*"))
    if not exp_dirs:
        raise FileNotFoundError("No experiment directories found")
    
    return sorted(exp_dirs, key=lambda x: x.stat().st_mtime, reverse=True)

def process_experiment_data(exp_dir):
    metrics_file = list(exp_dir.glob("csv/version_*/metrics.csv"))
    hparams_file = list(exp_dir.glob("csv/version_*/hparams.yaml"))
    
    if not metrics_file or not hparams_file:
        return None
    
    metrics_df = pd.read_csv(metrics_file[0])
    hparams = load_yaml(hparams_file[0])
    
    # Extract key metrics
    final_metrics = metrics_df.iloc[-1]
    metrics_dict = {
        'exp_name': exp_dir.parent.name,
        'run_id': exp_dir.name,
        'final_train_acc': final_metrics.get('train/acc_epoch', None),
        'final_val_acc': final_metrics.get('val/acc_epoch', None),
        'final_test_acc': final_metrics.get('test/acc_epoch', None),
        'final_train_loss': final_metrics.get('train/loss_epoch', None),
        'final_val_loss': final_metrics.get('val/loss_epoch', None),
        'final_test_loss': final_metrics.get('test/loss_epoch', None),
        'best_val_acc': metrics_df['val/acc_epoch'].max() if 'val/acc_epoch' in metrics_df else None,
        'best_val_loss': metrics_df['val/loss_epoch'].min() if 'val/loss_epoch' in metrics_df else None,
    }
    
    # Combine with hyperparameters
    return {**metrics_dict, **hparams}

def generate_results():
    # Find and process all experiments
    exp_dirs = find_latest_runs()
    all_data = []
    
    for exp_dir in exp_dirs:
        exp_data = process_experiment_data(exp_dir)
        if exp_data:
            all_data.append(exp_data)
    
    if not all_data:
        raise ValueError("No valid experiment data found")
    
    # Create DataFrame with all results
    results_df = pd.DataFrame(all_data)
    
    # Generate hyperparameters table
    hparam_cols = [col for col in results_df.columns if col not in 
                  ['exp_name', 'run_id', 'final_train_acc', 'final_val_acc', 'final_test_acc',
                   'final_train_loss', 'final_val_loss', 'final_test_loss', 'best_val_acc', 'best_val_loss']]
    
    hparams_table = tabulate(results_df[['exp_name', 'run_id'] + hparam_cols], 
                           headers='keys', tablefmt='pipe', floatfmt='.4f')
    
    # Generate metrics table
    metrics_table = tabulate(results_df[['exp_name', 'run_id', 'final_train_acc', 'final_val_acc', 
                                       'final_test_acc', 'best_val_acc', 'final_train_loss', 
                                       'final_val_loss', 'final_test_loss', 'best_val_loss']], 
                           headers='keys', tablefmt='pipe', floatfmt='.4f')
    


    # Change the output paths to save in a specific directory
    output_dir = "plots"  # Specify the output directory

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save tables
    #====added line
    hparams_path = os.path.join(output_dir, "hyperparameters_comparison.md")  # Define hparams_path
    with open(os.path.join( output_dir,"hyperparameters_comparison.md"), "w") as f:
        f.write("# Hyperparameters Comparison\n\n")
        f.write(hparams_table)


    metrics_path = os.path.join(output_dir, "metrics_comparison.md")  # Define metrics_path
    with open(os.path.join( output_dir, "metrics_comparison.md"), "w") as f:
        f.write("# Metrics Comparison\n\n")
        f.write(metrics_table)
    
    # Generate plots
    plt.figure(figsize=(15, 10))
    
    # Plot validation accuracy comparison
    plt.subplot(2, 2, 1)
    sns.barplot(data=results_df, x='exp_name', y='final_val_acc')
    plt.title('Validation Accuracy Comparison')
    plt.xticks(rotation=45)
    
    # Plot validation loss comparison
    plt.subplot(2, 2, 2)
    sns.barplot(data=results_df, x='exp_name', y='final_val_loss')
    plt.title('Validation Loss Comparison')
    plt.xticks(rotation=45)
    
    # Plot test accuracy comparison
    plt.subplot(2, 2, 3)
    sns.barplot(data=results_df, x='exp_name', y='final_test_acc')
    plt.title('Test Accuracy Comparison')
    plt.xticks(rotation=45)
    
    # Plot test loss comparison
    plt.subplot(2, 2, 4)
    sns.barplot(data=results_df, x='exp_name', y='final_test_loss')
    plt.title('Test Loss Comparison')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    # Save plot (fixed the savefig syntax)
    plot_path = os.path.join(output_dir, 'metrics_comparison.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved metrics comparison plot to: {plot_path}")
    
    print("\n=== Results Generation Complete ===")
    print(f"Successfully processed {len(results_df)} experiments")
    print("Generated files:")
    print(f"- {os.path.basename(hparams_path)}")
    print(f"- {os.path.basename(metrics_path)}")
    print(f"- {os.path.basename(plot_path)}")

if __name__ == "__main__":
    generate_results()