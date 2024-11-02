import os
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import torch 
import hydra
from omegaconf import DictConfig
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

# Find the most recent metrics.csv file
csv_files = glob("outputs/*/*/csv/version_*/metrics.csv")
if not csv_files:
    raise FileNotFoundError("No metrics.csv file found")
latest_csv = max(csv_files, key=os.path.getctime)

# Read the CSV file
df = pd.read_csv(latest_csv)

# Create training loss plot
plt.figure(figsize=(10, 6))
plt.plot(df["step"], df["train/loss_step"], label="Training Loss")
plt.plot(df["step"], df["val/loss_step"], label="Validation Loss")
plt.plot(df["step"], df["test/loss_step"], label="Test Loss", linestyle='--')
plt.xlabel("step")
plt.ylabel("Loss")
plt.title("Loss over step")
plt.legend()
plt.grid(True)
plt.savefig("train_loss_step.png")
plt.close()

# Create training accuracy plot
plt.figure(figsize=(10, 6))
plt.plot(df["step"], df["train/acc_step"], label="Training Accuracy")
plt.plot(df["step"], df["val/acc_step"], label="Validation Accuracy")
plt.plot(df["step"], df["test/acc_step"], label="Test Accuracy", linestyle='--')
plt.xlabel("step")
plt.ylabel("Accuracy")
plt.title("Accuracy over step")
plt.legend()
plt.grid(True)
plt.savefig("train_acc_step.png")
plt.close()

# Generate test metrics table
test_metrics = df.iloc[-1]
test_table = "| Metric | Value |\n|--------|-------|\n"
test_table += f"| Test Accuracy | {test_metrics['test/acc_epoch']:.4f} |\n"
test_table += f"| Test Loss | {test_metrics['test/loss_epoch']:.4f} |\n"

# =========================


# Write the test metrics table to a file
with open("test_metrics.md", "w") as f:
    f.write(test_table)

print("Plots and test metrics table generated successfully.")