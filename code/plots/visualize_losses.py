import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--training_losses", "-tl", type=Path, help="Path to the file to visualize", required=True)
parser.add_argument("--run_information", "-ri", type=Path, help="Path to the run information", required=True)
args = parser.parse_args()

file: Path = args.training_losses
run_information: Path = args.run_information

# load trainin_losses.tsv
train_loss = pd.read_csv(file)
run_info = pd.read_csv(run_information)


# visualize as line plot the columns total loss, vae loss, reconstruction loss, cancer accuracy and systems accuracy
sns.set_theme()
plt.figure(figsize=(10, 5), dpi=150)
plt.plot(train_loss["Total Loss"], label="Total Loss")
plt.plot(train_loss["VAE Loss"], label="VAE Loss")
plt.plot(train_loss["Reconstruction Loss"], label="Reconstruction Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
#plt.savefig("losses.png")
plt.show()


# visualize as line plot the columns cancer accuracy and systems accuracy
sns.set_theme()
plt.figure(figsize=(10, 5), dpi=150)
plt.plot(train_loss["Cancer Accuracy"], label="Cancer Accuracy")
plt.plot(train_loss["Systems Accuracy"], label="Systems Accuracy")
# show a vertical line based of the Best Epoch column in the run_info dataset
best_epoch = run_info["Best Epoch"].values[0]
plt.axvline(x=best_epoch, color="r", linestyle="--", label=f"Best Epoch: {best_epoch}")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
#plt.savefig("accuracies.png")
plt.show()
