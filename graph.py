import numpy as np
import os
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="Run training pipeline.")
parser.add_argument("-p", type=str, help="Path to dataset (.npz)", default=None)
parser.add_argument("-d", type=int, help="")
parser.add_argument("-w", type=int, help="")
parser.add_argument("-e", type=int, help="")

args = parser.parse_args()
folder_path = args.p if args.p else "data/num/"

group_by_depth = args.d
group_by_width = args.w
group_by_epoch = args.e

files = [
    os.path.join(folder_path, f)
    for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f))
]

# --- Setup (outside loop) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Axis formatting (done once)
for ax, ylabel, title in [
    (ax1, "Loss / Similarity", "Training Loss"),
    (ax2, "Similarity", "Cosine Similarity"),
]:
    ax.set(xlabel="Training Progress", ylabel=ylabel, title=title)
    ax.grid(True, axis="y")
    ax.xaxis.set_minor_locator(MultipleLocator(10))  # minor ticks every 10
    # for x in range(0, 500, 20):  # vertical grid lines every 20 steps
    #     ax.axvline(x, color="lightgray", linestyle="--", linewidth=0.5)


for a in files:
    data = np.load(a)
    losses = data["losses"]
    cosin_sims = data["cosin_sims"]
    cycles = data["cycles"]
    epochs = data["epochs"]
    depth = int(a.split("_")[1].replace("d", ""))
    width = int(a.split("_")[2].replace("w", ""))

    if group_by_epoch and epochs != group_by_epoch:
        continue
    if group_by_width and width != group_by_width:
        continue
    if group_by_depth and depth != group_by_depth:
        continue

    n_numbers = 10

    steps = list(range(len(losses)))
    step_interval = len(losses) // len(cosin_sims)
    steps_cosin = [i * step_interval for i in range(len(cosin_sims))]
    number_labels = [(i % n_numbers) + 1 for i in range(len(cosin_sims))]

    # Number of cycles
    total_cycles = len(losses) // n_numbers
    text = (
        (f"Width: {width}" if not group_by_width else "")
        + (f"Depth: {depth}" if not group_by_depth else "")
        + (f"Epoch: {epochs}" if not group_by_epoch else "")
    )
    ax1.plot(steps, losses, label=text)
    ax2.plot(steps_cosin, cosin_sims, label=text)

text = (
    (f"Width: {group_by_width}, " if group_by_width else "")
    + (f"Depth: {group_by_depth}, " if group_by_depth else "")
    + (f"Epoch: {group_by_epoch}" if group_by_epoch else "")
)

fig.suptitle(f"Fixed Info -> {text}", fontsize=14, fontweight="bold")
ax1.legend()
ax2.legend()
plt.tight_layout()
plt.show()
