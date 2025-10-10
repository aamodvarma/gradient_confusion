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
parser.add_argument("-c", type=int, help="")
parser.add_argument("-lr", type=float, help="")
parser.add_argument("-n", nargs="+", type=int)

args = parser.parse_args()
folder_path = args.p if args.p else "data/num/"

group_by_depth = args.d
group_by_width = args.w
group_by_epoch = args.e
group_by_lr = args.lr
group_by_numbers = args.n
group_by_cycles = args.c

if os.path.isdir(folder_path):
    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]
else:
    files = [folder_path]


def plot_min_cosine_by_depth(
    folder_path="data/num/",
    group_by_width=None,
    group_by_epoch=None,
    group_by_lr=None,
    group_by_depth=None,
):
    """
    Plots the min cosine similarity vs depth with mean and std across files.

    Parameters:
        folder_path (str): Path to the dataset (.npz files)
        group_by_width (int, optional): Filter files by width
        group_by_epoch (int, optional): Filter files by epoch
        group_by_lr (float, optional): Filter files by learning rate
    """
    depth_to_min_sims = {}
    width_to_min_sims = {}

    for a in files:
        data = np.load(a, allow_pickle=True)
        cosin_sims = data["cosin_sims"]
        epochs = data["epochs"]
        depth = int(a.split("_")[1].replace("d", ""))
        width = int(a.split("_")[2].replace("w", ""))
        lr = float(a.split("_")[4].replace("lr", "").replace(".npz", ""))

        # Apply filters
        if group_by_epoch and epochs != group_by_epoch:
            continue
        if group_by_width and width != group_by_width:
            continue
        if group_by_width and width != group_by_width:
            continue

        if group_by_lr and lr != group_by_lr:
            continue

        # min_sim = np.min(cosin_sims)
        if not group_by_depth:
            if depth not in depth_to_min_sims:
                depth_to_min_sims[depth] = []
            depth_to_min_sims[depth].append(cosin_sims)
        if not group_by_width:
            if width not in width_to_min_sims:
                width_to_min_sims[width] = []
            width_to_min_sims[width].append(cosin_sims)

    # Compute mean and std
    if not group_by_depth:
        depths = sorted(depth_to_min_sims.keys())
        means = [np.mean(depth_to_min_sims[d]) for d in depths]
        stds = [np.std(depth_to_min_sims[d]) for d in depths]
    if not group_by_width:
        widths = sorted(width_to_min_sims.keys())
        means = [np.mean(width_to_min_sims[d]) for d in widths]
        stds = [np.std(width_to_min_sims[d]) for d in widths]

    # Plot
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        depths if not group_by_depth else widths,
        means,
        yerr=stds,
        fmt="o-",
        capsize=5,
        ecolor="red",
        markerfacecolor="blue",
    )
    graph_type = "Depth" if not group_by_depth else "Width"
    plt.xlabel(graph_type)
    plt.ylabel("Min Cosine Similarity")
    plt.title(f"Min Cosine Similarity vs {graph_type}")
    plt.grid(True)
    plt.show()


def plot_min_cosine_and_epoch(
    folder_path="data/num/",
    group_by_width=None,
    group_by_epoch=None,
    group_by_lr=None,
    group_by_depth=None,
    group_by_numbers=None,
    group_by_cycles=None,
    default=False,
):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, ylabel, title in [
        (ax1, "Loss / Similarity", "Training Loss"),
        (ax2, "Similarity", "Cosine Similarity"),
    ]:
        ax.set(xlabel="Training Progress", ylabel=ylabel, title=title)
        ax.grid(True, axis="y")
        ax.xaxis.set_minor_locator(MultipleLocator(10))

    handles_ax1, labels_ax1, widths_ax1 = [], [], []
    handles_ax2, labels_ax2, widths_ax2 = [], [], []

    for a in files:
        data = np.load(a, allow_pickle=True)
        cycles = data["cycles"]
        epochs = data["epochs"]
        if not default:
            avg_errors = data["avg_errors"]
        depth = int(a.split("_")[1].replace("d", ""))
        width = int(a.split("_")[2].replace("w", ""))
        lr = float(a.split("_")[4].replace("lr", "").replace(".npz", ""))
        try:
            alpha_lrs = float(a.split("_")[5].split("-")[-1].replace(".npz", ""))
        except:
            alpha_lrs = None

        if group_by_epoch and epochs != group_by_epoch:
            continue
        if group_by_width and width != group_by_width:
            continue
        if group_by_depth and depth != group_by_depth:
            continue

        if group_by_cycles and cycles != group_by_cycles:
            continue
        if group_by_lr and lr != group_by_lr:
            continue

        losses = data["losses"]
        steps = list(range(len(losses)))
        if default:
            cosin_sims = data["cosin_sims"]
            step_interval = len(losses) // len(cosin_sims)
            steps_cosin = [i * step_interval for i in range(len(cosin_sims))]
        else:
            cosin_sims_numbered = data["cosin_sims_numbered"].item()
            step_interval = len(losses) // len(cosin_sims_numbered[0])
            steps_cosin = [
                i * step_interval for i in range(len(cosin_sims_numbered[0]))
            ]

        n_numbers = 10

        text = (
            (f"Width: {width}, " if not group_by_width else "")
            + (f"Depth: {depth}, " if not group_by_depth else "")
            + (f"Epoch: {epochs}, " if not group_by_epoch else "")
            + (f"Alpha: {alpha_lrs}, " if alpha_lrs else "")
        )

        if not default:
            avg_errors = data["avg_errors"]
            (line1,) = ax1.plot(steps_cosin, avg_errors, label=text)
        else:
            (line1,) = ax1.plot(steps, losses, label=text)
        handles_ax1.append(line1)
        labels_ax1.append(text)
        widths_ax1.append(width)

        if default:
            (line2,) = ax2.plot(steps_cosin, cosin_sims, label=text)
            handles_ax2.append(line2)
            labels_ax2.append(text)
            widths_ax2.append(width)
        else:
            lines2 = []
            for i in range(n_numbers):
                if group_by_numbers and i not in group_by_numbers:
                    continue
                (line2,) = ax2.plot(
                    steps_cosin, cosin_sims_numbered[i], label=f"Num: {i}"
                )
                lines2.append(line2)

            for line2 in lines2:
                handles_ax2.append(line2)
                labels_ax2.append(line2.get_label())
                widths_ax2.append(width)

    # Sort by width for the legend
    sorted_items_ax1 = sorted(
        zip(widths_ax1, handles_ax1, labels_ax1), key=lambda x: x[0]
    )
    sorted_handles_ax1 = [item[1] for item in sorted_items_ax1]
    sorted_labels_ax1 = [item[2] for item in sorted_items_ax1]

    sorted_items_ax2 = sorted(
        zip(widths_ax2, handles_ax2, labels_ax2), key=lambda x: x[0]
    )
    sorted_handles_ax2 = [item[1] for item in sorted_items_ax2]
    sorted_labels_ax2 = [item[2] for item in sorted_items_ax2]

    ax1.legend(sorted_handles_ax1, sorted_labels_ax1)
    ax2.legend(sorted_handles_ax2, sorted_labels_ax2)

    text = (
        (f"Width: {group_by_width}, " if group_by_width else "")
        + (f"Depth: {group_by_depth}, " if group_by_depth else "")
        + (f"Epoch: {group_by_epoch}" if group_by_epoch else "")
        + (f"LR: {group_by_lr}" if group_by_lr else "")
    )
    fig.suptitle(f"Fixed Info -> {text}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


# plot_min_cosine_by_depth(
#     folder_path, group_by_width, group_by_epoch, group_by_lr, group_by_depth
# )


plot_min_cosine_and_epoch(
    folder_path,
    group_by_width,
    group_by_epoch,
    group_by_lr,
    group_by_depth,
    group_by_numbers,
    group_by_cycles,
    default=False,
)
