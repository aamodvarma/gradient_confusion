import os
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Run training pipeline.")
parser.add_argument("-p", type=str, help="Path to dataset (.npz)", default=None)
parser.add_argument("-d", type=int, help="")
parser.add_argument("-w", type=int, help="")
parser.add_argument("-e", type=int, help="")
parser.add_argument("-c", type=int, help="")
parser.add_argument("-lr", type=float, help="")
parser.add_argument("-n", nargs="+", type=int)
parser.add_argument("-a", type=bool, default=False)

args = parser.parse_args()
folder_path = args.p if args.p else "data/num/"

group_by_depth = args.d
group_by_width = args.w
group_by_epoch = args.e
group_by_lr = args.lr
group_by_numbers = args.n
group_by_cycles = args.c
print_alpha = args.a

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
    graph_type="cosin_sims",
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
        if graph_type == "cosin_sims":
            cosin_sims = data["cosin_sims"]
        elif graph_type == "avg_errors":
            cosin_sims = data["avg_errors"]

        # window = 10  # adjust window size (higher = smoother)
        # cosin_sims = np.convolve(cosin_sims, np.ones(window) / window, mode="valid")
        epochs = data["epochs"]
        depth = int(a.split("_")[1].replace("d", ""))
        width = int(a.split("_")[2].replace("w", ""))
        lr = float(a.split("_")[4].replace("lr", "").replace(".npz", ""))

        if group_by_epoch and epochs != group_by_epoch:
            continue
        if group_by_width and width != group_by_width:
            continue
        if group_by_width and width != group_by_width:
            continue

        if group_by_lr and lr != group_by_lr:
            continue

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
        print(means)
        stds = [np.std(width_to_min_sims[d]) for d in widths]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))  # Create fig and ax instead of plt.figure()

    ax.errorbar(
        depths if not group_by_depth else widths,
        means,
        yerr=stds,
        fmt="o-",
        capsize=5,
        ecolor="red",
        markerfacecolor="blue",
    )

    comparison_type = "Depth" if not group_by_depth else "Width"
    ax.set_xlabel(comparison_type)

    if graph_type == "cosin_sims":
        ax.set_ylabel("Min Cosine Similarity")
        ax.set_title(f"Min Cosine Similarity vs {comparison_type}")
    elif graph_type == "avg_errors":
        ax.set_ylabel("Avg Error")
        ax.set_title(f"Avg Error vs {comparison_type}")
    ax.grid(True)

    return fig


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

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    # plt.close(fig2)

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
        print(data)
        cycles = data["cycles"]
        epochs = data["epochs"]
        if not default:
            avg_errors = data["avg_errors"]
        depth = int(a.split("_")[1].replace("d", ""))
        width = int(a.split("_")[2].replace("w", ""))
        lr = float(a.split("_")[4].replace("lr", "").replace(".npz", ""))
        try:
            alpha_lrs = (
                float(a.split("_")[5].split("-")[-1].replace(".npz", ""))
                if print_alpha
                else None
            )
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
            # step_interval = len(losses) // len(cosin_sims)
            # steps_cosin = [i * step_interval for i in range(len(cosin_sims))]

            cosin_sims_numbered = data["cosin_sims_numbered"].item()
            step_interval = len(losses) // len(cosin_sims_numbered[0])
            steps_cosin = [
                i * step_interval for i in range(len(cosin_sims_numbered[0]))
            ]
        else:
            cosin_sims_numbered = data["cosin_sims_numbered"].item()
            step_interval = len(losses) // len(cosin_sims_numbered[0])
            steps_cosin = [
                i * step_interval for i in range(len(cosin_sims_numbered[0]))
            ]

        n_numbers = 10

        text = (
            (f"Width: {width} " if not group_by_width else "")
            + (f"Depth: {depth}, " if not group_by_depth else "")
            + (f"Epoch: {epochs} " if not group_by_epoch else "")
            + (f"Alpha: {alpha_lrs} " if alpha_lrs else "")
        )

        if not default:
            avg_errors = data["avg_errors"]
            # avg_errors = np.array([b for a, b in cosin_sims_numbered.items()]).min(
            #     axis=0
            # )
            # print("MIN COS SSIM")

            # print(len(avg_errors))
            # window = 5  # adjust window size (higher = smoother)
            # avg_errors = np.convolve(avg_errors, np.ones(window) / window, mode="valid")

            print(len(avg_errors))
            print(len(steps_cosin))
            (line1,) = ax1.plot(steps_cosin, avg_errors, label=text)
        else:
            # (line1,) = ax1.plot(steps, losses, label=text)
            # (line1,) = ax1.plot(steps_cosin, cosin_sims, label=text)
            window = 10  # adjust window size (higher = smoother)
            cosin_sims_smooth = np.convolve(
                cosin_sims, np.ones(window) / window, mode="valid"
            )
            steps_cosin_smooth = steps_cosin[: len(cosin_sims_smooth)]
            (line1,) = ax1.plot(steps_cosin_smooth, cosin_sims_smooth, label=text)

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
    # ax2.set_xticks(np.arange(60, 500, 100))
    # for x in np.arange(20, 500, 100):
    #     ax2.axvline(x, color="black", linestyle="--", linewidth=1.2, alpha=0.9)

    text = (
        (f"Width: {group_by_width} " if group_by_width else "")
        + (f"Depth: {group_by_depth} " if group_by_depth else "")
        + (f"Epoch: {group_by_epoch} " if group_by_epoch else "")
        + (f"LR: {group_by_lr}" if group_by_lr else "")
    )
    fig1.suptitle(f"Fixed Info - {text}", fontsize=14, fontweight="bold")
    fig2.suptitle(f"Fixed Info - {text}", fontsize=14, fontweight="bold")
    # plt.tight_layout()
    # plt.show()
    return fig1, fig2


# fig1 = plot_min_cosine_by_depth(
#     folder_path,
#     group_by_width,
#     group_by_epoch,
#     group_by_lr,
#     group_by_depth,
#     graph_type="avg_errors",
# )


fig2, fig3 = plot_min_cosine_and_epoch(
    folder_path,
    group_by_width,
    group_by_epoch,
    group_by_lr,
    group_by_depth,
    group_by_numbers,
    group_by_cycles,
    default=False,
)
plt.show()
