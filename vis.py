import numpy as np
import pandas as pd
import os
import glob
import matplotlib as mlp
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

sns_colors = sns.color_palette("tab10")


import _pickle as cPickle


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", type=str, default="Return", choices=["Return", "Achievement"])

    args = parser.parse_args()
    # fmt: on
    return args


if __name__ == "__main__":
    args = parse_args()
    result_dir = "eval_results"

    xs = [i for i in range(500, 6000 + 1, 500)]
    model_tar_names = [f"checkpoint_{i}" for i in xs]

    # xs = [42000]
    # model_tar_names = [f"model_{i}" for i in xs]

    num_seeds = 2
    # Unpack Data
    path_human = os.path.join(result_dir, "human")
    returns = []
    for file in os.listdir(path_human):
        if file.endswith(".pickle"):
            full_path = os.path.join(path_human, file)

            with open(full_path, "rb") as input_file:
                rets, achievements = cPickle.load(input_file)

            if args.metric == "Return":
                returns.append(rets)
            else:
                returns.append(achievements)

    mean_returns_human = []
    std_returns_human_h = []
    std_returns_human_l = []
    std = 0
    for i in range(len(xs)):
        mean_returns_human.append(np.mean([ret[i] for ret in returns]))
        std = np.std([ret[i] for ret in returns])
        std_returns_human_h.append(mean_returns_human[i] + std)
        std_returns_human_l.append(mean_returns_human[i] - std)

    print(f"Mean {args.metric} human data = {mean_returns_human}, std = {std}")

    path_generator = os.path.join(result_dir, "generator")
    returns = []
    for file in os.listdir(path_generator):
        if file.endswith(".pickle"):
            full_path = os.path.join(path_generator, file)

            with open(full_path, "rb") as input_file:
                rets, achievements = cPickle.load(input_file)
                e = cPickle.load(input_file)

            if args.metric == "Return":
                returns.append(rets)
            else:
                returns.append(achievements)

    mean_returns_generator = []
    std_returns_generator_h = []
    std_returns_generator_l = []
    for i in range(len(xs)):
        mean_returns_generator.append(np.mean([ret[i] for ret in returns]))
        std = np.std([ret[i] for ret in returns])
        std_returns_generator_h.append(mean_returns_generator[i] + std)
        std_returns_generator_l.append(mean_returns_generator[i] - std)

    print(
        f"Mean {args.metric} generator data = {mean_returns_generator}, std = {std}"
    )

    fig, ax = plt.subplots()  # Create a figure and an axes.
    # ax.set_title(
    #     f"{env_name} All", fontdict={"fontsize": 18}
    # )  # Add a title to the axes.

    ax.plot(
        xs,
        mean_returns_human,
        label="Human Data",
        color=sns_colors[0],
        marker="o",
    )
    ax.fill_between(
        xs,
        std_returns_human_l,
        std_returns_human_h,
        color=sns_colors[0],
        alpha=0.3,
    )

    ax.plot(
        xs,
        mean_returns_generator,
        label="Generator Data",
        color=sns_colors[1],
        marker="o",
    )
    ax.fill_between(
        xs,
        std_returns_generator_l,
        std_returns_generator_h,
        color=sns_colors[1],
        alpha=0.3,
    )

    ax.legend()  # Add a legend.
    ax.set_xlabel(
        "# Gradient Updates", fontsize=16
    )  # Add an x-label to the axes.
    ax.set_ylabel(args.metric, fontsize=16)  # Add a y-label to the axes.
    # plt.xlabel('xlabel', f)
    filename = os.path.join(f"eval_{args.metric}.pdf")
    plt.savefig(filename, bbox_inches="tight")
    print(f"Saving round robin plot at {filename}")
