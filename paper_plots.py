import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from fastdtw import fastdtw
from matplotlib.lines import Line2D
from scipy.spatial.distance import euclidean
from automotive_data_preparation import load_obj

mpl.rcParams["figure.dpi"] = 300
plt.rcParams["font.family"] = "Times New Roman"

savefig_options = dict(format="png", dpi=300, bbox_inches="tight")


# Figure 1.) dtw and ed comparison
# starting point: https://ealizadeh.com/blog/introduction-to-dynamic-time-warping/


def compute_euclidean_distance_matrix(x, y) -> np.array:
    """Calculate distance matrix
    This method calcualtes the pairwise Euclidean distance between two sequences.
    The sequences can have different lengths.
    """
    dist = np.zeros((len(y), len(x)))
    for i in range(len(y)):
        for j in range(len(x)):
            dist[i, j] = (x[j] - y[i]) ** 2
    return dist


def compute_accumulated_cost_matrix(x, y) -> np.array:
    """Compute accumulated cost matrix for warp path using Euclidean distance
    """
    distances = compute_euclidean_distance_matrix(x, y)

    # Initialization
    cost = np.zeros((len(y), len(x)))
    cost[0, 0] = distances[0, 0]

    for i in range(1, len(y)):
        cost[i, 0] = distances[i, 0] + cost[i - 1, 0]

    for j in range(1, len(x)):
        cost[0, j] = distances[0, j] + cost[0, j - 1]

    # Accumulated warp path cost
    for i in range(1, len(y)):
        for j in range(1, len(x)):
            cost[i, j] = (
                min(
                    cost[i - 1, j],  # insertion
                    cost[i, j - 1],  # deletion
                    cost[i - 1, j - 1],  # match
                )
                + distances[i, j]
            )

    return cost


# Create two sequences
x = [
    1.07,
    1.04,
    1.05,
    1.09,
    1.12,
    1.20,
    1.35,
    1.50,
    1.63,
    1.75,
    1.86,
    2.02,
    2.16,
    2.21,
    2.19,
    2.06,
    1.99,
    2.09,
    2.11,
    2.14,
    2.20,
    2.13,
]

y = [
    2.05,
    1.97,
    1.93,
    1.97,
    2.06,
    2.18,
    2.35,
    2.50,
    2.63,
    2.75,
    2.93,
    3.00,
    3.05,
    3.10,
    3.06,
    3.03,
    3.01,
    3.04,
    3.08,
    3.11,
    3.09,
    3.14,
    3.12,
    3.11,
    3.00,
]

# compute dtw distance and warp path
dtw_distance, warp_path = fastdtw(x, y, dist=euclidean)
cost_matrix = compute_accumulated_cost_matrix(x, y)

# get plot defaults
linewidth = 1
fontsize = 12
markersize = 3
color1 = "forestgreen"
color2 = "dodgerblue"
color3 = "darkgray"
markercolor1 = "darkgreen"
markercolor2 = "royalblue"

# make new 1 by 2 fig
fig, (ax1, ax2) = plt.subplots(1, 2)


# ax1
xx = [(i, x[i]) for i in np.arange(1, len(x))]
yy = [(j, y[j]) for j in np.arange(1, len(y))]
for i, j in zip(xx, yy):
    ax1.plot([i[0], j[0]], [i[1], j[1]], "-", color=color3, linewidth=linewidth / 2)
ax1.plot(
    x,
    "-ro",
    color=color1,
    label="x",
    linewidth=linewidth,
    markersize=markersize,
    markerfacecolor=markercolor1,
    markeredgecolor=markercolor1,
)
ax1.plot(
    y,
    "-bo",
    color=color2,
    label="y",
    linewidth=linewidth,
    markersize=markersize,
    markerfacecolor=markercolor2,
    markeredgecolor=markercolor2,
)

# overwrite last three x marker in red
ax1.scatter(len(y) - 1, y[-1], color="firebrick", s=markersize * 5.5)
ax1.scatter(len(y) - 2, y[-2], color="firebrick", s=markersize * 5.5)
ax1.scatter(len(y) - 3, y[-3], color="firebrick", s=markersize * 5.5)


# ax2
for [map_x, map_y] in warp_path:
    ax2.plot(
        [map_x, map_y], [x[map_x], y[map_y]], "-", color=color3, linewidth=linewidth / 2
    )

ax2.plot(
    x,
    "-ro",
    color=color1,
    label="x",
    linewidth=linewidth,
    markersize=markersize,
    markerfacecolor=markercolor1,
    markeredgecolor=markercolor1,
)
ax2.plot(
    y,
    "-bo",
    color=color2,
    label="y",
    linewidth=linewidth,
    markersize=markersize,
    markerfacecolor=markercolor2,
    markeredgecolor=markercolor2,
)

# set ax titles
ax1.set_title("(a) Euclidean distance", fontsize=fontsize)
ax2.set_title("(b) DTW distance", fontsize=fontsize)

# adjust plot ratio
ratio = 0.8
x_left, x_right = ax1.get_xlim()
y_low, y_high = ax1.get_ylim()
ax1.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)
ax2.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

for ax in [ax1, ax2]:
    # remove ax ticks
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])

    # remove spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # set series names x and y
    annotations = ["x", "y"]
    ax.annotate(
        text="y",
        xy=(10, 1.5),
        ha="center",
        va="center",
        multialignment="center",
        fontsize=fontsize,
        color=color1,
    )
    ax.annotate(
        text="x",
        xy=(2, 2.25),
        ha="center",
        va="center",
        multialignment="center",
        fontsize=fontsize,
        color=color2,
    )

# layout and save
fig.tight_layout()
fig.savefig("fig1_ed_vs_dtw_distance.png", **savefig_options)


# Figure 2.) k-means example


def calc_distance(x0: float, y0: float, x1: float, y1: float) -> float:
    """Simple aux function to calculate the distance between to points."""
    return math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)


def list_mean(lst: list) -> float:
    """Simple aux funtion to calculate the average value of a list."""
    return sum(lst) / len(lst)


# get plot defaults
linewidth = 1
fontsize = 12
markersize = 15
markers = ["o", "s", "D"]
colors = ["gray", "darkgreen", "royalblue"]

# get data
x = [
    1.0,
    6.0,
    1.5,
    2.5,
    4.0,
    4.5,
    2.0,
    2.0,
    3.5,
    4.0,
    6.0,
    6.0,
    5.5,
    6.5,
    5.5,
    8.0,
    8.0,
    8.0,
    9.0,
    9.0,
]
y = [
    1.0,
    1.5,
    3.5,
    3.0,
    3.5,
    1.5,
    1.0,
    2.5,
    4.0,
    2.0,
    8.5,
    5.0,
    8.0,
    5.5,
    6.5,
    7.5,
    5.0,
    6.5,
    9.0,
    6.0,
]

# get centers for classes b
center_b1 = (2.5, 7.0)
center_b2 = (7.5, 2.5)

# get centers for classes b
center_c1 = (list_mean(x[:10]), list_mean(y[:10]))
center_c2 = (list_mean(x[10:]), list_mean(y[10:]))

# define classes
classes_a = [0] * 20
classes_b = [
    1
    if (
        calc_distance(_x, _y, center_b1[0], center_b1[1])
        <= calc_distance(_x, _y, center_b2[0], center_b2[1])
    )
    else 2
    for _x, _y in zip(x, y)
]
classes_c = [1] * 10 + [2] * 10


calc_distance(1, 1, center_b1[0], center_b1[1])


# make new 1 by 2 fig
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

for ax, c in zip([ax1, ax2, ax3], [classes_a, classes_b, classes_c]):
    for _x, _y, i in zip(x, y, range(20)):
        ax.scatter(_x, _y, s=markersize, c=colors[c[i]], marker=markers[c[i]])

# get stars for (b)
ax2.scatter(center_b1[0], center_b1[1], s=markersize * 4, c=colors[1], marker="*")
ax2.scatter(center_b2[0], center_b2[1], s=markersize * 4, c=colors[2], marker="*")

# get stars for (c)
ax3.scatter(center_b1[0], center_b1[1], s=markersize * 4, c="lightgray", marker="*")
ax3.scatter(center_b2[0], center_b2[1], s=markersize * 4, c="lightgray", marker="*")
ax3.scatter(center_c1[0], center_c1[1], s=markersize * 4, c=colors[1], marker="*")
ax3.scatter(center_c2[0], center_c2[1], s=markersize * 4, c=colors[2], marker="*")

ax3.arrow(
    center_b1[0],
    center_b1[1],
    (center_c1[0] - center_b1[0]) * 0.9,
    (center_c1[1] - center_b1[1]) * 0.9,
    length_includes_head=True,
    head_width=0.25,
    head_length=0.25,
    linewidth=0.5,
    linestyle=":",
    color="lightgray",
)

ax3.arrow(
    center_b2[0],
    center_b2[1],
    (center_c2[0] - center_b2[0]) * 0.9,
    (center_c2[1] - center_b2[1]) * 0.9,
    length_includes_head=True,
    head_width=0.25,
    head_length=0.25,
    linewidth=0.5,
    linestyle=":",
    color="lightgray",
)

# remove ax ticks
ax1.axes.get_xaxis().set_ticks([])
ax2.axes.get_xaxis().set_ticks([])
ax3.axes.get_xaxis().set_ticks([])
ax1.axes.get_yaxis().set_ticks([])
ax2.axes.get_yaxis().set_ticks([])
ax3.axes.get_yaxis().set_ticks([])

# set ax lims
ax1.set_ylim(0, 10)
ax2.set_ylim(0, 10)
ax3.set_ylim(0, 10)
ax1.set_xlim(0, 10)
ax2.set_xlim(0, 10)
ax3.set_xlim(0, 10)

# set ax titles
ax1.set_title("(a)", fontsize=fontsize)
ax2.set_title("(b)", fontsize=fontsize)
ax3.set_title("(c)", fontsize=fontsize)

# adjust plot ratio
ratio = 1.25
x_left, x_right = ax1.get_xlim()
y_low, y_high = ax1.get_ylim()
ax1.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)
ax2.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)
ax3.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

# remove spines
ax1.spines["top"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax3.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax3.spines["right"].set_visible(False)

fig.tight_layout()

fig.savefig("fig2_kmeans_in_three_steps.png", **savefig_options)


# Figure 3.) confusion matrix

# create new figure
fig = plt.figure()
ax = plt.subplot(111)

# defaults
fontsize = 16

# get simple scatter plot
x_values = [1, 1, 2, 2]
y_values = [2, 1, 2, 1]
plt.scatter(x=x_values, y=y_values, alpha=0.0)

# make matrix
plt.hlines(y=1.5, xmin=0, xmax=2.5, color="black", linewidth=0.85)
plt.vlines(x=1.5, ymin=0, ymax=2.5, color="black", linewidth=0.85)

# put annotations in matrix
annotations = [
    "True positive \n(TP)",
    "False positive \n(FP)",
    "False negative \n(FN)",
    "True negative\n(TN)",
]
for i, label in enumerate(annotations):
    plt.annotate(
        label,
        (x_values[i], y_values[i]),
        ha="center",
        va="center",
        multialignment="center",
        fontsize=fontsize,
    )

# set lims
ax.set_xlim(0.5, 2.5)
ax.set_ylim(0.5, 2.5)

# set labels
ax.set_ylabel("Actual values", fontsize=fontsize + 2, fontweight="bold")
ax.set_xlabel("Predicted values", fontsize=fontsize + 2, fontweight="bold", labelpad=10)

# adjust plot ratio
ratio = 0.5
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)
ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

# get custom x and y ticks
ax.axes.get_xaxis().set_ticks([1, 2])
ax.axes.get_yaxis().set_ticks([1, 2])
ax.set_xticklabels(["Positive class \n(1)", "Negative class \n(0)"], fontsize=fontsize)
ax.set_yticklabels(
    ["Positive class \n(1)", "Negative class \n(0)"],
    fontsize=fontsize,
    multialignment="center",
)

# move x axis
ax.xaxis.tick_top()
ax.xaxis.set_label_position("top")

fig.savefig("fig3_example_of_a_confusion_matrix.png", **savefig_options)


# Figure 4.) OK/nOK example
X_train, X_test, y_train, y_test = load_obj(
    f"train_test_{0}", load_from="data_prepared_automotive"
)

num_to_plot = 25

OK_runs_to_plot = [X_train[i] for i in range(len(X_train)) if y_train[i] == 0][
    :num_to_plot
]
nOK_runs_to_plot = [X_train[i] for i in range(len(X_train)) if y_train[i] == 1][
    :num_to_plot
]

fig = plt.figure()
ax = plt.subplot(111)

for i in range(len(OK_runs_to_plot)):
    plt.plot(nOK_runs_to_plot[i][0], color="firebrick", linewidth=0.75, alpha=0.75)
    plt.plot(OK_runs_to_plot[i][0], color="forestgreen", linewidth=0.75, alpha=0.75)

# set label names
plt.ylabel("Normalized torque [Nm]")
plt.xlabel("Angel of rotation [°]")

# set plot limits
plt.ylim(0, 1)
plt.xlim(0, 325)

# set title
plt.title("Example of OK- and nOK-samples of the screw driving data")

# get background
ax.grid(which="major", color="lightgray", linewidth=0.5)
ax.grid(which="minor", color="lightgray", linewidth=0.25)
ax.minorticks_on()

# make custom legend
custom_lines = [
    Line2D([0], [0], color="forestgreen", lw=2),
    Line2D([0], [0], color="firebrick", lw=2),
]
ax.legend(custom_lines, ["OK runs (25 examples)", "nOK runs (25 examples)"])

fig.tight_layout()

fig.savefig("fig4_example_of_ok&nok_samples.png", **savefig_options)


# Figure 5.) classiciation metrics
# fig 5 is created in automotive_data_evaluation.py


# Figure 6.) clustering example

from auxiliaries import plot_clustering_results
from automotive_data_clustering import prepare_ts_format

# scenario to plot
scenario = "automotive"
num_dataset = 0
num_cluster = 5

# load training and test data and format them for sklearn
x_train, x_test, y_train, y_test = load_obj(
    name=f"train_test_{num_dataset}", load_from=f"data_prepared_{scenario}"
)

x_train = prepare_ts_format(x_train)
x_test = prepare_ts_format(x_test)

# load prediction
y_pred = load_obj(
    name=f"k-means ({scenario})_c{num_cluster}_d{num_dataset}_prediction",
    load_from=f"predictions_{scenario}",
)

# load model
kmeans = load_obj(
    name=f"k-means ({scenario})_c{num_cluster}_d{num_dataset}_model",
    load_from=f"models_{scenario}",
)

# create and store image of the clustering
plot_clustering_results(
    model=kmeans,
    x_test=x_test,
    y_test=y_test,
    prediction=y_pred,
    num_clusters=num_cluster,
    num_datasets=num_dataset,
    scenario=scenario,
    filepath=f"images_{scenario}",
)

