import numpy as np
import pickle as pkl
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from tslearn.clustering import TimeSeriesKMeans


def save_obj(obj, name, save_to="data"):
    """Aux function that saves an object (preferabyl a dict) as a pickle file."""
    with open(save_to + "/" + name + ".pkl", "wb") as f:
        pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)


def load_obj(name, load_from="data"):  # default load
    """Aux function that loads an object (preferabyl a dict) from a pickle file."""
    with open(load_from + "/" + name + ".pkl", "rb") as f:
        return pkl.load(f)


def get_model(number_of_clusters: int):
    """Function to get the tslearn TimeSeriesKMeans model.
    - Parameter are pre-set for both the automotive and the aursad scenario."""
    return TimeSeriesKMeans(
        # Number of clusters to form
        n_clusters=number_of_clusters,
        # Maximum number of iterations of the k-means algorithm for a single run
        max_iter=20,
        # Inertia variation threshold
        tol=0.0001,
        # Number of time the k-means algorithm will be run with different centroid seeds
        n_init=1,
        # Metric to be used for both cluster assignment and barycenter computation
        metric="dtw",
        # Number of iterations for the barycenter computation process
        max_iter_barycenter=25,
        # Parameter values for the chosen metric
        metric_params=None,
        # The number of jobs to run in parallel for cross-distance matrix computations
        n_jobs=-1,
        # Whether to compute DTW inertia even if DTW is not the chosen metric
        dtw_inertia=False,
        # Print information about the inertia while learning the model
        verbose=False,
        # Generator used to initialize the centers
        random_state=42,
        # Method for initialization
        init="k-means++",
    )


def plot_clustering_results(
    model: TimeSeriesKMeans,
    x_test: np.ndarray,
    y_test: np.array,
    prediction: np.array,
    num_clusters: int,
    num_datasets: int,
    scenario: str,
    filepath: str,
) -> None:
    """Function to plot the results of a single clustering.
    model: TimeSeriesKMeans,
        Fitted model of the cluster analysis.
    - X_test : np.ndarray,
        x values of the test data.
    - y_test: np.array,
        y values of the test data. 
    - prediction: np.array,
        y values of the prediction. 
    - num_clusters: int, 
        Number of clusters of the model. 
    - num_datasets: int, 
        Number of the dataset used for the analysis.
    - scenario: str, 
        Name of the scenario.
    - filepath: str, 
        Path to store the resulting image."""

    # defaults
    mpl.rcParams["figure.dpi"] = 300
    plt.rcParams["font.family"] = "Times New Roman"
    fontsize = 14

    # get list of colors by class (OK:green vs nOK:red)
    colors = np.array(["firebrick" if y == 1 else "forestgreen" for y in y_test])

    # create a new figure
    plt.figure(figsize=[21, 7])  # changes later on

    # set title for figure
    plt.suptitle(
        f"Visualization of the clustering results for the {scenario} data with {num_clusters} clusters and for dataset {num_datasets}",
        fontsize=fontsize + 4,
    )

    # create a subplot for every cluster
    for yi in range(num_clusters):
        # new subplot
        ax = plt.subplot(1, num_clusters, yi + 1)
        for xx, c in zip(x_test[prediction == yi], colors[prediction == yi]):
            plt.plot(xx.ravel(), "k-", alpha=0.2, color=c)
        plt.plot(
            model.cluster_centers_[yi].ravel(),
            color="deepskyblue",
            linestyle=":",
            linewidth=1,
        )

        # get same ax limits for all clusters
        plt.xlim(0, x_test.shape[1])
        plt.ylim(0, 1)

        # set ax labels
        plt.xlabel("Angle [in °]", size=fontsize)
        plt.ylabel("Torque [normalized]", size=fontsize)

        # create a custom legend
        custom_lines = [
            Line2D([0], [0], color="forestgreen", lw=2),
            Line2D([0], [0], color="firebrick", lw=2),
            Line2D([0], [0], color="deepskyblue", lw=2, linestyle=":"),
        ]
        ax.legend(
            custom_lines,
            ["OK observations", "nOK observations", "Cluster center"],
            loc="lower right",
        )

        # show custom grid
        ax.grid(which="major", color="lightgray", linewidth=0.5)
        ax.grid(which="minor", color="lightgray", linewidth=0.25)
        ax.minorticks_on()

        # set subplot title
        plt.title(
            f"c={str(yi+1)} | #OK={str(list(y_test[prediction == yi]).count(0))} | #nOK={str(list(y_test[prediction == yi]).count(1))})",
        )
        plt.tight_layout()

    # store results in format
    for format in ["png"]:  #'svg', 'jpg'
        plt.savefig(
            f"{filepath}/cluster_{num_clusters}_data_{num_datasets}.png",
            dpi="figure",
            format=format,
        )

    plt.show()

