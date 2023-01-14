import pandas as pd

from auxiliaries import load_obj, np, plt, mpl
from sklearn.metrics import confusion_matrix, f1_score


def get_binary_clustering(
    y_test: np.ndarray, y_pred: np.ndarray, n_cluster: int
) -> "list[int]":
    """Aux function to convert a prediction with any number of clusters into a binary scenario."""

    # check cluster length
    if n_cluster < 2:
        msg = f"Wrong input for total number of cluster (n_cluster = {n_cluster})"
        raise ValueError(msg)

    # check for same length
    if len(y_test) != len(y_pred):
        msg = f"y values of test and prediction do not have the same length ({len(y_test)}!={len(y_pred)})"
        raise ValueError(msg)

    # conversion for binary evaluation
    if n_cluster != 2:
        # check each cluster
        class_OK, class_nOK = [], []
        for cluster in range(n_cluster):
            cluster_count = np.count_nonzero(y_pred == cluster)
            # cluster_count greater than total_count/n_cluster
            if cluster_count > int(len(y_test) / n_cluster):
                class_OK.append(cluster)
            # cluster_count smaller than total_count/n_cluster
            else:
                class_nOK.append(cluster)
        # make prediction binary according to class-cluster-relations
        y_pred_binary = [0 if y in class_OK else 1 for y in y_pred]
    else:
        y_pred_binary = y_pred
    return y_pred_binary


def get_confusion_matrix(y_true: np.array, y_pred: list, normalize: str) -> np.ndarray:
    """ Simple aux function to get a confusion matrix using sklearn."""
    return confusion_matrix(
        y_true=y_true.tolist(),
        y_pred=y_pred,
        labels=[0, 1],
        sample_weight=None,
        normalize=normalize,
    )


def get_f1_score(y_true: np.array, y_pred: list, how: str) -> float:
    """ Simple aux function to get the F1 score using sklearn."""
    return f1_score(y_true=y_true.tolist(), y_pred=y_pred, average=how)


def plot_result_metrics(df: pd.DataFrame) -> None:
    """Preconfigures plot for classification metrics."""

    fig = plt.figure()
    ax = plt.subplot(111)

    columns = ["TP", "FP", "FN", "TN"]
    labels = ["True positive", "False positive", "False negative", "True negative"]
    colors = ["forestgreen", "firebrick", "firebrick", "forestgreen"]
    styles = ["-", "-", "-.", "-."]

    for i in range(4):
        plt.plot(
            df[columns[i]], label=labels[i], linestyle=styles[i], color=colors[i],
        )

    plt.title("Classification metrics for results of the cluster analysis")
    plt.ylabel("Confusion matrix metrics (averaged & normalized)")
    plt.xlabel("Number of clusters created with k-means")

    plt.ylim(0, 1)
    plt.xlim(2, 10)

    ax.grid(which="major", color="lightgray", linewidth=0.5)
    ax.grid(which="minor", color="lightgray", linewidth=0.25)
    ax.minorticks_on()

    ax.legend(
        loc="center", bbox_to_anchor=(0.5, -0.25), fancybox=True, shadow=True, ncol=2
    )
    plt.savefig("fig5_confusion_matrix_metrics.png", **savefig_options)
    plt.show()


def plot_result_scores(df: pd.DataFrame) -> None:
    """Preconfigures plot for classification scores."""
    fig = plt.figure()
    ax = plt.subplot(111)

    columns = ["F1_micro", "F1_macro", "F1_weighted"]
    labels = [
        "F1 (micro)",
        "F1 (macro)",
        "F1 (weighted)",
    ]
    colors = ["navy", "maroon", "darkgreen"]
    styles = [":", "-", ":"]

    for i in range(3):
        plt.plot(
            df[columns[i]], label=labels[i], linestyle=styles[i], color=colors[i],
        )

    plt.title("Aux plot with more classification metrics")
    plt.ylabel("Average F1-Score value")
    plt.xlabel("Number of Kmeans-Cluster")

    plt.ylim(0, 1)
    plt.xlim(2, 10)

    ax.grid(which="major", color="lightgray", linewidth=0.5)
    ax.grid(which="minor", color="lightgray", linewidth=0.25)
    ax.minorticks_on()

    ax.legend(
        loc="center", bbox_to_anchor=(0.5, -0.21), fancybox=True, shadow=True, ncol=3
    )
    plt.savefig("scores.png", **savefig_options)
    plt.show()


def get_accuracy(TP: int, FP: int, FN: int, TN: int) -> float:
    return (TP + TN) / (TP + TN + FP + FN)


def get_precision(TP, FP, FN, TN):
    return TP / (TP + FP)


def get_recall(TP, FP, FN, TN):
    return TP / (TP + FN)


def get_evaluation(normalize: str) -> pd.DataFrame:
    # create new empty DataFrame for the results of the binary classification
    results = pd.DataFrame(columns=["n_cluster", "data_set", "TP", "FP", "FN", "TN",])

    for n_cluster in range(2, 11):
        for data_set in range(10):
            # Load training and test data
            _, _, _, y_test = load_obj(
                f"train_test_{data_set}", load_from="data_prepared_automotive"
            )

            # load prediction and make it binary
            y_pred = load_obj(
                f"k-means (automotive)_c{n_cluster}_d{data_set}_prediction",
                load_from="predictions_automotive",
            )
            y_pred_binary = get_binary_clustering(y_test, y_pred, n_cluster)

            cm = get_confusion_matrix(
                y_true=y_test, y_pred=y_pred_binary, normalize=normalize
            )
            results = results.append(
                {
                    "n_cluster": n_cluster,
                    "data_set": data_set,
                    "TP": cm[0][0],
                    "FP": cm[0][1],
                    "FN": cm[1][0],
                    "TN": cm[1][1],
                    "accuracy": get_accuracy(cm[0][0], cm[0][1], cm[1][0], cm[1][1]),
                    "precision": get_precision(cm[0][0], cm[0][1], cm[1][0], cm[1][1]),
                    "recall": get_recall(cm[0][0], cm[0][1], cm[1][0], cm[1][1]),
                    "F1_micro": get_f1_score(y_test, y_pred_binary, how="micro"),
                    "F1_macro": get_f1_score(y_test, y_pred_binary, how="macro"),
                    "F1_weighted": get_f1_score(y_test, y_pred_binary, how="weighted"),
                },
                ignore_index=True,
            )
    # group by cluster number and get the avg of all metrics/scores
    return results


# calculate scores and visualize them according to the preconfigurations
if __name__ == "__main__":
    # set some plot defaults
    mpl.rcParams["figure.dpi"] = 300
    plt.rcParams["font.family"] = "Times New Roman"
    savefig_options = dict(format="png", dpi=300, bbox_inches="tight")

    # get normalized results and calculate mean over the ten datasets
    results_for_plot = get_evaluation(normalize="true").groupby("n_cluster").mean()
    # plot fig5
    plot_result_metrics(results_for_plot)
    # plot supporting figure with metrics
    plot_result_scores(results_for_plot)

    # get data for table 1
    results = get_evaluation(normalize=None)
    results_mean = results.groupby("n_cluster").mean().add_suffix("_mean")
    results_var = results.groupby("n_cluster").var().add_suffix("_var")
    results_as_table = results_mean.join(results_var, on="n_cluster")
    results_as_table = results_as_table[
        [
            "accuracy_mean",
            "accuracy_var",
            "precision_mean",
            "precision_var",
            "recall_mean",
            "recall_var",
            "F1_macro_mean",
            "F1_macro_var",
        ]
    ]
    results_as_table = results_as_table.multiply(100).round(decimals=2)
    results_as_table.to_csv("clustering_results_as_table.csv")
