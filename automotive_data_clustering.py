import time

from auxiliaries import load_obj, save_obj, get_model, plot_clustering_results, np


def prepare_ts_format(ts: np.ndarray) -> np.array:
    """Small aux function for array formatting."""
    # Convert numpy lists to regular lists
    ts = [t.tolist()[0] for t in ts]
    # Make lists equal in length using None
    max_len = 400  # len(max(ts,key=len))
    ts = [t + [None] * (max_len - (len(t))) for t in ts]
    # Convert to np array
    ts = np.array(ts)
    # Reshape
    ts = ts.reshape(ts.shape + (1,))
    return ts


if __name__ == "__main__":
    # run analysis for the automotive data
    scenario = "k-means (automotive)"
    dataset = "automotive"

    # iter over all numbers of clusters (#c=2 -> ... -> #c=10)
    for num_cluster in range(2, 11):
        print(f"|-|--- Running cluster {num_cluster}.")

        # iter over all ten data sets of train and test samples (90 models in total)
        for num_dataset in range(10):
            print(f"|--- Running data set {num_dataset}.")

            # load training and test data and format them for sklearn
            x_train, x_test, y_train, y_test = load_obj(
                name=f"train_test_{num_dataset}", load_from=f"data_prepared_{dataset}"
            )
            x_train = prepare_ts_format(x_train)
            x_test = prepare_ts_format(x_test)

            # get k-means model for time-series clustering with DTW (from tslearn)
            kmeans = get_model(number_of_clusters=num_cluster)

            # fit the model to the training data
            time_start_fitting = time.time()
            kmeans.fit(x_train)
            print(
                f"Model train complete in: {(time.time()-time_start_fitting)/60} min."
            )

            # predict the y values of the test data
            time_start_predicting = time.time()
            y_pred = kmeans.predict(x_test)
            print(
                f"Model test complete in: {(time.time()-time_start_predicting)/60} min."
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
                filepath=f"images_{dataset}",
            )

            # store model and prediction
            save_obj(
                kmeans,
                name=f"{scenario}_c{num_cluster}_d{num_dataset}_model",
                save_to=f"models_{dataset}",
            )
            save_obj(
                y_pred,
                name=f"{scenario}_c{num_cluster}_d{num_dataset}_prediction",
                save_to=f"predictions_{dataset}",
            )
