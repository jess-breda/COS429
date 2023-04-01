import pandas as pd
import numpy as np
import sys

sys.path += ["pyc_code"]
from inference_ import inference

SAVE_PATH = "c:\\Users\\JB\\github\\COS429\\Assignment_3\\initial\\results"


def save_out_model_info(
    params,
    loss,
    train_accuracy,
    val_accuracy,
    time,
    save_and_return=True,
    eval_accuracy=np.nan,
):
    model_fit_df = make_model_fit_df(
        params, loss, train_accuracy, val_accuracy, save_and_return
    )

    model_summaries_df = update_model_summaries_df(
        params, loss, train_accuracy, val_accuracy, time, eval_accuracy, save_and_return
    )

    if save_and_return:
        return model_fit_df, model_summaries_df


def make_model_fit_df(
    params, loss, train_accuracy, val_accuracy, save_and_return=False
):
    save_file = params["save_file"]
    numIters = len(loss)

    model_fit_data = pd.DataFrame(
        {
            "name": [save_file] * numIters,
            "loss": loss,
            "train_acc": train_accuracy,
            "val_acc": val_accuracy,
        }
    )

    model_fit_data.to_csv(SAVE_PATH + save_file + "_fit_data.csv", index=False)
    print(f"{save_file} model fit df saved")
    if save_and_return:
        return model_fit_data


def update_model_summaries_df(
    params,
    loss,
    train_accuracy,
    val_accuracy,
    time,
    eval_accuracy,
    save_and_return=False,
):

    # grab what we want from params dict
    wanted_keys = {"learning_rate", "weight_decay", "batch_size", "save_file"}
    model_fit_info = dict((k, params[k]) for k in wanted_keys if k in params)

    # append / infer time and iters
    numIters = len(loss)
    model_fit_info["numIters"] = [numIters]  #  list to keep pandas happy later
    model_fit_info["time"] = np.round(time / 60, decimals=2)

    # make summary stats (min, mean, max)
    stats_names = ["loss", "train_acc", "val_acc"]
    stats = [loss, train_accuracy, val_accuracy]
    stat_dict = {}
    # set most important param first
    stat_dict["eval_acc"] = eval_accuracy

    for name, stat in zip(stats_names, stats):
        stat_dict[f"{name}_min"] = np.min(stat)
        stat_dict[f"{name}_mean"] = np.mean(stat)
        stat_dict[f"{name}_max"] = np.mean(stat)

    summary_params = {**model_fit_info, **stat_dict}  # merge dicts together
    summary_params_df = pd.DataFrame.from_dict(summary_params)

    # load in previous params
    try:

        prev_params_df = pd.read_csv(SAVE_PATH + "\\base_model_summary.csv")
        prev_params_df = prev_params_df.append(summary_params_df, ignore_index=True)
        prev_params_df.to_csv(SAVE_PATH + "\\base_model_summary.csv", index=False)
        print("\\base_model_summary updated")
        if save_and_return:
            return prev_params_df

    # if this is the first time, make it
    except:
        summary_params_df.to_csv(SAVE_PATH + "\\base_model_summary.csv", index=False)
        print("\\base_model_summary created")
        if save_and_return:
            return summary_params_df


def test(model, X_test, y_test):
    print(f"running test on {len(y_test)} images")
    output, _ = inference(model, X_test)
    y_hat = np.argmax(output, axis=0)
    accuracy = np.sum((y_hat - y_test) == 0) / len(y_test)

    print(f"accuracy is {accuracy}")

    return accuracy
