import sys

sys.path += ["layers"]
import numpy as np
from loss_crossentropy import loss_crossentropy
from IPython.display import clear_output
import matplotlib.pyplot as plt
import seaborn as sns
import time
from dropout_inference import dropout_inference

######################################################
# Set use_pcode to True to use the provided pyc code
# for inference, calc_gradient, loss_crossentropy and update_weights
use_pcode = True

# You can modify the imports of this section to indicate
# whether to use the provided pyc or your own code for each of the four functions.
if use_pcode:
    # import the provided pyc implementation
    sys.path += ["pyc_code"]
    from inference_ import inference
    from calc_gradient_ import calc_gradient
    from update_weights_ import update_weights
else:
    # import your own implementation
    from inference import inference
    from calc_gradient import calc_gradient
    from update_weights import update_weights
######################################################


def train_custom(model, input, label, params, numIters):
    """
    This training function is written specifically for classification,
    since it uses crossentropy loss and tests accuracy assuming the final output
    layer is a softmax layer. These can be changed for more general use.
    Args:
        model: Dictionary holding the model
        input: [any dimensions] x [num_inputs]
        label: [num_inputs]
        params: Paramters for configuring training
            params["learning_rate"]
            params["weight_decay"]
            params["batch_size"]
            params["save_path"]
            params["save_file"]
            params["live_plot_on"]
            params["itr_to_plot"]
            params["verbose"]
            params["early_stopping_on"]
            params["calculate_test"]
            params["itr_to_test"]
            params["X_test"]
            params["y_test"]
            Free to add more parameters to this dictionary for your convenience of training.
        numIters: Number of training iterations
    """
    # Initialize training parameters
    # Learning rate
    lr = params.get("learning_rate", 0.01)
    # Weight decay
    wd = params.get("weight_decay", 0.0005)
    # Batch size
    batch_size = params.get("batch_size", 128)
    # There is a good chance you will want to save your network model during/after
    # training. It is up to you where you save and how often you choose to back up
    # your model. By default the code saves the model in 'model.npz'.
    save_file = params.get("save_file", "model")
    save_path = params.get(
        "save_path", "c:\\Users\\JB\\github\\COS429\\Assignment_3\\initial\\results"
    )

    # update_params will be passed to your update_weights function.
    # This allows flexibility in case you want to implement extra features like momentum.
    update_params = {"learning_rate": lr, "weight_decay": wd}

    # monitoring things
    live_plot = params.get("live_plot_on", True)
    itr_plt = params.get("itr_to_plot", 1)
    verbose = params.get("verbose", True)
    early_stopping_on = params.get("early_stopping_on", True)

    # unpack validation/test test
    calculate_test = params.get("calculate_test", True)
    itr_to_test = params.get("itr_to_test", 100)
    X_test = params.get("X_test", np.nan)
    y_test = params.get("y_test", np.nan)

    # dropout
    p_act = params.get("p_act", 1)

    num_inputs = input.shape[-1]
    loss = np.zeros((numIters,))
    accuracy = np.zeros((numIters,))
    test_accuracy = np.full(numIters, np.nan)

    start = time.time()
    for i in range(numIters):
        # (1) Select a subset of the input to use as a batch
        batch_idx = np.sort(np.random.randint(0, num_inputs, batch_size))
        batch_inputs = input[:, :, :, batch_idx]
        batch_labels = label[batch_idx]

        # (2) Run inference on the batch
        output, activations = dropout_inference(model, batch_inputs, p=p_act)

        # (3.1) Calculate Train Batch Loss
        loss[i], dv = loss_crossentropy(
            output, batch_labels, hyper_params=None, backprop=True
        )

        # (3.2) Calculate Train Batch Accuracy
        y_hat = np.argmax(output, axis=0)
        accuracy[i] = np.sum((y_hat - batch_labels) == 0) / len(batch_labels)

        # (3.3) Calculate Validation Set Loss & Accuracy
        if calculate_test and (i % itr_to_test) == 0:

            ## run a forward pass & calculate
            print("running a test!") if i != 0 else None
            test_output, _ = inference(model, X_test)
            test_y_hat = np.argmax(test_output, axis=0)
            test_accuracy[i] = np.sum((test_y_hat - y_test) == 0) / len(y_test)
        else:
            if i == 0:
                pass
            else:
                test_accuracy[i] = test_accuracy[i - 1]

        # (3.4) break if things are going poorly or we're overfitting
        if i > 100 and early_stopping_on:
            if np.mean(accuracy[i - 100 : i]) > 0.95:
                print("early stopping model is over-trained!")
                break
        if np.isnan(loss).any():
            print("early stopping loss is exploding!, turn lr down")
            break

        # (4) Calculate gradients
        grads = calc_gradient(model, batch_inputs, activations, dv)

        # (4.1) Average gradient for momentum
        if i == 0:
            update_params["avg_grad"] = grads
        else:  # calculate average gradient
            for ilayer in range(len(grads)):
                layer_avg_grad = update_params["avg_grad"][ilayer]["W"]
                layer_new_grad = grads[ilayer]["W"]
                if len(layer_avg_grad) != 0:  # skip layers without gradients
                    update_params["avg_grad"][ilayer]["W"] = (
                        layer_new_grad + (i * layer_avg_grad)
                    ) / (i + 1)

        # (5) Update the weights of the model
        model = update_weights(model, grads, hyper_params=update_params)

        # (6) Monitor the progress of training
        if verbose:
            if i == 0:
                print(
                    f"*****Starting training! p: {p_act},bs: {batch_size}, lr: {update_params['learning_rate']}"
                    f" wd: {update_params['weight_decay']}, n_images: {num_inputs}, n_test: {len(y_test)}*****"
                )
            print(
                f"Fished itr {i} / {numIters}; cost: {np.round(loss[i], decimals=5)}"
                f" train: {accuracy[i]} val: {test_accuracy[i]}"
            )

        if live_plot and (i % itr_plt) == 0:
            clear_output(wait=True)
            plt.plot(range(i + 1), loss[: i + 1], color="firebrick")
            plt.plot(range(i + 1), accuracy[: i + 1], color="lightgreen")
            plt.plot(range(i + 1), test_accuracy[: i + 1], color="forestgreen")
            plt.show()

    ## save out model
    np.savez(save_path + save_file + ".npz", **model)

    ## make &  save out plot
    _, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    # plot aesthetics
    plt.tight_layout()
    sns.set_context("talk")
    _ = ax[0].set(title=f"Training Stats for {save_file[1:]}", ylabel="Training Loss")
    _ = ax[1].set(ylabel="Accuracy", xlabel="N Iters")
    # plot
    ax[0].plot(range(numIters), loss, color="firebrick")
    ax[1].plot(range(numIters), accuracy, color="lightgreen", label="train")
    ax[1].plot(range(numIters), test_accuracy, color="forestgreen", label="test")
    plt.legend()
    sns.set_context("notebook")
    # save
    plt.savefig(save_path + save_file, bbox_inches="tight")

    ## keep track of time
    train_time = np.round(time.time() - start, decimals=2)
    print(f"Time to train: {train_time}")

    return model, loss, accuracy, test_accuracy, train_time
