import sys

sys.path += ["layers"]
import numpy as np
from loss_crossentropy import loss_crossentropy
from IPython.display import clear_output
import matplotlib.pyplot as plt

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
            params["save_file"]
            params["plot_on"]
            params["itr_to_plot"]
            params["verbose"]
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
    save_file = params.get("save_file", "model.npz")

    # update_params will be passed to your update_weights function.
    # This allows flexibility in case you want to implement extra features like momentum.
    update_params = {"learning_rate": lr, "weight_decay": wd}

    # monitoring things
    plot = params.get("plot_on", True)
    itr_plt = params.get("itr_to_plot", 1)
    verbose = params.get("verbose", True)

    # unpack validation/test test
    calculate_test = params.get("calculate_test", True)
    itr_to_test = params.get("itr_to_test", 100)
    X_test = params.get("X_test", np.nan)
    y_test = params.get("y_test", np.nan)

    num_inputs = input.shape[-1]
    loss = np.zeros((numIters,))
    accuracy = np.zeros((numIters,))
    test_accuracy = np.full(numIters, np.nan)

    for i in range(numIters):
        # (1) Select a subset of the input to use as a batch
        batch_idx = np.sort(np.random.randint(0, num_inputs, batch_size))
        batch_inputs = input[:, :, :, batch_idx]
        batch_labels = label[batch_idx]

        # (2) Run inference on the batch
        output, activations = inference(model, batch_inputs)

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
            print("running a test!")
            test_output, _ = inference(model, X_test)
            test_y_hat = np.argmax(test_output, axis=0)
            test_accuracy[i] = np.sum((test_y_hat - y_test) == 0) / len(y_test)
        else:
            if i == 0:
                pass
            else:
                test_accuracy[i] = test_accuracy[i - 1]

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
        # TODO- how to get multiple plots to work here?
        # https://stackoverflow.com/questions/70437632/how-can-i-animate-a-matplotlib-plot-from-within-for-loop
        if verbose:
            print(
                f"Fished itr {i} / {numIters}; cost: {np.round(loss[i], 6)}"
                f" train: {accuracy[i]} val: {test_accuracy[i]}, lr: {update_params['learning_rate']}"
            )

        if plot and (i % itr_plt) == 0:
            clear_output(wait=True)
            plt.plot(range(i + 1), loss[: i + 1], color="firebrick")
            plt.plot(range(i + 1), accuracy[: i + 1], color="lightgreen")
            plt.plot(range(i + 1), test_accuracy[: i + 1], color="forestgreen")
            plt.show()

        np.savez(save_file, **model)

    return model, loss, test_accuracy, accuracy
