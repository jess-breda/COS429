import numpy as np


def dropout_inference(model, input, p):
    """
    Do forward propagation through the network to get the activation
    at each layer, and the final output
    Args:
        model: Dictionary holding the model
        input: [any dimensions] x [batch_size]
        p = prob that btwn 0 and 1 that a given neuron in a given layer stays active
    Returns:
        output: The final output of the model
        activations: A list of activations for each layer in model["layers"]
    """

    num_layers = len(model["layers"])
    activations = [
        None,
    ] * num_layers

    # TODO: FORWARD PROPAGATION CODE
    for layer in range(num_layers):
        this_layer = model["layers"][layer]
        function = this_layer["fwd_fn"]
        params = this_layer["params"]
        hyper_params = this_layer["hyper_params"]
        output, _, _ = function(input, params, hyper_params, False)

        # initalize random of shape output
        # threshold into mask given p
        # apply and scale
        if layer != (num_layers - 1):
            mask = (np.random.rand(*output.shape) < p) / p
            output *= mask

        activations[layer] = output
        input = output

    output = activations[-1]
    return output, activations
