"""
Script to make the basic model of
CONV -> POOL -> Relu -> flatten -> Linear -> Softmax
   with 3 5x5x3 filters for CONV
   with stride 2, size 2 for POOL
"""

import sys

sys.path += ["layers"]
import numpy as np
from init_layers import init_layers
from init_model import init_model
from inference import inference
from loss_euclidean import loss_euclidean

use_pcode = True

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


def init_base_model():
    # TODO could write some functions to make this flexible
    l = [
        init_layers("conv", {"filter_size": 5, "filter_depth": 3, "num_filters": 3}),
        init_layers("pool", {"filter_size": 2, "stride": 2}),
        init_layers("relu", {}),
        init_layers("flatten", {}),
        init_layers("linear", {"num_in": 588, "num_out": 10}),
        init_layers("softmax", {}),
    ]

    model = init_model(layers=l, input_size=[32, 32, 3], output_size=10, display=True)

    return model
