import argparse
import numpy as np

from fitting import affine_transform_loss, fit_affine_transform
from utils import Logger, numeric_gradient


parser = argparse.ArgumentParser()
parser.add_argument(
    'action',
    choices=['gradcheck', 'fit'])
parser.add_argument(
    '--learning-rate',
    type=float,
    default=1e-4,
    help='Learning rate to use for gradient descent')
parser.add_argument(
    '--steps',
    type=int,
    default=100,
    help='Numer of iterations to use for gradient descent')
parser.add_argument(
    '--data-file',
    default='points_case_1.npy',
    help='Path to input data file of correspondences')
parser.add_argument(
    '--print-loss-every',
    type=int,
    default=50,
    help='How frequently to print losses during fitting')
parser.add_argument(
    '--loss-plot',
    default=None,
    help='If given, save a plot of the losses to this file.')
parser.add_argument(
    '--animated-gif',
    default=None,
    help='If given, save an animated GIF of the fitting process to this file.')


def main(args):
    if args.action == 'gradcheck':
        gradcheck()
    elif args.action == 'fit':
        fit(args)


def gradcheck(num_trials=100, tolerance=1e-8):
    N = 100
    print('Running numeric gradient checks for affine_transform_loss')
    for _ in range(num_trials):
        X = np.random.randn(N, 2)
        Y = np.random.randn(N, 2)
        S = np.random.randn(2, 2)
        t = np.random.randn(2)
        f_S = lambda _: affine_transform_loss(X, Y, _, t)[0]  # noqa: E731
        f_t = lambda _: affine_transform_loss(X, Y, S, _)[0]  # noqa: E731
        loss, pred, grad_S, grad_t = affine_transform_loss(X, Y, S, t)
        if loss is None:
            print('FAIL: Forward pass not implemented')
            return
        elif grad_S is None or grad_t is None:
            print('FAIL: Backward pass not implemented')
            return
        numeric_grad_S = numeric_gradient(f_S, S)
        numeric_grad_t = numeric_gradient(f_t, t)
        grad_S_max_diff = np.abs(numeric_grad_S - grad_S).max()
        grad_t_max_diff = np.abs(numeric_grad_t - grad_t).max()
        if grad_S_max_diff > tolerance:
            print('FAIL: grad_S not within tolerance')
            print('grad_S:')
            print(grad_S)
            print('numeric_grad_S:')
            print(numeric_grad_S)
            print(f'Max difference: {grad_S_max_diff}')
            return
        if grad_t_max_diff > tolerance:
            print('FAIL: grad_t not within tolerance')
            print(f'grad_t: {grad_t}')
            print(f'grad_t_numeric: {numeric_grad_t}')
            print(f'Max difference: {grad_t_max_diff}')
            return
    print('PASSED')


def fit(args):
    data = np.load(args.data_file)
    X, Y = data[:, :2], data[:, 2:]
    logger = Logger(X, Y, print_every=args.print_loss_every)

    lr = args.learning_rate
    steps = args.steps
    S, t = fit_affine_transform(X, Y, logger, lr, steps)

    print('Final transform:')
    print('S = ')
    print(S)
    print('t = ')
    print(t)

    if args.loss_plot is not None:
        logger.save_loss_plot(args.loss_plot)

    if args.animated_gif is not None:
        logger.save_animated_gif(args.animated_gif)


if __name__ == '__main__':
    main(parser.parse_args())
