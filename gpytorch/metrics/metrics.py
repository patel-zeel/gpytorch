from math import pi

import torch

from ..likelihoods import Likelihood
from ..models.gp import GP

pi = torch.tensor(pi)


def negative_log_predictive_density(
    model: GP,
    likelihood: Likelihood,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
) -> float:
    model.eval()
    likelihood.eval()

    with torch.no_grad():
        observed_pred = likelihood(model(test_x))
        return -observed_pred.log_prob(test_y).item()


def mean_standardized_log_loss(
    model: GP,
    likelihood: Likelihood,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
) -> float:
    """
    Reference: GPML book
    """
    model.eval()
    likelihood.eval()

    with torch.no_grad():
        observed_pred = likelihood(model(test_x))
        f_mean = observed_pred.mean
        f_var = observed_pred.variance
        return 0.5 * (torch.log(2 * pi * f_var) + torch.square(test_y - f_mean) / (2 * f_var)).mean().item()


def coverage_error(
    model: GP,
    likelihood: Likelihood,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
) -> float:
    """
    Coverage error for 95% confidence intervals.
    TODO: Find a good reference to improvise this metric.
    """
    model.eval()
    likelihood.eval()

    with torch.no_grad():
        observed_pred = likelihood(model(test_x))
        lower, upper = observed_pred.confidence_region()
        n_samples_within_bounds = ((test_y > lower) * (test_y < upper)).sum()
        fraction = (n_samples_within_bounds / test_y.shape[0]).item()
        return abs(0.9545 - fraction)
