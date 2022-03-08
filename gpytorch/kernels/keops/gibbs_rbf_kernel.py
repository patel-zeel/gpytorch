#!/usr/bin/env python3
import torch

from ... import settings
from ...constraints import Positive
from ...distributions import MultivariateNormal
from ...lazy import KeOpsLazyTensor
from ...likelihoods import GaussianLikelihood
from ...means import ConstantMean
from ...mlls import LLSGPLossTerm
from ...models import ExactGP
from ..rbf_kernel import RBFKernel
from ..scale_kernel import ScaleKernel
from .keops_kernel import KeOpsKernel

try:
    # from pykeops.torch import LazyTensor as KEOLazyTensor

    class ExactGPModel(ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = ConstantMean()
            self.covar_module = ScaleKernel(RBFKernel())

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return MultivariateNormal(mean_x, covar_x)

    class GibbsRBFKernel(KeOpsKernel):
        """
        1D Gibbs kernel for 1D input data with keops.
        """

        has_lengthscale = False

        def __init__(self, inducing_points, active_dims, constraint=None, **kwargs):
            super(GibbsRBFKernel, self).__init__(active_dims=active_dims, **kwargs)

            if inducing_points.ndimension() == 1:
                inducing_points = inducing_points.unsqueeze(-1)
            self.inducing_points = inducing_points

            if constraint is None:
                constraint = Positive()
            self.constraint = constraint

            self.raw_lengthscale_inducing = torch.nn.Parameter(
                torch.zeros(
                    inducing_points.shape[0],
                    dtype=inducing_points.dtype,
                    device=inducing_points.device,
                )
            )

            self.likelihood = GaussianLikelihood()

            self.latent_model = ExactGPModel(
                self.inducing_points,
                # None,
                self.raw_lengthscale_inducing,
                # None,
                self.likelihood,
            )
            self.register_added_loss_term("lls_gp_loss_term")

        def _nonkeops_covar_func(self, x1, x2, diag=False):
            dist = self.covar_dist(
                x1,
                x2,
                square_dist=True,
                diag=diag,
            )
            return dist.exp()

        def covar_func(self, x1, x2, diag=False):
            with torch.autograd.enable_grad():
                if diag:
                    return self._nonkeops_covar_func(x1, x2, diag=True)
                self.latent_model.eval()
                with settings.detach_test_caches(False):
                    l1_dist = self.latent_model.likelihood(self.latent_model(x1))
                    l2_dist = self.latent_model.likelihood(self.latent_model(x2))
                    l1_mean = l1_dist.mean
                    l2_mean = l2_dist.mean
                    added_term = l1_dist.log_prob(l1_mean)
                    if self.training:
                        self.update_added_loss_term("lls_gp_loss_term", LLSGPLossTerm(added_term))

                    # l1_ = KEOLazyTensor(
                    #     self.constraint.transform(l1_mean).view(-1, 1)[..., :, None, :]
                    # )
                    # l2_ = KEOLazyTensor(
                    #     self.constraint.transform(l2_mean).view(-1, 1)[..., None, :, :]
                    # )
                    l1_ = l1_mean.view(-1, 1)[..., :, None, :]
                    l2_ = l2_mean.view(-1, 1)[..., None, :, :]

                    # x1_ = KEOLazyTensor(x1[..., :, None, :])
                    # x2_ = KEOLazyTensor(x1[..., None, :, :])
                    x1_ = x1[..., :, None, :]
                    x2_ = x1[..., None, :, :]

                    l1l2_sqr = (l1_ ** 2) + (l2_ ** 2)
                    x1x2_sqr = (x1_ - x2_) ** 2
                    l1l2_mul_sqrt = (l1_ * l2_).sqrt()

                    exp_term = (-x1x2_sqr / l1l2_sqr).exp()
                    prefix_term = l1l2_mul_sqrt / (l1l2_sqr / 2).sqrt()

                    return (prefix_term * exp_term).sum(-1)

                # if (
                #     diag
                #     or x1.size(-2) < settings.max_cholesky_size.value()
                #     or x2.size(-2) < settings.max_cholesky_size.value()
                # ):
                #     return self._nonkeops_covar_func(x1, x2, diag=diag)

        def forward(self, x1, x2, diag=False, **params):
            covar_func = lambda x1, x2, diag=diag: self.covar_func(x1, x2, diag)

            if diag:
                return covar_func(x1, x2, diag=True)

            return KeOpsLazyTensor(x1, x2, covar_func)


except ImportError:

    class GibbsRBFKernel(KeOpsKernel):
        def __init__(self, *args, **kwargs):
            super().__init__()
