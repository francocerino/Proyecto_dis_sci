import numpy as np
import attr
import integrals
import logging
import validators

logger = logging.getLogger("arby.basis")


class ReducedModel:
    # required performance metrics: prediction time and accuracy of the model.
    def __init__(
            self,
            regression_method=None,
            seed_global_rb=0,
            greedy_tol=1e-12,
            lmax=0,
            nmax=np.inf,
            normalize=False,
            all_training_set_for_reg=False,
            integration_rule="riemann"):

        # the default one can be Least Squares or Splines (like arby)
        self.regression_method = regression_method
        # the default seed is the first of the array "parameters"
        self.seed_global_rb = seed_global_rb
        self.greedy_tol = greedy_tol
        self.lmax = lmax
        self.nmax = nmax
        self.normalize = normalize
        self.all_training_set_for_reg = all_training_set_for_reg
        self.integration_rule = integration_rule

    # fit stage is offline.
    def fit(
            self,
            parameters,  # array N x d. N parameters. Each parameter is d dim.
            # array N x t. N train functions w/ values at t phys_points.
            training_set,
            physical_points,  # array 1 x t. physical_points
            ):
        # do parameter compression (find reduced basis RB).
        # can be multiple RB's in case of lmax>0.
        # do time compression, finding empirical times T_i.
        # regression stage: each regression has its own hyp-tuning.
        # if real training set:
        #     fit one regression for each T_i (map f:R^d->R)
        # elif complex training set:
        #     decompose complex data in phase and amplitude
        #     fit two regressions for each T_i (maps f:R^d->R)

        # Call validators
        validators.validate_parameters(parameters)
        validators.validate_physical_points(physical_points)
        validators.validate_training_set(training_set)

        self.reduced_basis = self.create_reduced_basis(
                #  self,  # [fc] por que funciona comentando el self?
                training_set,
                parameters,
                physical_points,
                greedy_tol=self.greedy_tol,
                lmax=self.lmax,
                nmax=self.nmax,
                seed_global_rb=self.seed_global_rb,
                integration_rule=self.integration_rule,
                normalize=self.normalize
                )

        # return   # [fc] ver la salida que corresponde

    # predict stage is online (must be as fast as possible).
    def predict(
                self,
                parameters,
                ):
        # search model built on subspace corresponding to parameter
        # compute prediction
        # return prediction
        pass


# to see:
# option to compute reduced basis only and project into it.
# 1 only regression for all times, instead one for each empirical time.
