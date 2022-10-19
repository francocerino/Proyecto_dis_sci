import numpy as np
import attr
import integrals
import logging

logger = logging.getLogger("arby.basis")


def _prune(greedy_errors, proj_matrix, num):
    """Prune arrays to have size num."""
    return greedy_errors[:num], proj_matrix[:num]


def _sq_errs_abs(proj_vector, basis_element, dot_product, diff_training):
    """Square of projection errors from precomputed projection coefficients.

    Since the training set is not a-priori normalized, this function computes
    errors computing the squared norm of the difference between training set
    and the approximation. This method trades accuracy by memory.

    Parameters
    ----------
    proj_vector : numpy.ndarray
        Stores projection coefficients of training functions onto the actual
        basis.
    basis_element : numpy.ndarray
        Actual basis element.
    dot_product : arby.Integration.dot
        Inherited dot product.
    diff_training : numpy.ndarray
        Difference between training set and projected set aiming to be
        actualized.

    Returns
    -------
    proj_errors : numpy.ndarray
        Squared projection errors.
    diff_training : numpy.ndarray
        Actualized difference training set and projected set.
    """
    diff_training = np.subtract(
        diff_training, np.tensordot(proj_vector, basis_element, axes=0)
    )
    return np.real(dot_product(diff_training, diff_training)), diff_training


def _gs_one_element(h, basis, integration, max_iter=3):
    """Orthonormalize a function against an orthonormal basis."""
    norm = integration.norm(h)
    e = h / norm

    for _ in range(max_iter):
        for b in basis:
            e -= b * integration.dot(b, e)
        new_norm = integration.norm(e)
        if new_norm / norm > 0.5:
            break
        norm = new_norm
    else:
        raise StopIteration("Max number of iterations reached ({max_iter}).")

    return e / new_norm, new_norm


def _sq_errs_rel(errs, proj_vector):
    """Square of projection errors from precomputed projection coefficients.

    This function takes advantage of an orthonormalized basis and a normalized
    training set to compute fewer floating-point operations than in the
    non-normalized case.

    Parameters
    ----------
    errs : numpy.array
        Projection errors.
    proj_vector : numpy.ndarray
        Stores the projection coefficients of the training set onto the actual
        basis element.

    Returns
    -------
    proj_errors : numpy.ndarray
        Squared projection errors.
    """
    return np.subtract(errs, np.abs(proj_vector) ** 2)


@attr.s(frozen=True, hash=False, slots=True)
class RB:
    """Container for RB information.

    Parameters
    ----------
    basis: np.ndarray
        Reduced basis object.
    indices: np.ndarray
        Greedy indices.
    errors: np.ndarray
        Greedy errors.
    projection_matrix: np.ndarray
        Projection coefficients.

    """

    basis: np.ndarray = attr.ib()
    indices: np.ndarray = attr.ib()
    errors: np.ndarray = attr.ib()
    projection_matrix: np.ndarray = attr.ib()


@attr.s(frozen=True, hash=False)
class Basis:
    """Basis object and utilities.

    Create a basis object introducing an orthonormalized set of functions
    ``data`` and an ``integration`` class instance to enable integration
    utilities for the basis.

    Parameters
    ----------
    data : numpy.ndarray
        Orthonormalized basis.
    integration : arby.integrals.Integration
        Instance of the ``Integration`` class.

    -->
    Attributes
    ----------
    Nbasis_ : int
        Number of basis elements.
    eim_ : arby.basis.EIM
        Container storing EIM information: ``Interpolant`` matrix and EIM
        ``nodes``.

    Methods
    -------
    interpolate(h)
        Interpolate a function h at EIM nodes.
    project(h)
        Project a function h onto the basis.
    projection_error(h)
        Compute the error from the projection of h onto the basis.
    -->

    """

    data: np.ndarray = attr.ib(converter=np.asarray)
    integration: np.ndarray = attr.ib(
        validator=attr.validators.instance_of(integrals.Integration)
    )


class ReducedModel:
    # required performance metrics: prediction time and accuracy of the model.
    def __init__(self,
                regression_method = None,
                seed_global_rb = 0,
                greedy_tol = 1e-12,
                lmax = 0,
                nmax = np.inf,
                normalize = False,
                all_training_set_for_reg = False,
                integration_rule = "riemann"
                ):

        # the default one can be Least Squares or Splines (like arby)
        self.regression_method = regression_method
        self.seed_global_rb = seed_global_rb  # the default seed is the first of the array "parameters"
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

        rb = self.reduced_basis(
                           #  self,  # [fc] por que funciona comentando el self?
                           training_set,
                           parameters,
                           physical_points,
                           greedy_tol = self.greedy_tol,
                           lmax = self.lmax,
                           nmax = self.nmax,
                           seed_global_rb = self.seed_global_rb,
                           integration_rule = self.integration_rule,
                           normalize = self.normalize
                           )

        return rb  # [fc] ver la salida que corresponde

    # predict stage is online (must be as fast as possible).
    def predict(
                self,
                parameters,
                ):
        # search model built on subspace corresponding to parameter
        # compute prediction
        # return prediction
        pass

    def project(
               self,
               parameters,
               test_set,
               ):

        pass
        
    # comenzamos la implementacion de reduced_basis
    # la idea es acoplar esto al método fit.
    def reduced_basis(
            self,
            training_set,
            parameters,
            physical_points,
            greedy_tol,
            lmax=0,
            nmax=np.inf,
            seed_global_rb=0,
            integration_rule="riemann",
            normalize=False
            ) -> RB:
        """Build a reduced basis from training data.

        This function implements the Reduced Basis (RB) greedy algorithm for
        building an orthonormalized reduced basis out from training data. The
        basis is built for reproducing the training functions within a user
        specified tolerance [TiglioAndVillanueva2021]_ by linear combinations
        of its elements. Tuning the ``greedy_tol`` parameter allows to control
        the representation accuracy of the basis.

        The ``integration_rule`` parameter specifies the rule for defining
        inner products. If the training functions (rows of the
        ``training_set``) does not correspond to continuous data (e.g. time),
        choose ``"euclidean"``. Otherwise choose any of the quadratures
        defined in the ``arby.Integration`` class.

        Set the boolean ``normalize`` to True if you want to normalize the
        training set before running the greedy algorithm. This condition not
        only emphasizes on structure over scale but may leads to noticeable
        speedups for large datasets.

        The output is a container which comprises RB data: a ``basis`` object
        storing the reduced basis and handling tools (see ``arby.Basis``); the
        greedy ``errors`` corresponding to the maxima over the
        ``training set`` of the squared projection errors for each greedy
        swept; the greedy ``indices`` locating the most relevant training
        functions used for building the basis; and the ``projection_matrix``
        storing the projection coefficients generated by the greedy algorithm.
        For example, we can recover the training set (more precisely, a
        compressed version of it) by multiplying the projection matrix
        with the reduced basis.

        Parameters
        ----------
        training_set : numpy.ndarray
            The training set of functions.
        physical_points : numpy.ndarray
            Physical points for quadrature rules.
        integration_rule : str, optional
            The quadrature rule to define an integration scheme.
            Default = "riemann".
        greedy_tol : float, optional
            The greedy tolerance as a stopping condition for the reduced basis
            greedy algorithm. Default = 1e-12.
        normalize : bool, optional
            True if you want to normalize the training set. Default = False.

        Returns
        -------
        arby.basis.RB
            Container for RB data. Contains (``basis``, ``errors``,
            ``indices``, ``projection_matrix``).

        Raises
        ------
        ValueError
            If ``training_set.shape[1]`` doesn't coincide with quadrature rule
            weights.

        Notes
        -----
        If ``normalize`` is True, the projection coefficients are with
        respect to the original basis but the greedy errors are relative
        to the normalized training set.

        References
        ----------
        .. [TiglioAndVillanueva2021] Reduced Order and Surrogate Models for
        Gravitational Waves. Tiglio, M. and Villanueva A. arXiv:2101.11608
        (2021)

        """

        integration = integrals.Integration(
                                            physical_points,
                                            rule=integration_rule
                                            )

        # useful constants
        Ntrain = training_set.shape[0]
        Nsamples = training_set.shape[1]
        max_rank = min(Ntrain, Nsamples)

        # validate inputs
        if Nsamples != np.size(integration.weights_):
            raise ValueError(
                "Number of samples is inconsistent with quadrature rule."
            )

        if np.allclose(np.abs(training_set), 0, atol=1e-30):
            raise ValueError("Null training set!")

        # ====== Seed the greedy algorithm and allocate memory ======

        # memory allocation
        greedy_errors = np.empty(max_rank, dtype=np.float64)
        proj_matrix = np.empty((max_rank, Ntrain), dtype=training_set.dtype)
        basis_data = np.empty((max_rank, Nsamples), dtype=training_set.dtype)

        norms = integration.norm(training_set)

        if normalize:  # [fc] ver esta parte. capaz hay que reorganizar.
            # normalize training set
            training_set = np.array(
                [
                    h if np.allclose(h, 0, atol=1e-15) else h / norms[i]
                    for i, h in enumerate(training_set)
                ]
            )

            # seed
            next_index = seed_global_rb
            seed = training_set[next_index]

            while next_index < Ntrain - 1:
                if np.allclose(np.abs(seed), 0):
                    next_index += 1
                    seed = training_set[next_index]
                else:
                    break

            greedy_indices = [next_index]
            basis_data[0] = training_set[next_index]
            proj_matrix[0] = integration.dot(basis_data[0], training_set)
            sq_errors = _sq_errs_rel
            errs = sq_errors(np.ones(Ntrain), proj_matrix[0])

        else:
            next_index = np.argmax(norms)
            greedy_indices = [next_index]
            basis_data[0] = training_set[next_index] / norms[next_index]
            proj_matrix[0] = integration.dot(basis_data[0], training_set)
            sq_errors = _sq_errs_abs
            errs, diff_training = sq_errors(
                proj_matrix[0], basis_data[0], integration.dot, training_set
            )

        next_index = np.argmax(errs)
        greedy_errors[0] = errs[next_index]
        sigma = greedy_errors[0]

        # ====== Start greedy loop ======
        logger.debug("\n Step", "\t", "Error")
        nn = 0
        while sigma > greedy_tol and nmax > nn + 1:
            nn += 1

            if next_index in greedy_indices:
                # Prune excess allocated entries
                greedy_errors, proj_matrix = _prune(
                                                    greedy_errors,
                                                    proj_matrix,
                                                    nn
                                                    )
                if normalize:
                    # restore proj matrix
                    proj_matrix = norms * proj_matrix
                return RB(
                    basis=Basis(data=basis_data[:nn], integration=integration),
                    indices=greedy_indices,
                    errors=greedy_errors,
                    projection_matrix=proj_matrix.T,
                )

            greedy_indices.append(next_index)
            basis_data[nn], _ = _gs_one_element(
                training_set[greedy_indices[nn]],
                basis_data[:nn],
                integration,
            )
            proj_matrix[nn] = integration.dot(basis_data[nn], training_set)
            if normalize:
                errs = sq_errors(errs, proj_matrix[nn])
            else:
                errs, diff_training = sq_errors(
                    proj_matrix[nn], basis_data[nn],
                    integration.dot, diff_training
                )
            next_index = np.argmax(errs)
            greedy_errors[nn] = errs[next_index]

            sigma = errs[next_index]

            logger.debug(nn, "\t", sigma)

        # Prune excess allocated entries
        greedy_errors, proj_matrix = _prune(greedy_errors, proj_matrix, nn + 1)
        if normalize:
            # restore proj matrix
            proj_matrix = norms * proj_matrix

        return RB(
            basis=Basis(data=basis_data[: nn + 1], integration=integration),
            indices=greedy_indices,
            errors=greedy_errors,
            projection_matrix=proj_matrix.T,
        )


# to see:
# option to compute reduced basis only and project into it.
# 1 only regression for all times, instead one for each empirical time.
