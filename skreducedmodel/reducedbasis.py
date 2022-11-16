"""Reduced Basis module."""

import logging

from anytree import Node

import numpy as np

from . import integrals

logger = logging.getLogger("arby.basis")


class ReducedBasis:
    """This class contain the methods and function to build
    the reduced basis.

    Parameters
    ----------

    index_seed_global_rb = ...
    lmax = ...
    nmax = ...
    greedy_tol = ...
    normalize =
    integration_rule =

    """

    def __init__(
        self,
        index_seed_global_rb=0,
        lmax=0,
        nmax=np.inf,
        greedy_tol=1e-12,
        normalize=False,
        integration_rule="riemann",
    ) -> None:

        # the default seed is the first of the array "parameters"
        self.index_seed_global_rb = index_seed_global_rb
        self.lmax = lmax
        self.nmax = nmax
        self.greedy_tol = greedy_tol
        self.normalize = normalize
        self.integration_rule = integration_rule
        self.__first_iteration = True

    # comenzamos la implementacion de reduced_basis
    # la idea es acoplar esto al método fit de ReducedModel.
    def fit(
        self,
        training_set,
        parameters,
        physical_points,
        # estas quiero que no se puedan modificar por el usuario.
        # son solo valores para la primera llamada de la
        # funcion fit de la recursion
        parent=None,
        node_idx=0,
        deep=0,
        index_seed=None,  # = self.index_seed_global_rb
    ) -> None:
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

        assert self.nmax > 0 and self.lmax >= 0

        if self.__first_iteration is True:
            index_seed = self.index_seed_global_rb
            assert parent is None and node_idx == 0 and deep == 0
            # index_seed == self.index_seed_global_rb
            self.__first_iteration = False

        # create a node for the tree
        # if the tree does not exists, create it
        if parent is not None:
            node = Node(
                name=parent.name + (node_idx,),
                parent=parent,
                parameters_ts=parameters,
            )
        else:
            self.tree = Node(name=(node_idx,), parameters_ts=parameters)
            node = self.tree

        integration = integrals.Integration(
            physical_points, rule=self.integration_rule
        )

        # useful constants
        ntrain = training_set.shape[0]
        nsamples = training_set.shape[1]
        max_rank = min(ntrain, nsamples)

        # validate inputs
        if nsamples != np.size(integration.weights_):
            raise ValueError(
                "Number of samples is inconsistent with quadrature rule."
            )

        if np.allclose(np.abs(training_set), 0, atol=1e-30):
            raise ValueError("Null training set!")

        # ====== Seed the greedy algorithm and allocate memory ======

        # memory allocation
        greedy_errors = np.empty(max_rank, dtype=np.float64)
        proj_matrix = np.empty((max_rank, ntrain), dtype=training_set.dtype)
        basis_data = np.empty((max_rank, nsamples), dtype=training_set.dtype)

        norms = integration.norm(training_set)

        if self.normalize:  # [fc] ver esta parte. capaz hay que reorganizar.
            # normalize training set
            training_set = np.array(
                [
                    h if np.allclose(h, 0, atol=1e-15) else h / norms[i]
                    for i, h in enumerate(training_set)
                ]
            )

            # choose seed
            next_index = index_seed
            seed = training_set[next_index]

            aux = 0
            # [fc] hacer tests de este loop
            while aux < ntrain - 1:
                if np.allclose(np.abs(seed), 0):
                    if next_index < ntrain - 1:
                        next_index += 1
                    else:
                        next_index = 0

                    seed = training_set[next_index]
                    aux += 1
                else:
                    break

            greedy_indices = [next_index]
            basis_data[0] = training_set[next_index]
            proj_matrix[0] = integration.dot(basis_data[0], training_set)
            sq_errors = _sq_errs_rel
            errs = sq_errors(np.ones(ntrain), proj_matrix[0])

        else:
            # choose seed
            next_index = index_seed  # old version: np.argmax(norms)
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
        print(nn, sigma, next_index)
        while sigma > self.greedy_tol and self.nmax > nn + 1:

            if next_index in greedy_indices:
                break

            nn += 1
            greedy_indices.append(next_index)
            basis_data[nn], _ = _gs_one_element(
                training_set[greedy_indices[nn]],
                basis_data[:nn],
                integration,
            )
            proj_matrix[nn] = integration.dot(basis_data[nn], training_set)
            if self.normalize:
                errs = sq_errors(errs, proj_matrix[nn])
            else:
                errs, diff_training = sq_errors(
                    proj_matrix[nn],
                    basis_data[nn],
                    integration.dot,
                    diff_training,
                )
            next_index = np.argmax(errs)
            greedy_errors[nn] = errs[next_index]

            sigma = errs[next_index]
            print(nn, sigma, next_index)
            logger.debug(nn, "\t", sigma)

        # Prune excess allocated entries
        greedy_errors, proj_matrix = _prune(greedy_errors, proj_matrix, nn + 1)
        if self.normalize:
            # restore proj matrix
            proj_matrix = norms * proj_matrix

        # a esto se lo puede guardar solo cuando el nodo es una hoja del árbol
        node.basis = basis_data[: nn + 1]
        node.indices = greedy_indices
        node.idx_anchor_0 = node.indices[0]
        node.idx_anchor_1 = node.indices[1]
        node.errors = greedy_errors
        node.projection_matrix = proj_matrix.T
        node.integration = integration

        if (
            deep < self.lmax
            and self.greedy_tol < node.errors[-1]
            and len(node.indices) > 1
        ):

            idxs_subspace0, idxs_subspace1 = self.partition(
                parameters, node.idx_anchor_0, node.idx_anchor_1
            )

            self.fit(
                training_set[idxs_subspace0],
                parameters[idxs_subspace0],
                physical_points,
                parent=node,
                node_idx=0,
                deep=deep + 1,
                index_seed=0,
            )

            self.fit(
                training_set[idxs_subspace1],
                parameters[idxs_subspace1],
                physical_points,
                parent=node,
                node_idx=1,
                deep=deep + 1,
                index_seed=0,
            )

    def transform(self, test_set, parameters, s=(None,)):
        # def project(self, h, s=(None,)):
        """Project a function h onto the basis.

        This method represents the action of projecting the function h onto the
        span of the basis.

        Parameters
        ----------
        h : np.ndarray
            Function or set of functions to be projected.
        s : tuple, optional
            Slice the basis. If the slice is not provided, the whole basis is
            considered. Default = (None,)

        Returns
        -------
        projected_function : np.ndarray
            Projection of h onto the basis.
        """
        s = slice(*s)
        projected_function = 0.0
        for e in self.basis[s]:
            projected_function += np.tensordot(
                self.integration.dot(e, test_set), e, axes=0
            )
        return projected_function

    def partition(self, parameters, idx_anchor0, idx_anchor1):
        """
        Parameters
        ----------
        parameters: array of parameters from the domain of the problem
        anchor1: first greedy parameter of the space to divide.
        anchor2: second greedy parameter of the space to divide.

        Returns
        -------
         indices de parametros que corresponden a cada subespacio
        """

        anchor0 = parameters[idx_anchor0]
        anchor1 = parameters[idx_anchor1]

        assert not np.array_equal(anchor0, anchor1)

        seed = 12345
        rng = np.random.default_rng(seed)

        # caso de arby con ts normalizado:
        # la semilla es el primer elemento,
        # por lo tanto, si quiero que los anchors vayan primero:

        # #idxs_subspace0 = []
        # #idxs_subspace1 = []
        idxs_subspace0 = [idx_anchor0]
        idxs_subspace1 = [idx_anchor1]

        # y usar a continuación del for --> if idx != idx_anchor0
        # and idx != idx_anchor1:
        # sirve para el caso de usar normalize = True en
        # reduced_basis()
        # da error en splines porque scipy
        # pide los parametros de forma ordenada

        for idx, parameter in enumerate(parameters):
            if idx != idx_anchor0 and idx != idx_anchor1:
                dist_anchor0 = np.linalg.norm(anchor0 - parameter)  # 2-norm
                dist_anchor1 = np.linalg.norm(anchor1 - parameter)
                if dist_anchor0 < dist_anchor1:
                    idxs_subspace0.append(idx)
                elif dist_anchor0 > dist_anchor1:
                    idxs_subspace1.append(idx)
                else:
                    # para distancias iguales se realiza
                    # una elección aleatoria.
                    # tener en cuenta que se puede agregar el
                    # parametro a ambos subespacios!
                    if rng.integers(2):
                        idxs_subspace0.append(idx)
                    else:
                        idxs_subspace1.append(idx)

        return np.array(idxs_subspace0), np.array(idxs_subspace1)


def _prune(greedy_errors, proj_matrix, num):
    """Prune arrays to have size num."""
    return greedy_errors[:num], proj_matrix[:num]


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
