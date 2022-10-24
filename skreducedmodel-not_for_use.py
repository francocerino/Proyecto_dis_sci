import numpy as np
import attr
import integrals
import logging
from anytree import Node, RenderTree

logger = logging.getLogger("arby.basis")


def partition(parameters, idx_anchor0, idx_anchor1):
    # devuelve indices de parametros que corresponden a cada subespacio
    
    # parameters: array of parameters from the domain of the problem
    # anchor1: first greedy parameter of the space to divide.
    # anchor2: second greedy parameter of the space to divide.    
    
    anchor0 = parameters[idx_anchor0]
    anchor1 = parameters[idx_anchor1]
    
    assert not np.array_equal(anchor0, anchor1)
    
    seed = 12345
    rng = np.random.default_rng(seed)
    
    # caso de arby con ts normalizado:
    # la semilla es el primer elemento,
    # por lo tanto, si quiero que los anchors vayan primero:
    
    ##idxs_subspace0 = []
    ##idxs_subspace1 = []
    idxs_subspace0 = [idx_anchor0] 
    idxs_subspace1 = [idx_anchor1]
    
    # y usar a continuación del for --> if idx != idx_anchor0 and idx != idx_anchor1: 
    # sirve para el caso de usar normalize = True en reduced_basis()
    # da error en splines porque scipy pide los parametros de forma ordenada
     
    for idx, parameter in enumerate(parameters):
        if idx != idx_anchor0 and idx != idx_anchor1:
            dist_anchor0 = np.linalg.norm(anchor0-parameter) # 2-norm
            dist_anchor1 = np.linalg.norm(anchor1-parameter)
            if dist_anchor0 < dist_anchor1:
                idxs_subspace0.append(idx)
            elif dist_anchor0 > dist_anchor1:
                idxs_subspace1.append(idx)
            else: # para distancias iguales se realiza una elección aleatoria.
                if rng.integers(2):
                    idxs_subspace0.append(idx)
                else:
                    idxs_subspace1.append(idx)

    return np.array(idxs_subspace0), np.array(idxs_subspace1)

def _prune(greedy_errors, proj_matrix, num):
    """Prune arrays to have size num."""
    return greedy_errors[:num], proj_matrix[:num]

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

class Surrogate:
    # required performance metrics: prediction time and accuracy of the model.
    
    # fit stage is offline.
    def fit(
        parameters, # array N x d. N parameters. Each parameter is d dimensional.  
        training_set, # array N x t. N training functions with values at t physical_points. 
        physical_points, # array 1 x t. physical_points where training_set functions are given.
        # the following parameters can be treated as "hyperparameters" of the problem
        regression_method = None, # the default one can be Least Squares or Splines (like arby)
        seed_rb = 0 , # the default seed is the first of the array "parameters"
        greedy_tol = 1e-12, 
        lmax = 0, 
        nmax = np.inf, 
        normalize = True,
        use_all_training_set_for_regressions = False
        ):
            
        # do parameter compression (find reduced basis RB). can be multiple RB's in case of lmax>0.
        # do time compression, finding empirical times T_i.
        
        # regression stage: each regression has its own hyp-tuning.
        # if real training set:
            # fit one regression for each T_i (map f:R^d->R)
        # elif complex training set:
            # decompose complex data in phase and amplitude
            # fit two regressions for each T_i (maps f:R^d->R)
        pass

    # predict stage is online (must be as fast as possible). 
    def predict(parameters):
        #search model built on subspace corresponding to parameter
        #compute prediction
        #return prediction  
        pass

    def reduced_basis(
        self,
        training_set,
        parameters, #[fc] en el caso lmax=0 no hacen falta para armar la rb
        physical_points,
        seed_rb=0,
        greedy_tol=1e-12,
        lmax=0,
        nmax=np.inf, 
        integration_rule="riemann",
        normalize=False,
        parent = None, 
        node_idx = 0, 
        l = 0,
        ) -> RB:
        """Build a reduced basis from training data.

        This function implements the Reduced Basis (RB) greedy algorithm for
        building an orthonormalized reduced basis out from training data. The basis
        is built for reproducing the training functions within a user specified
        tolerance [TiglioAndVillanueva2021]_ by linear combinations of its
        elements. Tuning the ``greedy_tol`` parameter allows to control the
        representation accuracy of the basis.

        The ``integration_rule`` parameter specifies the rule for defining inner
        products. If the training functions (rows of the ``training_set``) does not
        correspond to continuous data (e.g. time), choose ``"euclidean"``.
        Otherwise choose any of the quadratures defined in the ``arby.Integration``
        class.

        Set the boolean ``normalize`` to True if you want to normalize the training
        set before running the greedy algorithm. This condition not only emphasizes
        on structure over scale but may leads to noticeable speedups for large
        datasets.

        The output is a container which comprises RB data: a ``basis`` object
        storing the reduced basis and handling tools (see ``arby.Basis``); the
        greedy ``errors`` corresponding to the maxima over the ``training set`` of
        the squared projection errors for each greedy swept; the greedy ``indices``
        locating the most relevant training functions used for building the basis;
        and the ``projection_matrix`` storing the projection coefficients generated
        by the greedy algorithm. For example, we can recover the training set (more
        precisely, a compressed version of it) by multiplying the projection matrix
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
            Container for RB data. Contains (``basis``, ``errors``, ``indices``,
            ``projection_matrix``).

        Raises
        ------
        ValueError
            If ``training_set.shape[1]`` doesn't coincide with quadrature rule
            weights.

        Notes
        -----
        If ``normalize`` is True, the projection coefficients are with respect to
        the original basis but the greedy errors are relative to the normalized
        training set.

        References
        ----------
        .. [TiglioAndVillanueva2021] Reduced Order and Surrogate Models for
        Gravitational Waves. Tiglio, M. and Villanueva A. arXiv:2101.11608
        (2021)

        """
        assert nmax > 0 # no tiene sentido nmax == 0.
        
        # create node
        if parent != None:
            node = Node(name = parent.name + (node_idx,),
                parent = parent,
                parameters_ts = parameters)
        else:
            node = Node(name = (node_idx,),
                parameters_ts = parameters)    

        integration = integrals.Integration(physical_points,rule=integration_rule) 

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

        if normalize: # [fc] ver esta parte. capaz hay que reorganizar.
            # normalize training set
            training_set = np.array(
                [
                    h if np.allclose(h, 0, atol=1e-15) else h / norms[i]
                    for i, h in enumerate(training_set)
                ]
            )

            # seed
            next_index = seed_rb
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
                greedy_errors, proj_matrix = _prune(greedy_errors, proj_matrix, nn)
                if normalize:
                    # restore proj matrix
                    proj_matrix = norms * proj_matrix
                
                node.rb = RB(
                    basis=Basis(data=basis_data[:nn], integration=integration),
                    indices=greedy_indices,
                    errors=greedy_errors,
                    projection_matrix=proj_matrix.T,
                    )

                return node

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
                    proj_matrix[nn], basis_data[nn], integration.dot, diff_training
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
        
        setattr(node, "idx_seed", seed_rb)
        
        # hp-greedy recursion
        if l < lmax and greedy_tol < greedy_errors[-1] and nn >= 1:
            #partición de domminio y rb para cada uno de ellos
            #print(node.name," -> l=", l ,"err=",err_Nmax, "shape_rb_with_N_max=",shape_rb_with_N_max)
            idx_anchor0 = greedy_indices[0] # rb.indices -> indices de los elementos del ts que van a la rb
            idx_anchor1 = greedy_indices[1]
            ##idx_anchor0 = model.greedy_indices_[0]
            ##idx_anchor1 = model.greedy_indices_[1]
            
            setattr(node, "idx_anchor0", idx_anchor0)
            setattr(node, "idx_anchor1", idx_anchor1)
            
            print("parameters", parameters)
            print("greedy_indices", greedy_indices)

            idxs_subspace0, idxs_subspace1 = partition(parameters, idx_anchor0, idx_anchor1)

            child0 = self.reduced_basis(
                            self,
                            training_set[idxs_subspace0],
                            parameters[idxs_subspace0],
                            physical_points,
                            seed_rb=0,
                            greedy_tol = greedy_tol,
                            lmax = lmax,
                            nmax = nmax,
                            integration_rule=integration_rule,
                            normalize=normalize, 
                            parent=node, 
                            node_idx=0, 
                            l=l+1,
                     )

            child1 = self.reduced_basis(
                            self,
                            training_set[idxs_subspace1],
                            parameters[idxs_subspace1],
                            physical_points,
                            seed_rb=0,
                            greedy_tol = greedy_tol,
                            lmax = lmax,
                            nmax = nmax,
                            integration_rule=integration_rule,
                            normalize=normalize, 
                            parent=node, 
                            node_idx=0, 
                            l=l+1,
                     )
            
            child0.parent = node
            child1.parent = node
            return node
            
        else:
            # no hay particion de dominio (el nodo es una hoja)
            # se toma como máximo N_max elementos
            # en caso de querer agregar rb en todo nodo, esto va antes del if de hpgreedy.        
            
            basis_domain = basis_data[: nn + 1]
            parameters_domain = greedy_indices
            rb_errors = greedy_errors
            ##basis_domain = model.basis_.data[:N_max]
            ##parameters_domain = model.greedy_indices_[:N_max]
            ##rb_errors = model.greedy_errors_[:N_max]
            #print("leaf:  ",node.name," -> l=", l ,"err=",err_Nmax, "shape_rb_with_N_max=",shape_rb_with_N_max)
            setattr(node, "rb", basis_domain)
            setattr(node, "rb_parameters_idxs", parameters_domain)
            setattr(node, "rb_errors", rb_errors)
            setattr(node, "err_Nmax", greedy_errors[-1])
        #for rb	
            #setattr(node, "rb_data", rb)
            # for rom:
            ##setattr(node,"model",model)


            node.rb = RB(
                        basis=Basis(data=basis_data[: nn + 1], integration=integration),
                        indices=greedy_indices,
                        errors=greedy_errors,
                        projection_matrix=proj_matrix.T,
                      )

            return node


# to see:
# option to compute reduced basis only and project into it.
# 1 only regression for all times, instead one for each empirical time.