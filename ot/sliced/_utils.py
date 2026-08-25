# -*- coding: utf-8 -*-
"""
Useful functions for solvers for the (balanced) sliced transport problem.
"""

# Author: Adrien Corenflos <adrien.corenflos@aalto.fi>
#         Nicolas Courty   <ncourty@irisa.fr>
#         Rémi Flamary <remi.flamary@polytechnique.edu>
#         Eloi Tanguy <eloi.tanguy@math.cnrs.fr>
#         Laetitia Chapel <laetitia.chapel@irisa.fr>
#         Clément Bonet <clement.bonet.mapp@polytechnique.edu>
#
# License: MIT License

import numpy as np
from ..backend import get_backend, NumpyBackend
from ..utils import get_coordinate_circle


def get_random_projections(d, n_projections, seed=None, backend=None, type_as=None):
    r"""
    Generates n_projections samples from the uniform on the unit sphere of dimension :math:`d-1`: :math:`\mathcal{U}(\mathcal{S}^{d-1})`

    Parameters
    ----------
    d : int
        dimension of the space
    n_projections : int
        number of samples requested
    seed: int or RandomState, optional
        Seed used for numpy random number generator
    backend:
        Backend to use for random generation
    type_as: type, optional
        Type of the returned array

    Returns
    -------
    out: ndarray, shape (d, n_projections)
        The uniform unit vectors on the sphere

    Examples
    --------
    >>> n_projections = 100
    >>> d = 5
    >>> projs = get_random_projections(d, n_projections)
    >>> np.allclose(np.sum(np.square(projs), 0), 1.)  # doctest: +NORMALIZE_WHITESPACE
    True

    """

    if backend is None:
        nx = NumpyBackend()
    else:
        nx = backend

    if isinstance(seed, np.random.RandomState) and str(nx) == "numpy":
        projections = seed.randn(d, n_projections)
    else:
        if seed is not None:
            nx.seed(seed)
        projections = nx.randn(d, n_projections, type_as=type_as)

    projections = projections / nx.sqrt(nx.sum(projections**2, 0, keepdims=True))
    return projections


def get_projections_sphere(d, n_projections, seed=None, backend=None, type_as=None):
    r"""
    Generates n_projections samples from the uniform distribution on the Stiefel manifold of dimension :math:`d\times 2`: :math:`\mathbb{V}_{d,2}=\{X \in \mathbb{R}^{d\times 2}, X^TX=I_2\}`

    Parameters
    ----------
    d : int
        dimension of the space
    n_projections : int
        number of samples requested
    seed: int or RandomState, optional
        Seed used for numpy random number generator
    backend:
        Backend to use for random generation
    type_as: optional
        Type to use for random generation

    Returns
    -------
    out: ndarray, shape (n_projections, d, 2)

    Examples
    --------
    >>> n_projections = 100
    >>> d = 5
    >>> projs = get_projections_sphere(d, n_projections)
    >>> np.allclose(np.einsum("nij, nik -> njk", projs, projs), np.eye(2))  # doctest: +NORMALIZE_WHITESPACE
    True
    """
    if backend is None:
        nx = NumpyBackend()
    else:
        nx = backend

    if isinstance(seed, np.random.RandomState) and str(nx) == "numpy":
        Z = seed.randn(n_projections, d, 2)
    else:
        if seed is not None:
            nx.seed(seed)
        Z = nx.randn(n_projections, d, 2, type_as=type_as)

    projections, _ = nx.qr(Z)
    return projections


def projection_sphere_to_circle(
    x, n_projections=50, projections=None, seed=None, backend=None
):
    r"""
    Projection of :math:`x\in S^{d-1}` on circles using coordinates on [0,1[.

    To get the projection on the circle, we use the following formula:

    .. math::
        P^U(x) = \frac{U^Tx}{\|U^Tx\|_2}

    where :math:`U` is a random matrix sampled from the uniform distribution on the Stiefel manifold of dimension :math:`d\times 2`: :math:`\mathbb{V}_{d,2}=\{X \in \mathbb{R}^{d\times 2}, X^TX=I_2\}`
    and :math:`x` is a point on the sphere. Then, we apply the function get_coordinate_circle to get the coordinates on :math:`[0,1[`.

    Parameters
    ----------
    x : ndarray, shape (n_samples, dim)
        samples on the sphere
    n_projections : int, optional
        Number of projections used for the Monte-Carlo approximation
    projections: shape (n_projections, dim, 2), optional
        Projection matrix (n_projections and seed are not used in this case)
    seed: int or RandomState or None, optional
        Seed used for random number generator
    backend:
        Backend to use for random generation

    Returns
    -------
    Xp_coords: ndarray, shape (n_projections, n_samples)
        Coordinates of the projections on the circle
    """
    if backend is None:
        nx = get_backend(x)
    else:
        nx = backend

    n, d = x.shape

    if projections is None:
        projections = get_projections_sphere(
            d, n_projections, seed=seed, backend=nx, type_as=x
        )

    # Projection on S^1
    # Projection on plane
    Xp = nx.einsum("ikj, lk -> ilj", projections, x)

    # Projection on sphere
    Xp = Xp / nx.sqrt(nx.sum(Xp**2, -1, keepdims=True))

    # Get coordinates on [0,1[
    Xp_coords = nx.reshape(
        get_coordinate_circle(nx.reshape(Xp, (-1, 2))), (n_projections, n)
    )

    return Xp_coords, projections


def get_projections_spiral(
    d, n_projections, randomized=True, seed=None, backend=None, type_as=None
):
    r"""
    Generates n_projections points on the sphere via generalized
    spiral points (Rakhmanov, Saff & Zhou, 1994).

    Only implemented for d=3 (the 2-sphere :math:`S^2`).

    Parameters
    ----------
    d : int
        dimension of the space. Only d=3 is currently supported.
    n_projections : int
        number of samples requested
    randomized : bool, optional
        If True (default), applies a random (d, d) rotation to the
        deterministic spiral point set (RQSW), giving an unbiased estimator
        suitable for stochastic optimization. If False, returns the plain
        deterministic point set (QSW).
    seed: int or RandomState, optional
        Seed used for the random rotation. Ignored if randomized=False.
    backend:
        Backend to use for random generation
    type_as: type, optional
        Type of the returned array

    Returns
    -------
    out: ndarray, shape (d, n_projections)
        The (optionally rotated) spiral points on the sphere

    Examples
    --------
    >>> n_projections = 100
    >>> d = 3
    >>> projs = get_projections_spiral(d, n_projections, randomized=False)
    >>> np.allclose(np.sum(np.square(projs), 0), 1.)  # doctest: +NORMALIZE_WHITESPACE
    True
    >>> rprojs = get_projections_spiral(d, n_projections, randomized=True, seed=0)
    >>> np.allclose(np.sum(np.square(rprojs), 0), 1.)  # doctest: +NORMALIZE_WHITESPACE
    True

    """
    if d != 3:
        raise ValueError(
            f"get_projections_spiral is only implemented for d=3, got d={d}"
        )

    if backend is None:
        nx = NumpyBackend()
    else:
        nx = backend

    # sin/cos are not exposed by the backend abstraction (only arccos, atan2
    # exist), so the deterministic point construction is done in plain NumPy
    # and converted to the target backend at the end.
    i = np.arange(1, n_projections + 1)
    z = 1 - (2 * i - 1) / n_projections
    phi1 = np.arccos(z)
    phi2 = (1.8 * np.sqrt(n_projections) * phi1) % (2 * np.pi)
    theta_np = np.stack(
        [np.sin(phi1) * np.cos(phi2), np.sin(phi1) * np.sin(phi2), z], axis=0
    )  # shape (d, n_projections)
    theta = nx.from_numpy(theta_np, type_as=type_as)

    if not randomized:
        return theta

    if isinstance(seed, np.random.RandomState) and str(nx) == "numpy":
        Z = seed.randn(d, d)
    else:
        if seed is not None:
            nx.seed(seed)
        Z = nx.randn(d, d, type_as=type_as)

    Q, R = nx.qr(Z)
    Q = Q * nx.sign(nx.diag(R))[None, :]
    return nx.matmul(Q, theta)
