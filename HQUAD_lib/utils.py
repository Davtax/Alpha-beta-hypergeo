from __future__ import annotations

import numpy as np
from scipy.integrate import romb
from scipy.linalg import qr, solve

from .QUAD_utils import compute_x_quad


def gram_matrix(functions: np.ndarray | list[np.ndarray], ds: float) -> np.ndarray:
    """
    Compute the Gram matrix G_ij = <f_i|f_j> = int_0^1 dtau f_i(tau) f_j(tau), using Romberg integration.

    Parameters
    ----------
    functions: np.ndarray
        Matrix with the functions in each row, sampled in a grid of 2^k + 1 points. The dimension is [N x n_s]
    ds: float
        Spacing of the grid

    Returns
    -------
    G: np.ndarray
        Gram matrix with dimension [N x N]
    """
    functions = np.asarray(functions)
    N = len(functions)

    G = np.zeros((N, N))
    for i in range(N):
        G[i, :] = romb(functions[i] * functions, dx=ds, axis=-1)

    return G


def independent_basis(
    functions: np.ndarray | list[np.ndarray], ds: float, tolerance: float = 1e-10
) -> tuple[np.ndarray, np.ndarray]:
    """
    Obtain a linearly independent subset of an overcomplete set of functions, using a QR decomposition with pivoting of
    the Gram matrix.

    Parameters
    ----------
    functions: np.ndarray
        Matrix with the functions in each row. The dimension is [N x n_s]
    ds: float
        Spacing of the grid
    tolerance: float (optional, default=1e-10)
        Threshold for the diagonal elements of R used to define the numerical rank

    Returns
    -------
    independent_functions: np.ndarray
        Independent functions, sorted as in the original set. The dimension is [rank x n_s]
    G_independent: np.ndarray
        Gram matrix of the independent functions. The dimension is [rank x rank]
    """
    functions = np.asarray(functions)
    G = gram_matrix(functions, ds)

    _, R, pivot_indices = qr(G, pivoting=True)
    rank = np.sum(np.abs(np.diag(R)) > tolerance)

    independent_indices = sorted(pivot_indices[:rank])

    independent_functions = functions[independent_indices]
    G_independent = G[independent_indices, :][:, independent_indices]

    return independent_functions, G_independent


def split_basis(
    functions: np.ndarray | list[np.ndarray], ds: float, tolerance: float = 1e-10
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split each basis function f_n(tau) into two functions, one restricted to tau < 1/2 and the other to tau >= 1/2, such
    that the approximated pulse reads g(tau) ~ sum_n [c_n^(1) Theta(1/2 - tau) + c_n^(2) Theta(tau - 1/2)] f_n(tau).
    This removes the point symmetry f_n(1 - tau) = -f_n(tau) of the Landau-Zener basis. Finally, a linearly independent
    subset is obtained (see independent_basis).

    Parameters
    ----------
    functions: np.ndarray
        Matrix with the functions in each row. The dimension is [N x n_s]
    ds: float
        Spacing of the grid
    tolerance: float (optional, default=1e-10)
        Threshold for the diagonal elements of R used to define the numerical rank

    Returns
    -------
    independent_functions: np.ndarray
        Independent split functions. The dimension is [rank x n_s]
    G_independent: np.ndarray
        Gram matrix of the independent split functions. The dimension is [rank x rank]
    """
    functions = np.asarray(functions)
    N, n_s = functions.shape

    split_functions = np.vstack([functions] * 2)
    split_functions[:N, n_s // 2 :] = 0
    split_functions[N:, : n_s // 2] = 0

    return independent_basis(split_functions, ds, tolerance=tolerance)


def approximate_fun(
    ds: float,
    independent_functions: np.ndarray | list[np.ndarray],
    G_independent: np.ndarray,
    driving_sol: np.ndarray,
) -> np.ndarray:
    """
    Approximate a function g(tau) as a linear combination of the basis functions, g'(tau) = sum_i c_i f_i(tau). The
    coefficients are obtained by solving the linear system G c = b, with b_i = <g|f_i>.

    Parameters
    ----------
    ds: float
        Spacing of the grid
    independent_functions: np.ndarray
        Basis functions. The dimension is [rank x n_s]
    G_independent: np.ndarray
        Gram matrix of the basis functions. The dimension is [rank x rank]
    driving_sol: np.ndarray
        Function to be approximated, sampled in the same grid as the basis functions

    Returns
    -------
    driving_approx: np.ndarray
        Approximated function
    """
    independent_functions = np.asarray(independent_functions)
    rank = len(independent_functions)
    b = np.zeros(rank)
    for i in range(rank):
        b[i] = romb(driving_sol * independent_functions[i], dx=ds)

    c = solve(G_independent, b)
    driving_approx = np.sum(c[:, None] * independent_functions, axis=0)
    return driving_approx


def compute_pulse_and_approx(
    eigvalues: np.ndarray,
    eigmodes: np.ndarray,
    driving_parameter: np.ndarray,
    partial_hamiltonian: np.ndarray,
    alpha: float,
    beta: float,
    adiabatic_state: int,
    independent_functions: np.ndarray | list[np.ndarray],
    G_independent: np.ndarray,
    limit_sup: float,
    limit_inf: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve the (alpha, beta) hypergeometry for a given Hamiltonian, and approximate the obtained pulse using a basis of
    functions.

    Returns
    -------
    driving_sol: np.ndarray
        Exact pulse, sampled in the same grid as the basis functions
    driving_approx: np.ndarray
        Approximated pulse
    """
    s, driving_sol = compute_x_quad(
        eigvalues,
        eigmodes,
        driving_parameter,
        partial_hamiltonian,
        adiabatic_state,
        alpha,
        beta,
        n_s=2**20 + 1,
        limit_sup=limit_sup,
        limit_inf=limit_inf,
    )

    reduction = int(np.log2((len(s) - 1) / (len(independent_functions[0]) - 1)))
    s = s[:: 2**reduction]
    driving_sol = driving_sol[:: 2**reduction]

    if len(s) != len(independent_functions[0]):
        raise ValueError(
            f"The number of points in the solution does not match the number of points in the basis. {alpha=}, {beta=}, {len(s)=}, {len(independent_functions[0])=}"
        )

    ds = np.abs(s[1] - s[0])

    driving_approx = approximate_fun(
        ds, independent_functions, G_independent, driving_sol
    )

    return driving_sol, driving_approx


def compute_error_basis_approx(
    eigvalues: np.ndarray,
    eigmodes: np.ndarray,
    driving_parameter: np.ndarray,
    partial_hamiltonian: np.ndarray,
    alpha: float,
    beta: float,
    adiabatic_state: int,
    independent_functions: np.ndarray | list[np.ndarray],
    G_independent: np.ndarray,
    limit_sup: float,
    limit_inf: float,
) -> float:
    """
    Compute the reconstruction error E = int_0^1 dtau |g(tau) - g'(tau)| between the exact pulse g(tau) and its
    approximation g'(tau) using a basis of functions.
    """
    driving_sol, driving_approx = compute_pulse_and_approx(
        eigvalues,
        eigmodes,
        driving_parameter,
        partial_hamiltonian,
        alpha,
        beta,
        adiabatic_state,
        independent_functions,
        G_independent,
        limit_sup,
        limit_inf,
    )

    ds = 1 / (len(driving_sol) - 1)
    return float(np.sum(np.abs(driving_sol - driving_approx)) * ds)
