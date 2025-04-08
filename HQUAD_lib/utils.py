import numpy as np
from scipy.integrate import romb
from scipy.linalg import solve

from .QUAD_utils import compute_x_quad


def compute_error_basis_approx(eigvalues: np.ndarray, eigmodes: np.ndarray, driving_parameter: np.ndarray,
                               partial_hamiltonian, alpha: float, beta: float, adiabatic_state: int,
                               independent_functions: np.ndarray, G_independent: np.ndarray, limit_sup: float,
                               limit_inf: float) -> float:
    s, driving_sol = compute_x_quad(eigvalues, eigmodes, driving_parameter, partial_hamiltonian, adiabatic_state, alpha,
                                    beta, n_s=2 ** 20 + 1, limit_sup=limit_sup, limit_inf=limit_inf)

    reduction = int(np.log2((len(s) - 1) / (len(independent_functions[0]) - 1)))
    s = s[::2 ** reduction]
    driving_sol = driving_sol[::2 ** reduction]

    if len(s) != len(independent_functions[0]):
        raise ValueError(
            f'The number of points in the solution does not match the number of points in the basis. {alpha=}, {beta=}, {len(s)=}, {len(independent_functions[0])=}')

    ds = s[1] - s[0]

    rank = len(independent_functions)
    b = np.zeros(rank)
    for i in range(rank):
        b[i] = romb(driving_sol * independent_functions[i], dx=ds)

    c = solve(G_independent, b)
    driving_approx = np.sum(c[:, None] * independent_functions, axis=0)

    error = np.sum(np.abs(driving_sol - driving_approx)) * ds
    return error


def normalized_time(time_seg: float) -> str:
    """
    Normalize the time to a human-readable format
    """
    if time_seg < 60:
        return f'{time_seg:.2f} s'
    elif time_seg < 3600:
        mins, sec = divmod(time_seg, 60)
        return f'{int(mins)} min {sec:.2f} s'
    else:
        hours, time_seg = divmod(time_seg, 3600)
        mins, sec = divmod(time_seg, 60)
        return f'{int(hours)} h {int(mins)} min {sec:.2f} s'
