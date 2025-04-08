import warnings
from typing import Optional, Tuple

import numpy as np
from scipy.integrate import romb, solve_ivp
from scipy.interpolate import interp1d


def compute_adiabatic_parameter(x_vec: np.ndarray, states: np.ndarray, energies: np.ndarray, initial_state: int,
                                partial_Hamiltonian: np.ndarray, alpha: float, beta: float,
                                limit_sup: Optional[float] = np.inf, limit_inf: Optional[float] = 0) -> Tuple[
    np.ndarray, float]:
    """
    Compute the G_tensor needed for the adiabatic protocol and the value of tilde{c}.

    Parameters
    ----------
    x_vec: np.array
        Vectors with the values of the parameters we are interested to change with a total of N values
    states: np.ndarray
        Matrix with the D instant eigenstates of the system. The dimension is [N x D x D]
    energies: np.ndarray
        Matrix with the D instant eigenenergies of the system. The dimension is [N x D]
    initial_state: int
        Index for the initial state in which we begin the protocol
    partial_Hamiltonian: np.ndarray
        Matrix with the derivative of the Hamiltonian
    alpha: float
        Exponent for the adiabatic parameter
    beta: float
        Exponent for the adiabatic parameter
    limit_sup: float
        Upper limit for the G_tensor
    limit_inf: float
        Lower limit for the G_tensor

    Returns
    -------
    G_tensor: np.ndarray
        Matrix with the G_tensor computed
    """
    n, dim = np.shape(energies)  # Extract the number of steps for the independent variable, and the number of states

    counter = 0  # Temp variable to save the number of G_tensor computed
    G_tensor = np.zeros([n, dim - 1])  # Matrix to save the G_tensor
    for i in range(dim):  # Iterate over all the states
        if i != initial_state:  # If the state is not the initial one
            # Compute the factor, this includes a scalar product
            num = np.abs(
                np.einsum('ia,iab,ib->i', states[..., initial_state].conj(), partial_Hamiltonian, states[..., i],
                          optimize='greedy'))

            den = np.abs(energies[:, i] - energies[:, initial_state])
            G_tensor[:, counter] = num ** beta / (den ** alpha)

            counter += 1

    G_tensor = np.sum(G_tensor, axis=1)

    G_tensor[G_tensor < limit_inf] = limit_inf
    G_tensor[G_tensor > limit_sup] = limit_sup

    # Compute the c_tilda factor, that include a summation over all the states and an integration
    c_tilde = romb(np.sqrt(G_tensor), dx=np.abs(x_vec[1] - x_vec[0]))

    return G_tensor, float(c_tilde)


def compute_parameters_interpolation_ivp(x_vec: np.ndarray, G_tensor: np.ndarray, c_tilde: float, n_s: int) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Function to solve the ODE which gives the result for the parameters in terms of the adimensional variable s=[0,1]
    for the adiabatic protocol

    Parameters
    ----------
    x_vec: np.array
        Vector with the values of the independent variable
    G_tensor: np.ndarray
        Matrix with the factors of the adiabatic protocol
    c_tilde: float
        Value for the rescaled adiabatic parameter
    n_s: int
        Number of points for the interpolation

    Returns
    -------
    s: np.ndarray
        Vector of times
    x_sol: np.ndarray
        Solution for the driving parameter
    """

    sig = np.sign(x_vec[1] - x_vec[0])

    factor_interpolation = interp1d(x_vec, c_tilde / np.sqrt(G_tensor), kind='quadratic', fill_value="extrapolate")

    def model(t, y):  # EDO to be solved
        return sig * factor_interpolation(y)

    s = np.linspace(0, 1, n_s)
    sol = solve_ivp(model, [0, 1], [x_vec[0]], t_eval=s, method='RK45', atol=1e-8, rtol=1e-6, dense_output=True)
    s = sol.t
    x_sol = sol.y[0]

    return s, x_sol


def compute_x_quad(energies: np.ndarray, states: np.ndarray, x_vec: np.ndarray, partial_hamiltonian: np.ndarray,
                   adiabatic_state: int, alpha: float, beta: float, limit_sup: Optional[float] = np.inf,
                   limit_inf: Optional[float] = 0, n_s: Optional[int] = 2 ** 18 + 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the driving parameter for the adiabatic protocol

    Parameters
    ----------
    energies: np.ndarray
        Matrix with the D instant eigenenergies of the system. The dimension is [N x D]
    states: np.ndarray
        Matrix with the D instant eigenstates of the system. The dimension is [N x D x D]
    x_vec: np.ndarray
        Vector with the values of the independent variable
    partial_hamiltonian: np.ndarray
        Matrix with the derivative of the Hamiltonian
    adiabatic_state: int
        Index for the initial state in which we begin the protocol
    alpha: float
        Exponent for the adiabatic parameter
    beta: float
        Exponent for the adiabatic parameter
    limit_sup: float
        Upper limit for the G_tensor
    limit_inf: float
        Lower limit for the G_tensor
    n_s: Optional[int]
        Number of points for the interpolation

    Returns
    -------
    s: np.ndarray
        Vector of times
    x_sol: np.ndarray
        Solution for the driving parameter
    """
    G_tilde, c_tilde = compute_adiabatic_parameter(x_vec, states, energies, adiabatic_state, partial_hamiltonian, alpha,
                                                   beta, limit_sup=limit_sup, limit_inf=limit_inf)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        s, x_sol = compute_parameters_interpolation_ivp(x_vec, G_tilde, c_tilde, n_s)

    return s, x_sol
