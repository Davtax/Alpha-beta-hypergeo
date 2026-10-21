from __future__ import annotations

import numpy as np
import qutip as qt
import scipy as sp

from .hamiltonians import lz_problem, two_valleys_system

# Only the final state is needed to compute the fidelity. Storing the state at every time step of the pulse slows
# down the dynamics by more than an order of magnitude. Since the whole evolution is a single output interval, the
# maximum number of integration steps is increased.
SOLVER_OPTIONS = {"store_states": False, "store_final_state": True, "nsteps": 10**7}


def generalized_Pauli_Matrices(d: int, qutip: bool = True) -> list:
    """
    Diagonal generalized Gell-Mann matrices of dimension d, normalized as Tr[lambda_i lambda_j] = 2 delta_ij.

    Parameters
    ----------
    d: int
        Dimension of the Hilbert space
    qutip: bool (optional, default=True)
        If True, the matrices are returned as qutip.Qobj, otherwise as np.ndarray

    Returns
    -------
    matrices: list
        The d - 1 diagonal matrices
    """
    if d == 2:
        matrices = [np.array([[1, 0], [0, -1]])]

    else:
        matrices = generalized_Pauli_Matrices(d - 1, qutip=False)

        for i in range(len(matrices)):
            temp = matrices[i]
            temp = np.vstack((temp, np.zeros(d - 1)))
            temp = np.hstack((temp, np.zeros((d, 1))))
            matrices[i] = temp
        temp = np.eye(d)
        temp[-1, -1] = 1 - d
        temp *= np.sqrt(2 / (d * (d - 1)))
        matrices.append(temp)

    if qutip:
        return [qt.Qobj(matrix) for matrix in matrices]

    return matrices


def filter_pulse(
    pulse: np.ndarray, time: np.ndarray, order: int = 3, fc: float = 1.0
) -> np.ndarray:
    """
    Filter a pulse using a low-pass Butterworth filter.

    Parameters
    ----------
    pulse: np.ndarray
        Pulse to be filtered
    time: np.ndarray
        Time array
    order: int
        Order of the Butterworth filter
    fc: float
        Cutoff frequency

    Returns
    -------
    filtered_pulse: np.ndarray
        Filtered pulse
    """
    dt = np.abs(time[1] - time[0])

    # Butterworth
    sos = sp.signal.butter(order, fc, "low", fs=1 / dt, output="sos")
    filtered_pulse = sp.signal.sosfilt(sos, pulse - np.min(pulse)) + np.min(pulse)

    return filtered_pulse


def fidelity_LZ(
    z0: float,
    pulse: np.ndarray,
    tf: float,
    x: float = 1,
    gamma: float = 0,
    x_bar: float = 0,
    z_bar: float = 0,
) -> float:
    """
    Transfer fidelity for the Landau-Zener Hamiltonian H(t) = (z(t) + z_bar) sigma_z + (x + x_bar) sigma_x. The initial
    and target states are the ground states of the noiseless Hamiltonian at z = z0 and z = -z0, respectively.

    Parameters
    ----------
    z0: float
        Initial value of the pulse, used to define the initial and target states
    pulse: np.ndarray
        Pulse z(t), sampled in a uniform time grid in [0, tf]
    tf: float
        Total pulse time
    x: float (optional, default=1)
        Coupling between the diabatic states
    gamma: float (optional, default=0)
        Dephasing rate. If non-zero, the dynamics is computed with a Lindblad master equation with the jump operator
        sqrt(gamma) * sigma_z, i.e., with T_2 = 1 / (2 gamma)
    x_bar, z_bar: float (optional, default=0)
        Quasi-static deviations of x and z(t)

    Returns
    -------
    fidelity: float
        Population of the target state at the end of the protocol
    """
    time = np.linspace(0, tf, len(pulse))

    H_0 = qt.Qobj(lz_problem(x + x_bar, 0))
    H_driving = qt.Qobj(lz_problem(0, 1))
    H_T = [H_0, [H_driving, qt.coefficient(pulse + z_bar, tlist=time)]]

    psi_0 = qt.Qobj(lz_problem(x, z0)).groundstate()[1]
    psi_target = qt.Qobj(lz_problem(x, -z0)).groundstate()[1]

    c_ops = [qt.sigmaz() * np.sqrt(gamma)] if gamma != 0 else None

    psi_f = qt.mesolve(
        H_T, psi_0, [0, tf], c_ops=c_ops, options=SOLVER_OPTIONS
    ).final_state

    return qt.fidelity(psi_target, psi_f) ** 2


def fidelity_shuttling(
    pulse: np.ndarray,
    tf: float,
    tc: float,
    phiL: float,
    phiR: float,
    DeltaL: float,
    DeltaR: float,
    gamma: float = 0,
    adiabatic_state: int = 1,
) -> float:
    """
    Transfer fidelity for the shuttling model (two dots with two valleys each). The initial and target states are the
    instantaneous eigenstates with index adiabatic_state (sorted by increasing energy) of the Hamiltonian at the
    initial and final values of the pulse, respectively. By default, the first excited state is transferred, i.e.,
    |L, v_L^+> -> |R, v_R^+>.

    Parameters
    ----------
    pulse: np.ndarray
        Detuning pulse epsilon(t), sampled in a uniform time grid in [0, tf]
    tf: float
        Total pulse time
    tc, phiL, phiR, DeltaL, DeltaR: float
        Parameters of the shuttling Hamiltonian (see hamiltonians.two_valleys_system)
    gamma: float (optional, default=0)
        Dephasing rate. If non-zero, the dynamics is computed with a Lindblad master equation with jump operators
        sqrt(gamma) * lambda_i, with lambda_i the diagonal generalized Pauli matrices.
    adiabatic_state: int (optional, default=1)
        Index of the instantaneous eigenstate to be transferred

    Returns
    -------
    fidelity: float
        Population of the target state at the end of the protocol
    """
    time = np.linspace(0, tf, len(pulse))

    H_0 = qt.Qobj(two_valleys_system(0, tc, phiL, phiR, DeltaL, DeltaR))
    H_driving = qt.Qobj(two_valleys_system(1, 0, 0, 0, 0, 0))
    H_T = [H_0, [H_driving, qt.coefficient(pulse, tlist=time)]]

    psi_0 = qt.Qobj(
        two_valleys_system(pulse[0], tc, phiL, phiR, DeltaL, DeltaR)
    ).eigenstates()[1][adiabatic_state]
    psi_target = qt.Qobj(
        two_valleys_system(pulse[-1], tc, phiL, phiR, DeltaL, DeltaR)
    ).eigenstates()[1][adiabatic_state]

    if gamma != 0:
        c_ops = [
            c_op * np.sqrt(gamma) for c_op in generalized_Pauli_Matrices(4, qutip=True)
        ]
    else:
        c_ops = None

    psi_f = qt.mesolve(
        H_T, psi_0, [0, tf], c_ops=c_ops, options=SOLVER_OPTIONS
    ).final_state
    return qt.fidelity(psi_target, psi_f) ** 2
