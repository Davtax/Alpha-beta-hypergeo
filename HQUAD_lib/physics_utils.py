from typing import List, Optional, Union

import numpy as np
import qutip as qt
import scipy as sp

from .hamiltonians import lz_problem, two_valleys_system


def generalized_Pauli_Matrices(d: int, qutip: Optional[bool] = True) -> List[Union[np.ndarray, qt.Qobj]]:
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
        temp[-1, -1] = (1 - d)
        temp *= np.sqrt(2 / (d * (d - 1)))
        matrices.append(temp)

    if qutip:
        matrices = [qt.Qobj(matrix) for matrix in matrices]

    return matrices


def filter_pulse(pulse: np.ndarray, time: np.ndarray, order: Optional[int] = 3, fc: Optional[float] = 1.) -> np.array:
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
    sos = sp.signal.butter(order, fc, 'low', fs=1 / dt, output='sos')
    filtered_pulse = sp.signal.sosfilt(sos, pulse - np.min(pulse)) + np.min(pulse)

    return filtered_pulse


def fidelity_LZ(z0: float, pulse: np.ndarray, tf: float, x: Optional[float] = 1, gamma: Optional[float] = 0,
                x_bar: Optional[float] = 0, z_bar: Optional[float] = 0) -> float:
    time = np.linspace(0, tf, len(pulse))

    H_0 = qt.Qobj(lz_problem(x + x_bar, 0))
    H_driving = qt.Qobj(lz_problem(0, 1))
    H_T = [H_0, [H_driving, qt.coefficient(pulse + z_bar, tlist=time)]]
    H_T = qt.QobjEvo(H_T)

    psi_0 = qt.Qobj(lz_problem(x, z0, )).groundstate()[1]
    psi_target = qt.Qobj(lz_problem(x, -z0)).groundstate()[1]

    if gamma != 0:
        c_ops = [qt.sigmaz() * np.sqrt(gamma)]
    else:
        c_ops = None

    psi_f = qt.mesolve(H_T, psi_0, time, c_ops=c_ops).states[-1]

    return qt.fidelity(psi_target, psi_f) ** 2


def fidelity_shuttling(pulse: np.ndarray, tf: float, tc: float, phiL: float, phiR: float, DeltaL: float, DeltaR: float,
                       gamma: Optional[float] = 0) -> float:
    time = np.linspace(0, tf, len(pulse))

    H_0 = qt.Qobj(two_valleys_system(0, tc, phiL, phiR, DeltaL, DeltaR))
    H_driving = qt.Qobj(two_valleys_system(1, 0, 0, 0, 0, 0))
    H_T = [H_0, [H_driving, qt.coefficient(pulse, tlist=time)]]
    H_T = qt.QobjEvo(H_T)

    psi_0 = qt.Qobj(two_valleys_system(pulse[0], tc, phiL, phiR, DeltaL, DeltaR)).groundstate()[1]
    psi_target = qt.Qobj(two_valleys_system(pulse[-1], tc, phiL, phiR, DeltaL, DeltaR)).groundstate()[1]

    if gamma != 0:
        c_ops = generalized_Pauli_Matrices(4, qutip=True)
        c_ops = [c_op * np.sqrt(gamma) for c_op in c_ops]
    else:
        c_ops = None

    psi_f = qt.mesolve(H_T, psi_0, time, c_ops=c_ops).states[-1]
    return qt.fidelity(psi_target, psi_f) ** 2
