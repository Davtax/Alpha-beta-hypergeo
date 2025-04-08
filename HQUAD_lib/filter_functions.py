import numpy as np
import qutip as qt
from typing import Tuple

PAULI_X = np.array([[0, 1], [1, 0]])
PAULI_Y = np.array([[0, -1j], [1j, 0]])
PAULI_Z = np.array([[1, 0], [0, -1]])
PAULIS = [PAULI_X, PAULI_Y, PAULI_Z]


def hamiltonian(z: float, x: float) -> np.ndarray:
    return np.array([[z, x], [x, -z]])


def _compute_U(x: float, pulse: np.ndarray, tf: float) -> Tuple[np.ndarray, qt.Qobj]:
    time = np.linspace(0, tf, len(pulse))

    H_0 = qt.Qobj(hamiltonian(0, x))
    H_driving = qt.Qobj(hamiltonian(1, 0))
    H_T = [H_0, [H_driving, qt.coefficient(pulse, tlist=time)]]

    return time, qt.propagator(H_T, time)


def compute_FF_LZ(x: float, pulse: np.ndarray, tf: float, omegas: np.ndarray) -> np.ndarray:
    time, U_t = _compute_U(x, pulse, tf)
    U_t = np.array([U_i[:] for U_i in U_t])

    dt = time[1] - time[0]

    R_ij = np.zeros((3, 3, len(time)))
    for i in range(3):
        for j in range(3):
            R_ij[i, j] = np.einsum('tba,bc,tcd,da->t', U_t.conj(), PAULIS[i], U_t, PAULIS[j]).real

    R_ij_omega = np.einsum('abt,to->abo', R_ij, np.exp(1j * omegas[None, :] * time[:, None])) * dt

    FF = np.zeros((3, len(omegas)))
    for alpha in range(3):
        for k in range(3):
            FF[alpha] += np.abs(R_ij_omega[alpha, k]) ** 2

    return FF / 2
