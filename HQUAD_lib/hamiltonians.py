import numpy as np


def lz_problem(x: float, z: float) -> np.ndarray:
    return np.array([[z, x], [x, -z]])


def lambda_system(eps12: float, eps23: float, tau1: float, tau2: float) -> np.ndarray:
    return np.array([[eps12, tau1, 0], [tau1, 0, tau2], [0, tau2, eps23]])


def two_valleys_system(eps: float, tc: float, phiL: float, phiR: float, DeltaL: float, DeltaR: float) -> np.ndarray:
    tee = tc / 2 * (1 + np.exp(1j * (phiL - phiR)))
    tgg = tee.conj()

    teg = tc / 2 * (np.exp(1j * phiL) - np.exp(1j * phiR))
    tge = -teg.conj()

    H = np.zeros((4, 4), dtype=complex)
    H[0, 0:] = [eps / 2 + DeltaL, 0, tee, teg]
    H[1, 1:] = [eps / 2 - DeltaL, tge, tgg]
    H[2, 2:] = [-eps / 2 + DeltaR, 0]
    H[3, 3:] = [-eps / 2 - DeltaR]

    H = (H + H.T.conj()) - np.diag(np.diag(H))
    return H
