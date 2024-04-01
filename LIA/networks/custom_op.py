
# import numpy as np
# from typing import Union
import torch

'''
def householder_vectorized(a):
    """Use this version of householder to reproduce the output of np.linalg.qr
    exactly (specifically, to match the sign convention it uses)

    based on https://rosettacode.org/wiki/QR_decomposition#Python
    """
    v = a / (a[0] + np.copysign(np.linalg.norm(a), a[0]))
    v[0] = 1
    tau = 2 / (v.T @ v)

    return v,tau

def qr_decomposition(A: np.ndarray) -> Union[np.ndarray, np.ndarray]:
    m,n = A.shape
    R = A.copy()
    Q = np.identity(m)

    for j in range(n - (m == n)):
        # Apply Householder transformation.
        v, tau = householder_vectorized(R[j:, j, np.newaxis])

        H = np.identity(m)
        H[j:, j:] -= tau * (v @ v.T)
        R = H @ R
        Q = H @ Q

    return Q[:n].T, np.triu(R[:n])
'''


def householder(a):
    a_norm = torch.linalg.norm(a) * (a[0] / torch.abs(a[0]))
    v = a / (a[0] + a_norm)
    v[0] = 1
    tau = 2 / (v.t() @ v)

    return v, tau


def qr_decomposition(A):
    m, n = A.shape
    R = A.clone()
    Q = torch.eye(m)

    for j in range(n):
        a = R[j:, j].unsqueeze(1)
        v, tau = householder(a)

        H = torch.eye(m)
        H[j:, j:] -= tau * (v @ v.t())
        R = H @ R
        Q = H @ Q

    return Q[:n].t()


def diag_embedding(a):
    m, n = a.shape
    b = torch.eye(n)
    stack_list = []
    for i in range(m):
        stack_list.append(a[i]*b)

    return torch.stack(stack_list)


'''
if __name__ == '__main__':
    A = torch.tensor([[12., -51, 4], [6, 167, -68], [-4, 24, -41]])
    Q = qr_decomposition(A)
'''
