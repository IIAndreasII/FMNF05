import numpy as np
from scipy.linalg import eig
import cv2

A = np.matrix([[0, 2], [2, 3]])


def qr_eig(A, k):
    m = A.shape[0]
    Q = np.eye(m)
    R = A.copy()

    for i in range(1, k):
        Q, R = np.linalg.qr(A @ Q)
        # Q = Q @ Q_k
        # R = R_k @ Q_k

    return Q.T, np.diag(Q.T @ A @ Q)


e_vec, e_val = qr_eig(A, 100)
"""
ev = -1, -2
e_vec = [[1, 0], [-10, 1]]
"""

# print(e_vec)
# print(e_val)


def unshifted_qr(A, k):
    m = A.shape[0]
    Q = np.eye(m)
    Qbar = Q.copy()
    R = A.copy()
    for i in range(1, k):
        Q, R = np.linalg.qr(R @ Q)
        Qbar = Qbar @ Q
    lam = np.diag(R @ Q)
    return lam, Qbar


e_val, e_vec = unshifted_qr(A, 10000)
# print(e_val)
# print(e_vec)


LENA = "lena_gray.bmp"


def gray_scale_img_svd(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Compute the SVD of the image
    U, S, Vt = np.linalg.svd(img)

    # Print the dimensions of the U, S, and Vt matrices
    print("Dimensions of U:", U.shape)
    print("Dimensions of S:", S.shape)
    print("Dimensions of Vt:", Vt.shape)

    return U, S, Vt


def reconstruct(U, S, Vt, k=256):
    S_k = np.diag(S[:k])
    U_k = U[:, :k]
    VT_k = Vt[:k, :]
    A = U_k @ S_k @ VT_k
    return A


# U, S, Vt = gray_scale_img_svd(LENA)
# A = reconstruct(U, S, Vt)

# cv2.imshow("Reconstructed", A.astype(np.uint8))
# cv2.waitKey(0)

A = np.array([[1, 2], [2, 1]], dtype=np.float64)

x0 = np.array([1, 1]).T

# To fix the error, we need to make sure that the shape of x0 matches the number of columns in A.
# We can do this by transposing x0 using the T attribute, like this:


import numpy as np


def power_method(A, max_iter=1000, tol=1e-6):
    n = A.shape[0]
    x = np.random.rand(n)
    x = x / np.linalg.norm(x)

    for _ in range(max_iter):
        x_next = np.dot(A, x)
        x_next_norm = np.linalg.norm(x_next)
        x_next = x_next / x_next_norm

        if np.linalg.norm(x_next - x) < tol:
            break

        x = x_next

    eigenvalue = np.dot(x, np.dot(A, x))
    eigenvector = x

    return eigenvalue, eigenvector


def inverse_power_method(A, max_iter=1000, tol=1e-6):
    n = A.shape[0]
    x = np.random.rand(n)
    x = x / np.linalg.norm(x)
    A_inv = np.linalg.inv(A)

    for _ in range(max_iter):
        x_next = np.dot(A_inv, x)
        x_next_norm = np.linalg.norm(x_next)
        x_next = x_next / x_next_norm

        if np.linalg.norm(x_next - x) < tol:
            break

        x = x_next

    eigenvalue = 1 / np.dot(x, np.dot(A_inv, x))
    eigenvector = x

    return eigenvalue, eigenvector


def deflate(A, eigenvalue, eigenvector):
    n = A.shape[0]
    outer_product = np.outer(eigenvector, eigenvector)
    A_deflated = A - eigenvalue * outer_product
    return A_deflated


def approximate_all_eigenvalues(A, max_iter=1000, tol=1e-6):
    eigenvalues = []
    n = A.shape[0]

    for _ in range(n):
        eigenvalue, eigenvector = power_method(A, max_iter, tol)
        eigenvalues.append(eigenvalue)
        A = deflate(A, eigenvalue, eigenvector)

    return eigenvalues


# Example usage
eigenvalues = approximate_all_eigenvalues(A)
print("Eigenvalues:", eigenvalues)
