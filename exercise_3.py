import numpy as np


def seidel(a, x, b, max_iter=10):
    n = len(a)
    for j in range(0, n):
        d = b[j]
        for i in range(0, n):
            if j != i:
                d -= a[j][i] * x[i]

        x[j] = d / a[j][j]
    for i in range(0, max_iter):
        x = seidel(a, x, b)
    return x


x = [0, 0, 0]
a = [[3, 1, -1], [2, 4, 1], [-1, 2, 5]]
b = [4, 1, 1]
iter = 10


A = [
    [3, -1, 0, 0, 0, 1 / 2],
    [-1, 3, -1, 0, 1 / 2, 0],
    [0, -1, 3, -1, 0, 0],
    [0, 0, -1, 3, -1, 0],
    [0, 1 / 2, 0, -1, 3, -1],
    [1 / 2, 0, 0, 0, -1, 3],
]

b = [5 / 2, 3 / 2, 1, 1, 3 / 2, 5 / 2]

A = np.array(A)
b = np.array(b)
x0 = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
print(A)
print(b)


def jacobi_method(A, b, x0, tol=1e-6, max_iter=10000):
    """
    Solves a linear system of equations Ax = b using the Jacobi method.
    :param A: n x n matrix
    :param b: n x 1 vector
    :param x0: n x 1 vector, initial guess
    :param tol: float, tolerance for convergence
    :param max_iter: int, maximum number of iterations
    :return: n x 1 vector, solution to the linear system
    """
    n = len(b)
    x = x0.copy()
    for k in range(max_iter):
        x_new = np.zeros_like(x)
        for i in range(n):
            s = np.dot(A[i, :], x) - A[i, i] * x[i]
            x_new[i] = (b[i] - s) / A[i, i]
        print(x_new - x)
        if np.max(np.abs(x_new - x)) < tol:
            return x_new
        x = x_new.copy()
    raise ValueError("Jacobi method did not converge in {} iterations".format(max_iter))


# print(jacobi_method(A, b, x0))


def sor_method(A, b, x0, omega, tol=1e-6, max_iter=1000):
    """
    Solves a linear system of equations Ax = b using the Successive Over-Relaxation (SOR) method.
    :param A: n x n matrix
    :param b: n x 1 vector
    :param x0: n x 1 vector, initial guess
    :param omega: float, relaxation parameter (0 < omega < 2)
    :param tol: float, tolerance for convergence
    :param max_iter: int, maximum number of iterations
    :return: n x 1 vector, solution to the linear system
    """
    n = len(b)
    x = x0.copy()
    for k in range(max_iter):
        for i in range(n):
            s = np.dot(A[i, :], x) - A[i, i] * x[i]
            x[i] = (1 - omega) * x[i] + (omega / A[i, i]) * (b[i] - s)
        if np.max(np.abs(A.dot(x) - b)) < tol:
            return x
    raise ValueError("SOR method did not converge in {} iterations".format(max_iter))


omega = 1.1

print(sor_method(A, b, x0, omega))
