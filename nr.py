from math import exp, sin, cos, pow

import numpy as np

def f(x):
    return exp(x) + sin(x) - 4
    return x**3 -x
    s = sin(x)
    return exp(s**3) + x**6 - 2*x**4 - x**3 - 1
    #return x*x*x - 6*x*x + 4*x + 12

def fp(x):
    return 3*x*x - 1
    return (6*x**3 - 8*x - 3)*x*x + 3*exp(sin(x)**3)*sin(x)*sin(x)*cos(x)

    #return 3*x*x - 12*x + 4

def newton(x0, k, p=False):
    x = list([x0])
    x.append(x0 - f(x0)/fp(x0))
    for i in range (1, k+1):
        v = x[i] - f(x[i])/fp(x[i])
        x.append(v)
        if p:
            print(f"x_{i+1}: {v:.52f}")
    return x[-1]

def secant(x0, x1, k):
    x = list([x0, x1])
    for i in range(1, k):
        denom = (f(x[i]) - f(x[i-1]))
        if denom == 0:
            break
        x_i_1 = x[i] - (f(x[i])*(x[i] - x[i-1])) / denom
        x.append(x_i_1)
        print(f"{x_i_1:.17f}")
    return x[-1]

def back_sub(U, b):
    n = len(b)
    x = [0]*n
    for i in range(n - 1, -1, -1):
        s = sum(U[i][j] * x[j] for j in range(i+1, n))
        x[i] = (b[i] - s) / U[i][i]
    return x

def forward_sub(L, b):
    n = len(b)
    x = [0]*n
    for i in range(n):
        x[i] = b[i]
        for j in range(i):
            x[i] -= L[i][j]*x[j]
        x[i] /= L[i][i]
    return x

def compute_U(A, b):
    n = len(b)
    # Make a copy of A and b to avoid modifying the original matrices
    Ab = [row[:] for row in A]
    bb = b[:]

    # Forward elimination
    for i in range(n):
        # Find pivot row
        pivot_row = max(range(i, n), key=lambda j: abs(Ab[j][i]))
        # Swap pivot row with current row (if needed)
        if pivot_row != i:
            Ab[i], Ab[pivot_row] = Ab[pivot_row], Ab[i]
            bb[i], bb[pivot_row] = bb[pivot_row], bb[i]
        # Eliminate entries below pivot
        for j in range(i+1, n):
            factor = Ab[j][i] / Ab[i][i]
            for k in range(i, n):
                Ab[j][k] -= factor * Ab[i][k]
            bb[j] -= factor * bb[i]
    return Ab, bb

def gaussian_elimination(A, b):
    """
    Solve the system Ax = b using Gaussian elimination.
    A: an n x n matrix
    b: a column vector of length n
    Returns: a column vector x of length n such that Ax = b
    """
    U, b = compute_U(A, b)
    return back_sub(U, b)

def LU_decomposition(A):
    """
    Perform LU decomposition of the matrix A.
    A: an n x n matrix
    Returns: a tuple (L, U) where L is an n x n lower triangular matrix and U is an n x n upper triangular matrix such that A = LU
    """
    n = len(A)
    # Initialize L and U
    L = [[0] * n for _ in range(n)]
    U = [[0] * n for _ in range(n)]
    for i in range(n):
        L[i][i] = 1

    # Perform Gaussian elimination with partial pivoting
    for j in range(n):
        max_row = j
        for i in range(j+1, n):
            if abs(A[i][j]) > abs(A[max_row][j]):
                max_row = i
        A[j], A[max_row] = A[max_row], A[j]
        L[j], L[max_row] = L[max_row], L[j]
        U[j][j] = A[j][j]
        for i in range(j+1, n):
            L[i][j] = A[i][j] / U[j][j]
            U[j][i] = A[j][i]
        for i in range(j+1, n):
            for k in range(j+1, n):
                A[i][k] -= L[i][j] * U[j][k]

    return (L, U)

def solve_lu(A, b):
    """
    Solve the system Ax = b using LU decomposition.
    A: an n x n matrix
    b: a column vector of length n
    Returns: a column vector x of length n such that Ax = b
    """
    n = len(A)
    # Perform LU decomposition of A
    L, U = LU_decomposition(A)

    # Solve Ly = b for y using forward substitution
    y = [0] * n
    for i in range(n):
        s = sum(L[i][j] * y[j] for j in range(i))
        y[i] = b[i] - s

    # Solve Ux = y for x using backward substitution
    x = [0] * n
    for i in range(n-1, -1, -1):
        s = sum(U[i][j] * x[j] for j in range(i+1, n))
        x[i] = (y[i] - s) / U[i][i]

    return x


#A = [[1,2,-1],[2,1,-2],[-3,1,1]]
A = [[4,2,0], [4,4,2], [2,2,3]]
L = [[1,0,0],[2,1,0],[-3,-7/3,1]]
#b = [3,3,-6]
b = [1,2,3]
#U = compute_U(A, b)#[[1,2,-1],[0,-3,0],[0,0,-2]]

print(solve_lu(A, b))



#b = forward_sub(L, b)
#x = back_sub(U, b)
#print(x)
