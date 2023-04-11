import numpy as np
import matplotlib.pyplot as plt


def lagrange_interpolation(x, y):
    n = len(x)

    def L(k, x_i):
        out = 1.0
        for i in range(n):
            if i != k:
                out *= (x_i - x[i]) / (x[k] - x[i])
        return out

    def P(x_i):
        out = 0.0
        for k in range(n):
            out += y[k] * L(k, x_i)
        return out

    return P


# Define the function to interpolate
def f(x):
    return np.exp(-(x**2))


# Define the interval
a, b = -1.0, 1.0

# Define the values of N to use
N_values = [5, 10, 20, 40]


def show_lagrange():
    # Plot the results for each value of N
    for N in N_values:
        # Define the equidistant spaced nodes
        x = np.linspace(a, b, N + 1)
        y = f(x)

        # Compute the Lagrange polynomial
        P = lagrange_interpolation(x, y)

        # Evaluate the Lagrange polynomial at some points
        x_vals = np.linspace(a, b, 100)
        y_vals = P(x_vals)

        # Plot the results
        plt.plot(x_vals, y_vals, label=f"N={N}")

    # Plot the true function
    x_vals = np.linspace(a, b, 100)
    f_vals = f(x_vals)
    plt.plot(x_vals, f_vals, "--", label="True Function")

    plt.legend()
    plt.show()


def chebyshev_roots(N):
    """
    Compute the roots of the Chebyshev polynomial of degree N
    """
    n = np.arange(1, N + 1)
    x = np.cos((2 * n - 1) * np.pi / (2 * N))
    return x


def show_chebyshev():
    # Define the degree of the Chebyshev polynomial
    N = 10

    # Compute the Chebyshev nodes
    x = chebyshev_roots(N + 1)
    x = 0.5 * (a + b + (b - a) * x)

    # Evaluate f at the Chebyshev nodes
    y = f(x)

    # Compute the Lagrange polynomial
    P = lagrange_interpolation(x, y)

    # Evaluate the Lagrange polynomial at some points
    x_vals = np.linspace(a, b, 100)
    y_vals = P(x_vals)

    # Plot the results
    plt.plot(x_vals, y_vals, label="P_chev")
    plt.plot(x, y, "o", label="Nodes")
    # plt.plot(x_vals, f(x_vals), label='True Function')
    plt.legend()
    plt.show()


# Define the values of N to use
N_values = [4, 7, 10]


def show_err():
    # Define the x values at which to compute the errors
    x_vals = np.linspace(a, b, 100)

    # Compute the equidistant spaced nodes
    for N in N_values:
        x = np.linspace(a, b, N + 1)
        y = f(x)

        # Compute the Lagrange polynomial
        P_eq = lagrange_interpolation(x, y)

        # Compute the absolute errors
        err_eq = np.abs(f(x_vals) - P_eq(x_vals))

        # Plot the results
        plt.plot(x_vals, err_eq, label=f"N={N}")

    # Compute the Chebyshev nodes
    for N in N_values:
        x = chebyshev_roots(N + 1)
        x = 0.5 * (a + b + (b - a) * x)
        y = f(x)

        # Compute the Lagrange polynomial
        P_cheb = lagrange_interpolation(x, y)

        # Compute the absolute errors
        err_cheb = np.abs(f(x_vals) - P_cheb(x_vals))

        # Plot the results
        plt.plot(x_vals, err_cheb, "--", label=f"N={N}")

    plt.legend()
    plt.show()


show_chebyshev()
