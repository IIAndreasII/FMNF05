import matplotlib.pyplot as plt
from math import cos, sin, exp, sqrt, pi
import sys
import math

sys.setrecursionlimit(50000)

def f(x):
    return cos(x)

a = 0
b = 10
M = 64

def composite_simpsons_rule(f, a, b, M):
    h = (b-a)/M
    y = [a + m*h for m in range(M+1)]
    integral_m = [(h/6)*(f(y[m-1]) + 4*f((y[m-1]+y[m])/2) + f(y[m])) for m in range(1, M+1)]
    approx_integral = sum(integral_m)

    # Create lists of x and y coordinates for each point
    x_coords = []
    y_coords = []
    for m in range(M+1):
        x_coords.append(a + m*h)
        y_coords.append(f(x_coords[m]))

    # Plot the points
    plt.scatter(x_coords, y_coords)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Composite Simpson\'s Rule')
    #plt.show()

    return approx_integral

def trapezoidal_rule(f, a, b, h):
    x = [a + i*h for i in range(int((b-a)/h)+1)]
    return h*(sum(f(xi) for xi in x) - 0.5*f(a) - 0.5*f(b))

def composite_quadrature_method(f, a, b, M):
    h = (b-a)/M
    CQM = 0
    for m in range(M):
        x1 = a + m*h
        x2 = a + (m+1)*h
        CQM += trapezoidal_rule(f, x1, x2, h/2)
    return CQM

I = composite_simpsons_rule(f, a, b, M)
step_size = 1

print("| Step size (h) | Approximation (CQM[f]) | Exact (I[f]) | Absolute error (|CQM[f] - I[f]|) |")
for i in range(4):
    h = step_size / 2**i
    CQM = composite_quadrature_method(f, a, b, int((b-a)/h))
    abs_err = abs(CQM - I)
    print(f"| {h:.4f}        | {CQM:.6f}              | {I:.6f}    | {abs_err:.6f}                         |")


def adaptive_trapezoid(f, a, b, tol):
    # Initial subinterval
    subintervals = [(a, b)]
    # Initial approximation
    approx = 0
    # Initialize error estimate
    error = tol + 1

    while error > tol:
        # Approximate integral over each subinterval using trapezoidal rule
        approx_new = 0
        for (a_i, b_i) in subintervals:
            h = (b_i - a_i) / 2
            x0 = a_i
            x1 = a_i + h
            x2 = b_i
            approx_new += h * (f(x0) + f(x2) + 2*f(x1)) / 2

        # Compute error estimate
        error = abs(approx_new - approx)

        # Check if error is within tolerance
        if error <= tol:
            break

        # Subdivide each subinterval
        subintervals_new = []
        for (a_i, b_i) in subintervals:
            h = (b_i - a_i) / 2
            x0 = a_i
            x1 = a_i + h
            x2 = b_i
            subintervals_new.append((a_i, x1))
            subintervals_new.append((x1, b_i))

        # Update subintervals and approximation
        subintervals = subintervals_new
        approx = approx_new

    return approx


tol = 0.5*10**(-8)

a = lambda x: exp(x**2)
b = lambda x: sin(x**2)
c = lambda x: x**x

a_interval = (0, 1)
b_interval = (0, sqrt(pi))
c_interval = (0, 1)

print(adaptive_trapezoid(a, a_interval[0], a_interval[1], tol))
print(adaptive_trapezoid(b, b_interval[0], b_interval[1], tol))
print(adaptive_trapezoid(c, c_interval[0], c_interval[1], tol))


# TODO: un-nest the function
def adaptive_simpsons(f, a, b, tol, max_depth):
    c = (a + b) / 2.0
    fa = f(a)
    fb = f(b)
    fc = f(c)

    def simpsons_rule(fa, fb, fc, h):
        return (h * (fa + 4.0 * fc + fb)) / 6.0

    def adaptive_simpsons_recursive(f, a, b, fa, fb, fc, tol, depth):
        c = (a + b) / 2.0
        h = b - a
        fd = f(c - h / 4.0)
        fe = f(c + h / 4.0)
        fcd = f(c)
        S = simpsons_rule(fa, fcd, fd, h / 2.0) + simpsons_rule(fcd, fb, fe, h / 2.0)

        if abs(S - simpsons_rule(fa, fb, fc, h)) < 15.0 * tol or depth > max_depth:
            return S + (S - simpsons_rule(fa, fb, fc, h)) / 15.0
        return adaptive_simpsons_recursive(f, a, c, fa, fd, fc, tol / 2.0, depth + 1) + adaptive_simpsons_recursive(f, c, b, fc, fe, fb, tol / 2.0, depth + 1)

    return adaptive_simpsons_recursive(f, a, b, fa, fb, fc, tol, 0)

def normal_distribution(x):
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-x**2 / 2.0)

tolerance = 0.5 * 10**(-8)

P_1 = adaptive_simpsons(normal_distribution, -1, 1, tolerance, 10)
P_2 = adaptive_simpsons(normal_distribution, -2, 2, tolerance, 10)
P_3 = adaptive_simpsons(normal_distribution, -3, 3, tolerance, 10)

print("The probability to be within 1 standard deviation: ", P_1)
print("The probability to be within 2 standard deviations: ", P_2)
print("The probability to be within 3 standard deviations: ", P_3)



