import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

A = np.array([[1, 0, 1], [1, 0, 1], [1, 1, 0], [1, 1, 0], [1, 1, 2]])
b = np.array([3, 2, 3, 4, 6])

Q, R = np.linalg.qr(A)
x = np.linalg.solve(R, Q.T @ b)

x_vals = np.linspace(-1, 2, 50)
y_vals = np.linspace(-1, 3, 50)
x_mesh, y_mesh = np.meshgrid(x_vals, y_vals)

z_mesh = x[0] + x[1] * x_mesh + x[2] * y_mesh

c0 = x[0]
c1 = x[1]
c2 = x[2]

print(f"z = {x[0]:.4f} + {x[1]:.4f}x + {x[2]:.4f}y")


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.scatter(A[:, 1], A[:, 2], b, c="red", marker="o", label="Data points")
ax.plot_surface(x_mesh, y_mesh, z_mesh, alpha=0.5)

ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
ax.set_zlabel("Z axis")

ax.legend()

plt.show()
