import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# constantes thermiques pour le cuivre
k = 385  # W/(m*K)
c = 385  # J/(kg*K)
rho = 8960  # kg/m^3

# taille de la plaque
L = 1  # m
W = 1  # m

# grille de la plaque
nx = 20
ny = 20
dx = L / (nx - 1)
dy = W / (ny - 1)

# conditions initiales
T = np.zeros((nx, ny))
T[int(nx / 2), int(ny / 2)] = 100

# source de chaleur
q = 50  # W/m^2

# paramètres temporels
dt = 0.01  # s
nt = 100

# démarrage de l'animation
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# boucle temporelle
for n in range(1, nt):
    Tn = T.copy()
    T[1:-1, 1:-1] = (Tn[1:-1, 1:-1] + k * dt / (c * rho * dx**2) *
                    (Tn[2:, 1:-1] - 2 * Tn[1:-1, 1:-1] + Tn[:-2, 1:-1] +
                     Tn[1:-1, 2:] - 2 * Tn[1:-1, 1:-1] + Tn[1:-1, :-2]) +
                    dt * q / (c * rho))

    # affichage de la température
    X, Y = np.meshgrid(np.linspace(0, L, nx), np.linspace(0, W, ny))
    ax.clear()
    cbar = fig.get_axes()[1]
    cbar.remove()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Temperature (K)')
    surf = ax.plot_surface(X, Y, T, cmap='jet')
    cbar = fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.draw()
    plt.pause(0.001)

plt.ioff()
plt.show()
