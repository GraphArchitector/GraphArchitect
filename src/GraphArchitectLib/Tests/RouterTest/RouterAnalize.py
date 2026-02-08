import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Диапазоны значений
L_n = np.linspace(0.01, 50, 300)  # от 10 до 10000
C_n = np.linspace(1, 2, 130)  # от 1 до 100
Q_n = 1.8  # фиксированное качество
t_sc = 1.0  # масштабный коэффициент
w_Q = 1.0   # вес качества
w_c = 1.0   # вес стоимости
w_l = 1.0   # вес длины

# сетка значений
L_grid, C_grid = np.meshgrid(L_n, C_n)

# R_n
R_n = (t_sc * (w_Q * Q_n - w_c * C_grid)) / (w_l * np.log10(L_grid+2))

# 3D график
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(L_grid, C_grid, R_n,
                      cmap='viridis',
                      edgecolor='none')

# Настройка осей
ax.set_xlabel('L_n')
ax.set_ylabel('C_n')
ax.set_zlabel('R_n')
ax.set_title(f'(Q_n = {Q_n})')


fig.colorbar(surf)

plt.show()