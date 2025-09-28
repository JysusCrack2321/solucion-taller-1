

import numpy as np
from scipy.interpolate import interp1d  
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

temperaturas = [0, 25, 75, 100]  # °C
solubilidades = [179, 211, 340, 487]  # g
# g/100mL → g/m³ (×10 por 100mL→L, ×1000 L→m³)

def func_solubilidad(temp):
    """Interpolación cuadrática de solubilidad en función de temperatura"""
    return interp1d(temperaturas, solubilidades, kind='quadratic', fill_value='extrapolate')(temp)
# Ejemplo de uso e impresión de la función de solubilidad

temps=np.arange(0,101,5)
solubilidades2=np.zeros_like(temps, dtype=float)
for i, t in enumerate(temps):
    solubilidades2[i] = func_solubilidad(t)
plt.plot(temps, solubilidades2, 'b-', label='Solubilidad interpolada')
plt.plot(temperaturas, solubilidades, 'ro', label='Datos de tabla')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Solubilidad (g/m³)')
plt.title('Solubilidad de azúcar en agua vs Temperatura')
plt.legend()
plt.grid()
plt.show()