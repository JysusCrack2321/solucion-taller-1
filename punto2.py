import math
import numpy as np


# Volúmenes de los tanques (m^3)
V1 = 1.0
V2 = 1.2
V3 = 1.5

# Concentraciones iniciales (kg/m^3)
C1_0 = 20.0
C2_0 = 5.0
C3_0 = 0.0

# Conversión de caudales: L/min -> m^3/s
def Lmin_to_m3s(Q_Lmin):
    return (Q_Lmin / 1000.0) / 60.0

# Caudales externos (m^3/s)
Q0 = Lmin_to_m3s(0.3)
Q1 = Lmin_to_m3s(1.5)
Q2 = Lmin_to_m3s(1.8)
Q3 = Lmin_to_m3s(0.6)

# Caudales internos (calculados previamente por balance) en L/min -> m^3/s
QA = Lmin_to_m3s(18.72)
QB = Lmin_to_m3s(15.84)
QC = Lmin_to_m3s(12.0)

# Concentraciones de alimentación (kg/m^3)
c_Q0 = 0.0                    # Línea 0
c_Q2 = 6.25e3 / 1000.0        # Línea 2: 6.25×10^3 g/m^3 -> 6.25 kg/m^3

# Concentración de Línea 1 es variable:
# c_Q1(t) = c1*(2.0 + sin(0.003491 t) + 0.3 sin(0.005236 t))
# con c1 = 2.5e4 g/m^3 = 25 kg/m^3

c1 = 2.5e4 / 1000.0   # 25 kg/m^3

def c_Q1(t):
    return c1 * (2.0 + np.sin(0.003491*t) + 0.3*np.sin(0.005236*t))

# ------------------------------
# Verificación imprimiendo constantes
# ------------------------------
if __name__ == "__main__":
    print("Volúmenes [m^3]:", V1, V2, V3)
    print("Concentraciones iniciales [kg/m^3]:", C1_0, C2_0, C3_0)
    print("Caudales externos [m^3/s]: Q0=%.6f, Q1=%.6f, Q2=%.6f, Q3=%.6f" % (Q0, Q1, Q2, Q3))
    print("Caudales internos [m^3/s]: QA=%.6f, QB=%.6f, QC=%.6f" % (QA, QB, QC))
    print("c_Q0 =", c_Q0, "kg/m^3")
    print("c_Q2 =", c_Q2, "kg/m^3")
    print("Ejemplo c_Q1(t=0) =", c_Q1(0.0), "kg/m^3")
