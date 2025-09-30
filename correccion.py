# %%
import numpy as np
from dataclasses import dataclass
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ---------- Parámetros del modelo ----------
@dataclass
class Params:
    # Físicos/geométricos (pon tus valores en SI)
    rho: float = 1000.0      # [kg/m^3] densidad
    d: float = 0.5e-3        # [m] "altura" de la ranura/canal
    W: float = 5e-3          # [m] "ancho" de la ranura/canal
    mu: float = 1.0e-3       # [Pa·s] viscosidad
    L: float = 10e-3         # [m] longitud del canal
    R: float = 8.314         # [J/(mol·K)] constante gas ideal
    T: float = 298.0         # [K] temperatura
    H: float = 2.0e-3        # [m] altura máxima de la cámara
    A: float = 1.0e-4        # [m^2] área efectiva del diafragma

    # Mecánicos
    k1: float = 5e4          # [N/m]
    k2: float = 0.0          # [N/m^2] (no lineal opcional)
    c1: float = 5.0          # [N·s/m]
    c2: float = 0.0          # [N·s^2/m^2] (no lineal opcional)

    # Presión externa
    p_ext: float = 101325.0  # [Pa]

    # Excitación f(t) en la ecuación de x¨
    f_type: str = "pulse"     # "sine" | "pulse" | "const"
    f_amp: float = 50.0      # [N] amplitud fuerza
    f_freq: float = 30.0     # [Hz] solo para "sine"
    f_bias: float = 0.0      # [N] sesgo DC
    pulse_t0: float = 0.05   # [s]
    pulse_dt: float = 0.01   # [s]


def f_drive(t: float, P: Params) -> float:
    """Fuerza de excitación f(t) que entra en x¨."""
    if P.f_type == "sine":
        return P.f_bias + P.f_amp * np.sin(2*np.pi*P.f_freq*t)
    elif P.f_type == "pulse":
        return P.f_bias + (P.f_amp if (P.pulse_t0 <= t <= P.pulse_t0 + P.pulse_dt) else 0.0)
    else:  # "const"
        return P.f_bias + P.f_amp


def rhs(t, y, P: Params):
    """
    Sistema en primer orden:
      y = [x, v, p] = [x, dx/dt, p]
    Ecuaciones:
      x' = v
      v' = f(t) - k1 x - k2 x^2 - c1 v - c2 v^2 - A*(p - p_ext)
      p' = rho*p*((d*W^3)/(12*mu*L))*(R*T)/(d^2*(H - x)) + (p/(H - x))*v
    """
    x, v, p = y

    # Seguridad: evitar división por cero (H - x debe ser > 0)
    denom = P.H - x
    eps = 1e-9
    if denom < eps:
        denom = eps

    # Coeficiente del término viscoso/geométrico (según tu ecuación)
    Q_coeff = (P.d * P.W**3) / (12.0 * P.mu * P.L)

    # p'
    dpdt = P.rho * p * Q_coeff * (P.R * P.T) / (P.d**2 * denom) + (p / denom) * v

    # v'
    delta_p = p - P.p_ext
    dvdt = f_drive(t, P) - P.k1 * x - P.k2 * x**2 - P.c1 * v - P.c2 * v**2 - P.A * delta_p

    # x'
    dxdt = v

    return [dxdt, dvdt, dpdt]


def event_touch_ceiling(t, y, P: Params):
    """
    Evento para detener si x -> H (golpea el techo). Cuando value=0, se detiene.
    """
    x = y[0]
    margin = 1e-6  # 1 micra de margen
    return (P.H - margin) - x

event_touch_ceiling.terminal = True
event_touch_ceiling.direction = -1  # cruza decreciendo hacia 0


# ---------- Simulación ----------
if __name__ == "__main__":
    P = Params()

    # Condiciones iniciales
    x0 = 0.0            # [m]
    v0 = 0.0            # [m/s]
    p0 = P.p_ext        # [Pa] empezar en equilibrio
    y0 = [x0, v0, p0]

    # Ventana temporal
    t0, tf = 0.0, 0.2   # [s]
    t_eval = np.linspace(t0, tf, 4000)

    sol = solve_ivp(
        fun=lambda t, y: rhs(t, y, P),
        t_span=(t0, tf),
        y0=y0,
        t_eval=t_eval,
        events=lambda t, y: event_touch_ceiling(t, y, P),
        rtol=1e-7,
        atol=1e-9,
        method="RK45",   # Cambia a "Radau" si notas rigidez
    )

    if not sol.success:
        print("Integración no exitosa:", sol.message)

    t = sol.t
    x, v, p = sol.y

    # ---------- Gráficas ----------
    plt.figure()
    plt.plot(t, x)
    plt.xlabel("t [s]")
    plt.ylabel("x(t) [m]")
    plt.title("Desplazamiento del diafragma")

    plt.figure()
    plt.plot(t, v)
    plt.xlabel("t [s]")
    plt.ylabel("v(t) = dx/dt [m/s]")
    plt.title("Velocidad")

    plt.figure()
    plt.plot(t, p)
    plt.xlabel("t [s]")
    plt.ylabel("p(t) [Pa]")
    plt.title("Presión interna")

    # Retrato de fase (x vs v)
    plt.figure()
    plt.plot(x, v)
    plt.xlabel("x [m]")
    plt.ylabel("v [m/s]")
    plt.title("Retrato de fase")

    plt.show()
