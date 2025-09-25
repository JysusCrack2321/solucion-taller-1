import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class SistemaTanques:
    def __init__(self):
        # Parámetros del sistema - TODOS en m³/s y kg/m³
        self.V1, self.V2, self.V3 = 1.0, 1.2, 1.5  # m³
        
        # Conversión L/min → m³/s
        def Lmin_to_m3s(x):
            return x / 1000.0 / 60.0
        
        # Caudales [m³/s]
        self.Q0 = Lmin_to_m3s(0.3)
        self.Q1 = Lmin_to_m3s(1.5)
        self.Q2 = Lmin_to_m3s(1.8)
        self.Q3 = Lmin_to_m3s(0.6)
        self.QA = Lmin_to_m3s(18.72)
        self.QB = Lmin_to_m3s(15.84)
        self.QC = Lmin_to_m3s(12.0)
        
        # Concentraciones fijas [kg/m³]
        self.c_Q2 = 6.25  # kg/m³
        
        print("=== PARÁMETROS DEL SISTEMA ===")
        print(f"Volúmenes: V1={self.V1}, V2={self.V2}, V3={self.V3} [m³]")
        print(f"Caudales [m³/s]: Q0={self.Q0:.6f}, Q1={self.Q1:.6f}, Q2={self.Q2:.6f}")
        print(f"QA={self.QA:.6f}, QB={self.QB:.6f}, QC={self.QC:.6f}, Q3={self.Q3:.6f}")
    
    def c_Q1(self, t):
        """Concentración variable en Q1 [kg/m³]"""
        # c1 = 25,000 g/m³ = 25 kg/m³
        c1_base = 25.0
        return c1_base * (2.0 + np.sin(0.003491*t) + 0.3*np.sin(0.005236*t))
    
    def sistema_edos(self, t, C):
        """
        Sistema de EDOs: dC/dt = f(t, C)
        C = [C1, C2, C3] - concentraciones en cada tanque [kg/m³]
        """
        C1, C2, C3 = C
        
        # TANQUE 1: dC1/dt
        entrada_T1 = (self.Q0 * 0 +                    # Q0 (agua pura)
                     self.Q1 * self.c_Q1(t) +          # Q1 con c_Q1(t)
                     0.75 * self.QC * C3 +             # 3/4 QC recirculado
                     0.5 * self.QB * C2)               # 1/2 QB de T2
        
        salida_T1 = self.QA * C1
        dC1dt = (entrada_T1 - salida_T1) / self.V1
        
        # TANQUE 2: dC2/dt  
        entrada_T2 = (0.75 * self.QA * C1 +            # 3/4 QA de T1
                     self.Q2 * self.c_Q2)              # Q2 con c_Q2 = 6.25
        
        salida_T2 = self.QB * C2
        dC2dt = (entrada_T2 - salida_T2) / self.V2
        
        # TANQUE 3: dC3/dt
        entrada_T3 = (0.5 * self.QB * C2 +             # 1/2 QB de T2
                     0.25 * self.QA * C1)              # 1/4 QA de T1
        
        salida_T3 = (self.Q3 + self.QC) * C3           # Q3 + QC total salida
        dC3dt = (entrada_T3 - salida_T3) / self.V3
        
        return [dC1dt, dC2dt, dC3dt]
    
    def resolver_rk4(self, t_span=(0, 8*3600), C0=[20.0, 5.0, 0.0], n_points=1000):
        """Resuelve el sistema usando RK4 (solve_ivp)"""
        
        # Tiempo de evaluación
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        
        # Resolver con RK45 (Runge-Kutta de 4º-5º orden)
        sol = solve_ivp(self.sistema_edos, t_span, C0, 
                       method='RK45', t_eval=t_eval, rtol=1e-6, atol=1e-8)
        
        return sol.t, sol.y
    
    def graficar_resultados(self, t, C):
        """Grafica las concentraciones vs tiempo"""
        C1, C2, C3 = C
        
        plt.figure(figsize=(12, 8))
        
        # Concentraciones vs tiempo
        plt.subplot(2, 1, 1)
        plt.plot(t/3600, C1, label='C1 - Tanque 1', linewidth=2)
        plt.plot(t/3600, C2, label='C2 - Tanque 2', linewidth=2)
        plt.plot(t/3600, C3, label='C3 - Tanque 3', linewidth=2)
        
        plt.xlabel('Tiempo [horas]')
        plt.ylabel('Concentración [kg/m³]')
        plt.title('Evolución de Concentraciones - Sistema de 3 Tanques')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # c_Q1(t) vs tiempo
        plt.subplot(2, 1, 2)
        c_Q1_vals = [self.c_Q1(ti) for ti in t]
        plt.plot(t/3600, c_Q1_vals, 'r--', label='c_Q1(t) - Entrada T1', linewidth=2)
        plt.xlabel('Tiempo [horas]')
        plt.ylabel('Concentración Q1 [kg/m³]')
        plt.title('Concentración de Entrada Q1')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def analizar_resultados(self, t, C):
        """Análisis de los resultados"""
        C1, C2, C3 = C
        
        print("\n=== ANÁLISIS DE RESULTADOS ===")
        print(f"Concentración final C1: {C1[-1]:.2f} kg/m³")
        print(f"Concentración final C2: {C2[-1]:.2f} kg/m³")
        print(f"Concentración final C3: {C3[-1]:.2f} kg/m³")
        print(f"Concentración línea Q3: {C3[-1]:.2f} kg/m³")
        
        # Verificar balance de masa
        masa_final = C1[-1]*self.V1 + C2[-1]*self.V2 + C3[-1]*self.V3
        print(f"Masa total en sistema: {masa_final:.2f} kg")

# 🚀 EJECUCIÓN PRINCIPAL
if __name__ == "__main__":
    # Crear sistema
    sistema = SistemaTanques()
    
    # Condiciones iniciales [kg/m³]
    C0 = [20.0, 5.0, 0.0]
    
    # Resolver por 8 horas (28800 segundos)
    t, C = sistema.resolver_rk4(t_span=(0, 100*3600), C0=C0)
    
    # Graficar resultados
    sistema.graficar_resultados(t, C)
    
    # Análisis
    sistema.analizar_resultados(t, C)