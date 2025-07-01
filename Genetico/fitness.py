import cupy as cp # type: ignore
from config import d, dt, num_steps
from lectura_datos import preprocesar_datos
import sys
import os

# Agrega el directorio padre al path para importar módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modelo_rdc import spread_infection_raw, courant

############################## CARGADO DE MAPAS ###############################################

datos = preprocesar_datos()

wx = datos["wx"]
wy = datos["wy"]
h_dx_mapa = datos["h_dx"]
h_dy_mapa = datos["h_dy"]
vegetacion = datos["vegetacion"]
beta_veg = datos["beta_veg"].astype(cp.float32)
gamma = datos["gamma"].astype(cp.float32)
ny, nx = datos["ny"], datos["nx"]

############################## INCENDIO DE REFERENCIA ###############################################

R_host = cp.load("R_final.npy")
burnt_cells = cp.where(R_host > 0.5, 1, 0)

############################## CÁLCULO DE FUNCIÓN DE FITNESS ###############################################

def aptitud(D, A, B, x, y):
    # Los parámetros ya deben cumplir Courant y (x,y) debe ser un punto válido
    
    # Población inicial de susceptibles e infectados
    S_i = cp.ones((ny, nx), dtype=cp.float32)  # Todos son susceptibles inicialmente
    I_i = cp.zeros((ny, nx), dtype=cp.float32) # Ningún infectado al principio
    R_i = cp.zeros((ny, nx), dtype=cp.float32)

    # Si hay combustible, encender fuego
    S_i[x, y] = 0
    I_i[x, y] = 1

    S_new_i = cp.empty_like(S_i)
    I_new_i = cp.empty_like(I_i)
    R_new_i = cp.empty_like(R_i)

    # Iterar sobre las simulaciones
    for t in range(num_steps):
        spread_infection_raw(S=S_i, I=I_i, R=R_i, S_new=S_new_i, I_new=I_new_i, R_new=R_new_i, 
                         dt=dt, d=d, beta=beta_veg, gamma=gamma, 
                         D=D, wx=wx, wy=wy, h_dx=h_dx_mapa, h_dy=h_dy_mapa, A=A, B=B)
        
        S_i, S_new_i = S_new_i, S_i
        I_i, I_new_i = I_new_i, I_i
        R_i, R_new_i = R_new_i, R_i

        # Esto va a decir si la simulación explota o no
        if not (cp.all((R_i >= 0) & (R_i <= 1))):
            break

    if not cp.all((R_i >= 0) & (R_i <= 1)):
        return float('inf') # Pasa a la siguiente combinación sin guardar resultados

    # Celdas quemadas en el incendio simulado: si R_i > 0.5 esa celda está quemada
    burnt_cells_sim = cp.where(R_i > 0.5, 1, 0)

    union = cp.sum((burnt_cells | burnt_cells_sim))  # Celdas quemadas en al menos un mapa (unión)
    interseccion = cp.sum((burnt_cells & burnt_cells_sim))

    # Calcular el fitness
    fitness = (union - interseccion) / cp.sum(burnt_cells)

    return fitness