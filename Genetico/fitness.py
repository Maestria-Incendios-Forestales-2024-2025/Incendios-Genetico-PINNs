import cupy as cp # type: ignore
from modelo_rdc import spread_infection, courant
from config import d, dt, num_steps
from lectura_datos import preprocesar_datos

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

D = cp.float32(50) # Coeficiente de difusión
A = cp.float32(5e-4) # Coeficiente de advección por viento
B = cp.float32(15) # Coeficiente de advección por pendiente

############################## INCENDIO DE REFERENCIA ###############################################

R_host = cp.load("R_final.npy")
burnt_cells = cp.where(R_host > 0.5, 1, 0)

############################## CÁLCULO DE FUNCIÓN DE FITNESS ###############################################

def aptitud(D, A, B, x, y):
    # Chequeo que se cumpla la condición de courant
    while not courant(dt, D, A, B, d, wx, wy, h_dx=h_dx_mapa, h_dy=h_dy_mapa):
        # Seleccionar aleatoriamente qué parámetro reducir
        param_to_modify = cp.random.choice(["D", "A", "B"])

        if param_to_modify == "D":
            D *= cp.random.uniform(0.8, 0.99)
        elif param_to_modify == "A":
            A *= cp.random.uniform(0.8, 0.99)
        elif param_to_modify == "B":
            B *= cp.random.uniform(0.8, 0.99)

    while vegetacion[x.astype(cp.int32), y.astype(cp.int32)] <= 2:
        x, y = cp.random.randint(500, 900), cp.random.randint(500, 900)

    # Población inicial de susceptibles e infectados
    S_i = cp.ones((ny, nx), dtype=cp.float32)  # Todos son susceptibles inicialmente
    I_i = cp.zeros((ny, nx), dtype=cp.float32) # Ningún infectado al principio
    R_i = cp.zeros((ny, nx), dtype=cp.float32)

    # Si hay combustible, encender fuego
    S_i[x.astype(cp.int32), y.astype(cp.int32)] = 0
    I_i[x.astype(cp.int32), y.astype(cp.int32)] = 1

    S_new_i = cp.empty_like(S_i)
    I_new_i = cp.empty_like(I_i)
    R_new_i = cp.empty_like(R_i)

    # Iterar sobre las simulaciones
    for t in range(num_steps):
        spread_infection(S=S_i, I=I_i, R=R_i, S_new=S_new_i, I_new=I_new_i, R_new=R_new_i, 
                         dt=dt.astype(cp.float32), d=d.astype(cp.float32), beta=beta_veg, gamma=gamma, 
                         D=D.astype(cp.float32), wx=wx, wy=wy, h_dx=h_dx_mapa, h_dy=h_dy_mapa, A=A.astype(cp.float32), B=B.astype(cp.float32))
        
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