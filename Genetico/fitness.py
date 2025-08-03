import cupy as cp # type: ignore
from config import d, dt
from lectura_datos import preprocesar_datos
from lectura_datos import leer_incendio_referencia
import sys
import os

# Agrega el directorio padre al path para importar módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modelo_rdc import spread_infection_raw

############################## CARGADO DE MAPAS ###############################################

datos = preprocesar_datos()

wx = datos["wx"]
wy = datos["wy"]
h_dx_mapa = datos["h_dx"]
h_dy_mapa = datos["h_dy"]
beta_veg = datos["beta_veg"].astype(cp.float32)
gamma = datos["gamma"].astype(cp.float32)
ny, nx = datos["ny"], datos["nx"]

############################## CÁLCULO DE FUNCIÓN DE FITNESS ###############################################

def aptitud_batch(parametros_batch, burnt_cells, num_steps=10000):
    """
    Calcula el fitness para múltiples combinaciones de parámetros en paralelo.
    
    Args:
        parametros_batch: Lista de tuplas (D, A, B, x, y)
    
    Returns:
        Lista de valores de fitness
    """
    batch_size = len(parametros_batch)
    
    # Inicializar arrays para el batch
    S_batch = cp.ones((batch_size, ny, nx), dtype=cp.float32)
    I_batch = cp.zeros((batch_size, ny, nx), dtype=cp.float32)
    R_batch = cp.zeros((batch_size, ny, nx), dtype=cp.float32)
    
    # Configurar puntos de ignición para cada simulación
    for i, (D, A, B, x, y) in enumerate(parametros_batch):
        S_batch[i, y, x] = 0
        I_batch[i, y, x] = 1
    
    # Arrays para los nuevos estados
    S_new_batch = cp.empty_like(S_batch)
    I_new_batch = cp.empty_like(I_batch)
    R_new_batch = cp.empty_like(R_batch)
    
    # Expandir arrays de parámetros para el batch (broadcast_to hace una copia eficiente)
    beta_veg_batch = cp.broadcast_to(beta_veg, (batch_size, ny, nx)) 
    gamma_batch = cp.broadcast_to(gamma, (batch_size, ny, nx))
    wx_batch = cp.broadcast_to(wx, (batch_size, ny, nx))
    wy_batch = cp.broadcast_to(wy, (batch_size, ny, nx))
    h_dx_batch = cp.broadcast_to(h_dx_mapa, (batch_size, ny, nx))
    h_dy_batch = cp.broadcast_to(h_dy_mapa, (batch_size, ny, nx))
    
    # Crear arrays de parámetros D, A, B para cada simulación
    D_batch = cp.array([param[0] for param in parametros_batch], dtype=cp.float32)
    A_batch = cp.array([param[1] for param in parametros_batch], dtype=cp.float32)  # A es el índice 3
    B_batch = cp.array([param[2] for param in parametros_batch], dtype=cp.float32)  # B es el índice 4

    # Simular en paralelo
    simulaciones_validas = cp.ones(batch_size, dtype=cp.bool_)

    print(f'Numero de pasos a simular: {num_steps}')
    
    for t in range(num_steps):
        # Llamar al kernel con todos los parámetros necesarios
        spread_infection_raw(
            S=S_batch, I=I_batch, R=R_batch, 
            S_new=S_new_batch, I_new=I_new_batch, R_new=R_new_batch,
            dt=dt, d=d, beta=beta_veg_batch, gamma=gamma_batch,
            D=D_batch, wx=wx_batch, wy=wy_batch, 
            h_dx=h_dx_batch, h_dy=h_dy_batch, A=A_batch, B=B_batch
        )
        
        # Intercambiar arrays
        S_batch, S_new_batch = S_new_batch, S_batch
        I_batch, I_new_batch = I_new_batch, I_batch
        R_batch, R_new_batch = R_new_batch, R_batch
        
        # Verificar si alguna simulación explota
        validas = cp.all((R_batch >= 0) & (R_batch <= 1), axis=(1, 2))
        simulaciones_validas &= validas
        
        # Si todas las simulaciones explotaron, terminar
        if not cp.any(simulaciones_validas):
            break
    
    # Calcular fitness para cada simulación en paralelo
    fitness_values = []
    
    # Crear máscaras para celdas quemadas (todo el batch de una vez)
    burnt_cells_sim_batch = cp.where(R_batch > 0.001, 1, 0)  # Shape: (batch_size, ny, nx)
    
    # Expandir burnt_cells para el batch
    burnt_cells_expanded = cp.broadcast_to(burnt_cells[cp.newaxis, :, :], (batch_size, ny, nx))
    
    # Calcular unión e intersección para todo el batch
    union_batch = cp.sum(burnt_cells_expanded | burnt_cells_sim_batch, axis=(1, 2))  # Shape: (batch_size,)
    interseccion_batch = cp.sum(burnt_cells_expanded & burnt_cells_sim_batch, axis=(1, 2))  # Shape: (batch_size,)
    
    # Calcular fitness para todo el batch
    burnt_cells_total = cp.sum(burnt_cells)
    fitness_batch = (union_batch - interseccion_batch) / burnt_cells_total
    
    # Procesar resultados
    for i in range(batch_size):
        if not simulaciones_validas[i]:
            fitness_values.append(float('inf'))
        else:
            fitness_values.append(float(fitness_batch[i]))
        
        # # Información de debug
        # D, A, B, x, y = parametros_batch[i]
        # if simulaciones_validas[i]:
        #     print(f'Sim {i}: fitness={fitness_batch[i]:.4f}, D={D:.4f}, A={A:.4f}, B={B:.4f}, x={x}, y={y}')
        #     print(f'  Celdas quemadas: {burnt_cells_total}, Simuladas: {cp.sum(burnt_cells_sim_batch[i])}')
        # else:
        #     print(f'Sim {i}: fitness=inf (simulación explotó), D={D:.4f}, A={A:.4f}, B={B:.4f}, x={x}, y={y}')
    
    return fitness_values