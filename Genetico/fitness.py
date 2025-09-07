import cupy as cp # type: ignore
from config import d, dt
from lectura_datos import preprocesar_datos
import cupyx.scipy.ndimage # type: ignore
import sys
import os

# Agrega el directorio padre al path para importar módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modelo_rdc import spread_infection_adi

############################## BATCH #########################################################

def create_batch(array_base, n_batch):
    # Se repite array_base n_batch veces en un bloque contiguo
    return cp.tile(array_base[cp.newaxis, :, :], (n_batch, 1, 1)).copy()

############################## CARGADO DE MAPAS ###############################################

datos = preprocesar_datos()

vegetacion = datos["vegetacion"]
wx = datos["wx"]
wy = datos["wy"]
h_dx_mapa = datos["h_dx"]
h_dy_mapa = datos["h_dy"]
ny, nx = datos["ny"], datos["nx"]

############################## FUNCIÓN PARA MAPEAR PARÁMETROS DE VEGETACIÓN ###############################################

def crear_mapas_parametros_batch(parametros_batch, vegetacion, ajustar_beta_gamma=True, beta_fijo=None, gamma_fijo=None):
    """
    Crea mapas de beta y gamma personalizados para cada simulación en el batch
    basados en los parámetros optimizados y el tipo de vegetación.
    
    Args:
        parametros_batch: Lista de tuplas (D, A, B, x, y, beta_params, gamma_params)
        vegetacion: Mapa de tipos de vegetación (ny, nx)
    
    Returns:
        beta_batch, gamma_batch: Arrays (batch_size, ny, nx) con valores mapeados
    """
    batch_size = len(parametros_batch)
    
    # Crear arrays de salida
    beta_batch = cp.zeros((batch_size, ny, nx), dtype=cp.float32)
    gamma_batch = cp.zeros((batch_size, ny, nx), dtype=cp.float32)
    
    # Obtener tipos únicos de vegetación
    veg_types = cp.array([3, 4, 5, 6, 7], dtype=cp.int32)
    
    # Para cada simulación en el batch
    for i, params in enumerate(parametros_batch):
        # Inicializar con valores por defecto
        beta_map = cp.zeros_like(vegetacion, dtype=cp.float32)
        gamma_map = cp.zeros_like(vegetacion, dtype=cp.float32)
        if ajustar_beta_gamma and len(params) == 7: # Exp2
            beta_val = params[5]  # un solo valor escalar
            gamma_val = params[6]  # un solo valor escalar
            beta_map = cp.full_like(vegetacion, beta_val, dtype=cp.float32)
            gamma_map = cp.full_like(vegetacion, gamma_val, dtype=cp.float32)
        elif ajustar_beta_gamma and len(params) == 5: # Exp3
            beta_params = params[3]  # Lista de betas por tipo de vegetación
            gamma_params = params[4]  # Lista de gammas por tipo de vegetación
            
            # Mapear valores según tipo de vegetación
            for j, veg_type in enumerate(veg_types):
                veg_type = int(veg_type)
                # Crear máscara para este tipo de vegetación
                mask = (vegetacion == veg_type)
                
                # Asignar valores si tenemos parámetros para este tipo
                if j < len(beta_params) and j < len(gamma_params):
                    beta_map = cp.where(mask, beta_params[j], beta_map)
                    gamma_map = cp.where(mask, gamma_params[j], gamma_map)
        else: 
            beta_params = beta_fijo  # Lista de betas por tipo de vegetación
            gamma_params = gamma_fijo  # Lista de gammas por tipo de vegetación
            
            # Mapear valores según tipo de vegetación
            for j, veg_type in enumerate(veg_types):
                veg_type = int(veg_type)
                # Crear máscara para este tipo de vegetación
                mask = (vegetacion == veg_type)
                
                # Asignar valores si tenemos parámetros para este tipo
                if j < len(beta_params) and j < len(gamma_params):
                    beta_map = cp.where(mask, beta_params[j], beta_map)
                    gamma_map = cp.where(mask, gamma_params[j], gamma_map)
        
        beta_batch[i] = beta_map
        gamma_batch[i] = gamma_map

    sigma = (0, 10.0, 10.0)

    # Suavizado de los mapas
    beta_batch = cupyx.scipy.ndimage.gaussian_filter(beta_batch, sigma=sigma)
    gamma_batch = cupyx.scipy.ndimage.gaussian_filter(gamma_batch, sigma=sigma)

    return beta_batch, gamma_batch

############################## CÁLCULO DE FUNCIÓN DE FITNESS POR BATCH ###############################################

def aptitud_batch(parametros_batch, burnt_cells, num_steps=10000, ajustar_beta_gamma=True, beta_fijo=None, gamma_fijo=None,
                  ajustar_ignicion=True, ignicion_fija_x=None, ignicion_fija_y=None):
    """
    Calcula el fitness para múltiples combinaciones de parámetros en paralelo.
    
    Args:
        parametros_batch: Lista de tuplas (D, A, B, x, y, beta_params, gamma_params)
    
    Returns:
        Lista de valores de fitness
    """
    
    # Resto del código original...
    batch_size = len(parametros_batch)

    print(f'Batch size: {batch_size}')
    
    # Inicializar arrays para el batch
    S_batch = cp.ones((batch_size, ny, nx), dtype=cp.float32)
    I_batch = cp.zeros((batch_size, ny, nx), dtype=cp.float32)
    R_batch = cp.zeros((batch_size, ny, nx), dtype=cp.float32)
    
    # Configurar puntos de ignición para cada simulación
    for i, params in enumerate(parametros_batch):
        # Extraer coordenadas (D, A, B, x, y, ...)
        if ajustar_ignicion:
            x, y = int(params[3]), int(params[4])
            S_batch[i, y, x] = 0
            I_batch[i, y, x] = 1
        else: 
            x, y = ignicion_fija_x, ignicion_fija_y
            S_batch[i, y, x] = 0
            I_batch[i, y, x] = 1

    # Arrays para los nuevos estados
    S_new_batch = cp.empty_like(S_batch)
    I_new_batch = cp.empty_like(I_batch)
    R_new_batch = cp.empty_like(R_batch)
    
    # Crear mapas de parámetros de vegetación personalizados
    beta_batch, gamma_batch = crear_mapas_parametros_batch(parametros_batch, vegetacion, ajustar_beta_gamma=ajustar_beta_gamma,
                                                           beta_fijo=beta_fijo, gamma_fijo=gamma_fijo)

    # Crear arrays de parámetros D, A, B para cada simulación
    D_batch = cp.array([param[0] for param in parametros_batch], dtype=cp.float32)
    A_batch = cp.array([param[1] for param in parametros_batch], dtype=cp.float32)
    B_batch = cp.array([param[2] for param in parametros_batch], dtype=cp.float32)

    # Simular en paralelo
    simulaciones_validas = cp.ones(batch_size, dtype=cp.bool_)

    print(f'Numero de pasos a simular: {num_steps}')
    paso_explosion = cp.full(batch_size, -1, dtype=cp.int32)  # -1 significa no explotó
    
    for t in range(num_steps):
        # Llamar al kernel con todos los parámetros necesarios
        spread_infection_adi(
            S=S_batch, I=I_batch, R=R_batch, 
            S_new=S_new_batch, I_new=I_new_batch, R_new=R_new_batch,
            dt=dt, d=d, beta=beta_batch, gamma=gamma_batch,
            D=D_batch, wx=wx, wy=wy, 
            h_dx=h_dx_mapa, h_dy=h_dy_mapa, A=A_batch, B=B_batch, vegetacion=vegetacion
        )

        # Intercambiar arrays
        S_batch, S_new_batch = S_new_batch, S_batch
        I_batch, I_new_batch = I_new_batch, I_batch
        R_batch, R_new_batch = R_new_batch, R_batch
        
        # Verificar si alguna simulación explota
        # Condiciones más estrictas para detectar problemas temprano
        validas = cp.all((R_batch >= -1e-6) & (R_batch <= 1 + 1e-6) & 
                         (I_batch >= -1e-6) & (I_batch <= 1 + 1e-6) &
                         (S_batch >= -1e-6) & (S_batch <= 1 + 1e-6), axis=(1, 2))
        
        # Detectar valores extremos que pueden causar problemas
        valores_extremos = cp.any((R_batch > 10) | (R_batch < -10), axis=(1, 2))
        
        # Registrar el paso de explosión para simulaciones que acaban de explotar
        nuevas_explosiones = simulaciones_validas & (~validas | valores_extremos)
        paso_explosion = cp.where(nuevas_explosiones, t + 1, paso_explosion)
        
        # Actualizar estado de validez
        simulaciones_validas &= validas & (~valores_extremos)
        
        # Si todas las simulaciones explotaron, terminar
        if not cp.any(simulaciones_validas):
            print(f"Todas las simulaciones explotaron en el paso {t+1}")
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
        fitness_values.append(float(fitness_batch[i]))

    return fitness_values