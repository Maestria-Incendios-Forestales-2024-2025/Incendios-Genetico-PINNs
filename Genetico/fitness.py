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
                if j < len(beta_params) and j < len(gamma_params) and veg_type != 5:
                    beta_map = cp.where(mask, beta_params[j], beta_map)
                    gamma_map = cp.where(mask, gamma_params[j], gamma_map)
                elif veg_type == 5:  # Tipo de vegetación 5
                    beta_map = cp.where(mask, 1.0, beta_map)
                    gamma_map = cp.where(mask, 0.5, gamma_map)
        else: # Exp1
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
                  ajustar_ignicion=True, ignicion_fija_x=None, ignicion_fija_y=None, debug=True):
    """
    Calcula el fitness para múltiples combinaciones de parámetros en paralelo.
    
    Args:
        parametros_batch: Lista de tuplas (D, A, B, x, y, beta_params, gamma_params)
        burnt_cells: mapa binario de celdas quemadas de referencia (ny, nx) en CPU o GPU
        num_steps: cantidad de pasos de simulación
        ajustar_beta_gamma: si se ajustan beta/gamma (si no, se usan beta_fijo/gamma_fijo)
        ajustar_ignicion: si se usan (x, y) por candidato (si no, ignición fija)
        debug: si True, imprime información de depuración (costoso por transferencias CPU-GPU)
    
    Returns:
        Lista de valores de fitness
    """
    
    # Resto del código original...
    batch_size = len(parametros_batch)

    if debug:
        print(f"[DEBUG] Batch size: {batch_size}")
    
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

    if debug:
        print(f"[DEBUG] Numero de pasos a simular: {num_steps}")
    paso_explosion = cp.full(batch_size, -1, dtype=cp.int32)  # -1 significa no explotó

    printed_all_exploded = False  # <-- flag para imprimir solo una vez

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
        
        # ✅ Solo imprime una vez si todas explotan (solo en modo debug)
        if not cp.any(simulaciones_validas):
            if debug and not printed_all_exploded:
                print(f"[DEBUG] Todas las simulaciones explotaron en el paso {t+1}")
                printed_all_exploded = True
            # Cortamos temprano si ya no queda ninguna simulación válida
            break

    # ==========================
    # DEBUG FINAL DE VALORES
    # ==========================
    if debug:
        print("\n[DEBUG] Resumen final de la simulación:")
        print(f"  - Simulaciones válidas restantes: {int(cp.sum(simulaciones_validas).get())}/{batch_size}")

        for name, arr in [("S", S_batch), ("I", I_batch), ("R", R_batch)]:
            print(f"  - {name}: min={float(cp.min(arr).get()):.4f}, "
                  f"max={float(cp.max(arr).get()):.4f}, "
                  f"mean={float(cp.mean(arr).get()):.4f}")

        # Traemos a host siempre para usarlo también en el detalle por simulación
        pasos_host = paso_explosion.get()
        if cp.any(paso_explosion >= 0):
            print(f"  - Simulaciones que explotaron: {int((pasos_host >= 0).sum())}")
            print(f"  - Primeras explosiones en pasos: {sorted(set(pasos_host[pasos_host >= 0]))[:5]}")

        # Detalle por simulación para detectar NaN y rangos por campo
        print("\n[DEBUG] Detalle por simulación (min/max y NaN):")
        # Cálculo vectorizado en GPU
        s_min = cp.min(S_batch, axis=(1, 2))
        s_max = cp.max(S_batch, axis=(1, 2))
        i_min = cp.min(I_batch, axis=(1, 2))
        i_max = cp.max(I_batch, axis=(1, 2))
        r_min = cp.min(R_batch, axis=(1, 2))
        r_max = cp.max(R_batch, axis=(1, 2))

        s_has_nan = cp.any(cp.isnan(S_batch), axis=(1, 2))
        i_has_nan = cp.any(cp.isnan(I_batch), axis=(1, 2))
        r_has_nan = cp.any(cp.isnan(R_batch), axis=(1, 2))

        # Transferencia a host en bloque
        s_min_h = s_min.get(); s_max_h = s_max.get()
        i_min_h = i_min.get(); i_max_h = i_max.get()
        r_min_h = r_min.get(); r_max_h = r_max.get()
        s_nan_h = s_has_nan.get(); i_nan_h = i_has_nan.get(); r_nan_h = r_has_nan.get()

        # Conteo de NaNs por campo
        print(f"  - Conteo NaN: S={int(s_nan_h.sum())}, I={int(i_nan_h.sum())}, R={int(r_nan_h.sum())}")
        for idx in range(batch_size):
            print(
                f"    Sim {idx:>3}: "
                f"S[min={s_min_h[idx]:.4g}, max={s_max_h[idx]:.4g}, nan={bool(s_nan_h[idx])}] | "
                f"I[min={i_min_h[idx]:.4g}, max={i_max_h[idx]:.4g}, nan={bool(i_nan_h[idx])}] | "
                f"R[min={r_min_h[idx]:.4g}, max={r_max_h[idx]:.4g}, nan={bool(r_nan_h[idx])}]"
            )

        # Parámetros de simulaciones problemáticas
        print("\n[DEBUG] Parámetros de simulaciones problemáticas (NaN o explosión):")
        any_printed = False
        for idx in range(batch_size):
            has_nan = bool(s_nan_h[idx] or i_nan_h[idx] or r_nan_h[idx])
            exploded = pasos_host[idx] >= 0
            if has_nan or exploded:
                params = parametros_batch[idx]
                # Intentamos formatear de manera amigable según la longitud de tupla
                try:
                    if isinstance(params, (list, tuple)):
                        if len(params) == 7:
                            Dp, Ap, Bp, xp, yp, betap, gammap = params
                            pstr = (f"D={float(Dp):.4g}, A={float(Ap):.4g}, B={float(Bp):.4g}, "
                                    f"x={int(xp)}, y={int(yp)}, beta={betap}, gamma={gammap}")
                        elif len(params) == 5 and ajustar_beta_gamma:
                            Dp, Ap, Bp, betas, gammas = params
                            # betas/gammas pueden ser iterables; convertimos a lista plana corta
                            try:
                                betas_list = list(betas)
                            except Exception:
                                betas_list = [betas]
                            try:
                                gammas_list = list(gammas)
                            except Exception:
                                gammas_list = [gammas]
                            pstr = (f"D={float(Dp):.4g}, A={float(Ap):.4g}, B={float(Bp):.4g}, "
                                    f"betas={betas_list}, gammas={gammas_list}")
                        else:
                            pstr = str(params)
                    else:
                        pstr = str(params)
                except Exception:
                    pstr = str(params)
                pe = int(pasos_host[idx]) if exploded else None
                print(f"    Sim {idx:>3}: exploded_step={pe} | NaN={has_nan} | {pstr}")
                any_printed = True
        if not any_printed:
            print("    (ninguna)")

    # Calcular fitness
    burnt_cells_sim_batch = cp.where(R_batch > 0.001, 1, 0)
    burnt_cells_expanded = cp.broadcast_to(burnt_cells[cp.newaxis, :, :], (batch_size, ny, nx))
    union_batch = cp.sum(burnt_cells_expanded | burnt_cells_sim_batch, axis=(1, 2))
    interseccion_batch = cp.sum(burnt_cells_expanded & burnt_cells_sim_batch, axis=(1, 2))
    burnt_cells_total = cp.sum(burnt_cells)
    fitness_batch = (union_batch - interseccion_batch) / burnt_cells_total

    fitness_values = [float(fitness_batch[i]) for i in range(batch_size)]

    if debug:
        print(f"[DEBUG] Fitness (primeros 5): {fitness_values[:5]}")
        print("[DEBUG] Fin de aptitud_batch.\n")

    return fitness_values