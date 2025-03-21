import numpy as np
import cupy as cp # type: ignore
from modelo_rdc import spread_infection, courant
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from train_pinn import train_pinn

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

############################## CARGADO DE MAPAS ###############################################

# Ruta de los archivos
ruta_mapas = ['/home/lucas.becerra/Incendios/mapas_steffen_martin/ang_wind.asc',   # Dirección del viento
              '/home/lucas.becerra/Incendios/mapas_steffen_martin/speed_wind.asc', # Velocidad del viento
              '/home/lucas.becerra/Incendios/mapas_steffen_martin/asc_slope.asc',  # Pendiente del terreno
              '/home/lucas.becerra/Incendios/mapas_steffen_martin/asc_CIEFAP.asc', # Vegetación
              '/home/lucas.becerra/Incendios/mapas_steffen_martin/asc_aspect.asc'] # Orientación del terreno

# Función para leer archivos .asc
def leer_asc(ruta):
    with open(ruta, 'r') as f:
        for i in range(6):  
            f.readline()
        data = np.loadtxt(f) 
        return cp.array(data, dtype=cp.float32) 

# Leer los mapas
datos = [leer_asc(mapa) for mapa in ruta_mapas]

# Asignar cada matriz a una variable
vientod = datos[0]
vientov = datos[1]
pendiente = datos[2]
vegetacion = datos[3]
orientacion = datos[4]

############################## PARÁMETROS DE LOS MAPAS ###############################################

# Coeficiente de difusión
D_I = 50 # metros^2 / hora. Si la celda tiene 30 metros, en una hora avanza 1/3 del tamaño de la celda
d = 30 # Tamaño de cada celda
beta_veg = cp.where(vegetacion <= 2, 0, 0.1 * vegetacion) # Parámetros del modelo SI
gamma = cp.where(vegetacion <= 2, 100, 0.1) # Hacemos una máscara donde vegetación <=2, gamma >> 1/dt. Sino, vale 0.1. 
# Transformación del viento a coordenadas cartesianas
wx = vientov * cp.cos(5/2 * cp.pi - vientod * cp.pi / 180) * 1000
wy = - vientov * cp.sin(5/2 * cp.pi - vientod * cp.pi / 180) * 1000
# Cálculo de la pendiente (usando mapas de pendiente y orientación)
h_dx_mapa = cp.tan(pendiente * cp.pi / 180) * cp.cos(orientacion * cp.pi / 180 - cp.pi/2)
h_dy_mapa = cp.tan(pendiente * cp.pi / 180) * cp.sin(orientacion * cp.pi / 180 - cp.pi/2)
# Constante A adimensional
A = 5e-4 # 10^-3 está al doble del límite de estabilidad
# Constante B
B = 15 # m/h

############################## DE CUPY A TORCH ###############################################

# Convertir de CuPy a PyTorch directamente en GPU
def cupy_to_torch(cp_array):
    return torch.as_tensor(cp_array, device=device, dtype=torch.float32)

beta = cupy_to_torch(beta_veg)
wx = cupy_to_torch(wx)
wy = cupy_to_torch(wy)
gamma = cupy_to_torch(gamma)
h_dx = cupy_to_torch(h_dx_mapa)
h_dy = cupy_to_torch(h_dy_mapa)

############################## ENTRENAMIENTO DEL MODELO ###############################################

model = train_pinn(beta, gamma, D_I, A, wx, h_dx, B, wy, h_dy, epochs=10000)

############################## GUARDADO DEL MODELO ENTRENADO ###############################################

torch.save(model.state_dict(), "modelo_entrenado.pth")
