import numpy as np
import cupy as cp # type: ignore
from modelo_rdc import spread_infection, courant
import csv
import argparse
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
D = 50 # metros^2 / hora. Si la celda tiene 30 metros, en una hora avanza 1/3 del tamaño de la celda
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

############################## GRILLA DE ENTRENAMIENTO ###############################################

# Definir dimensiones espaciales y temporales
Ny, Nx = vientod.shape  # Resolución espacial
T_max = 1000         # Número de pasos de tiempo
dt = 1/6            # Paso temporal en horas

# Crear grilla de entrenamiento
t = torch.linspace(0, T_max * dt, T_max).to(device)  # Tiempo en horas
x = torch.linspace(0, Nx, Nx).to(device)  
y = torch.linspace(0, Ny, Ny).to(device) 

T, X, Y = torch.meshgrid(t, x, y, indexing='ij')  # Grilla 3D en GPU

# Aplanar para pasarlo a la red
t_train = T.reshape(-1, 1)
x_train = X.reshape(-1, 1)
y_train = Y.reshape(-1, 1)

# Concatenar en un solo tensor (entrada de la red)
train_points = torch.cat([t_train, x_train, y_train], dim=1)

############################## CONDICIONES INICIALES ###############################################

# Mapas iniciales (condiciones iniciales en t=0)
I0 = torch.zeros((Nx, Ny), device="cuda")  # Todo vacío excepto punto de ignición
I0[700, 700] = 1.0  # Ejemplo de ignición en (500, 300)

S0 = torch.ones((Nx, Ny), device="cuda") - I0  # Todo combustible menos ignición
R0 = torch.zeros((Nx, Ny), device="cuda")  # Nada "recuperado" aún

