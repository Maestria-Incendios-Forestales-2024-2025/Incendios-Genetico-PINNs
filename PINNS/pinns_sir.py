import numpy as np
import torch # type: ignore
from train_pinn import train_pinn

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

############################## PARÁMETROS DE LOS MAPAS ###############################################

# Parámetros que uso para resolver numéricamente
# Tamaño de cada celda
d = 30 # metros
# Coeficiente de difusión
D = 50 # metros^2/h
# Paso temporal
dt = 1/6 # horas

D_I = D * dt / (d**2)
beta_val = 0.3 * dt
gamma_val = 0.1 * dt
epochs_adam = 1000

Nx, Ny = 1768, 1060
mean_x, mean_y = 700 / Nx, 700 / Ny
sigma_x, sigma_y = 10 / Nx, 10 / Ny

############################## ENTRENAMIENTO DEL MODELO ###############################################

start_time = torch.cuda.Event(enable_timing=True)
end_time = torch.cuda.Event(enable_timing=True)

start_time.record()
model, loss = train_pinn(D_I, beta_val, gamma_val, mean_x, mean_y, sigma_x, sigma_y, epochs_adam=epochs_adam)
end_time.record()

# Sincronizar para asegurar medición correcta
torch.cuda.synchronize()
training_time = start_time.elapsed_time(end_time) / 1000  # Convertir de ms a s

print(f"Training time: {training_time:.2f} seconds")

############################## GUARDADO DEL MODELO ENTRENADO ###############################################

torch.save(model.state_dict(), f"modelo_entrenado_pinn_2d_{epochs_adam}.pth")
