import numpy as np
import torch # type: ignore
from train_pinn import train_pinn

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

############################## PARÁMETROS DE LOS MAPAS ###############################################

D_I = 0.005
beta_val = 0.3
gamma_val = 0.1
epochs_adam = 10000

############################## ENTRENAMIENTO DEL MODELO ###############################################

start_time = torch.cuda.Event(enable_timing=True)
end_time = torch.cuda.Event(enable_timing=True)

start_time.record()
model, loss = train_pinn(D_I, beta_val, gamma_val, epochs_adam=epochs_adam)
end_time.record()

# Sincronizar para asegurar medición correcta
torch.cuda.synchronize()
training_time = start_time.elapsed_time(end_time) / 1000  # Convertir de ms a s

print(f"Training time: {training_time:.2f} seconds")

############################## GUARDADO DEL MODELO ENTRENADO ###############################################

torch.save(model.state_dict(), f"modelo_entrenado_pinn_2d_{epochs_adam}.pth")
